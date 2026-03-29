#!/usr/bin/env python3
"""
WikiText-2 子集 PPL：FP16 vs 朴素 Q4_0 vs AWQ-Q4_0（与 export_int4.py / export_int4_awq.py 同一套量化定义）。

用于简历/报告的可复现数字：同一模型、同一 eval token 数、仅差「是否 AWQ 缩放后再 Q4_0」。

用法（在仓库根目录）:
  python tools/export_qwen3/eval_awq_ppl.py --model_dir /path/to/Qwen3-8B \\
      --max_tokens 8192 --chunk_len 2048 --cache_awq /tmp/awq_scales.pt

快速试跑（租卡省时间）:
  --modes naive,awq --max_tokens 1024 --calibration builtin
  或 --wikitext_calib_samples 24（仍用 wikitext2 校准时减少 forward 次数）

首次含 awq 模式会跑校准+网格搜索；指定 --cache_awq 可写入/读取缓存。
依赖: transformers, torch, datasets（与 export_int4_awq 一致）
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from export_int4_awq import (  # noqa: E402
    collect_activation_scales,
    grid_search_alpha,
    load_safetensors,
)


def q40_roundtrip_dequant(w: torch.Tensor, group_size: int) -> torch.Tensor:
    """与 export 中 Q4_0 一致的 round-trip：量化再反量化，返回 FP32。"""
    w = w.float()
    orig_shape = w.shape
    flat = w.reshape(-1, group_size)
    wmax = torch.abs(flat).max(dim=1).values
    scale = (wmax / 7.0).clamp(min=1e-10)
    quant = (flat / scale[:, None]).round().clamp(-8, 7)
    deq = (quant * scale[:, None]).reshape(orig_shape)
    return deq


def awq_effective_weight(
    w: torch.Tensor, awq_scale: torch.Tensor, group_size: int
) -> torch.Tensor:
    """
    等价于推理时 (x * (1/s)) @ dequant(Q(w*s))^T 的合并 Linear 权重：
    W_eff[i,j] = W_dq[i,j] / s[j]，前向仍为 y = x @ W_eff.T
    """
    s = awq_scale.float().clamp(min=1e-10)
    w_scaled = w * s.unsqueeze(0)
    w_dq = q40_roundtrip_dequant(w_scaled, group_size)
    inv = 1.0 / s
    return w_dq * inv.unsqueeze(0)


def effective_weight_for_layer(
    w: torch.Tensor,
    short_name: str,
    awq_results: dict,
    group_size: int,
) -> torch.Tensor:
    r = awq_results.get(short_name)
    if r is None or float(r.get("alpha", 0.0)) <= 0.0:
        return q40_roundtrip_dequant(w, group_size)
    return awq_effective_weight(w, r["scale"], group_size)


def build_layer_name_map(n_layers: int) -> dict[str, str]:
    m: dict[str, str] = {}
    for i in range(n_layers):
        m[f"wq.{i}"] = f"model.layers.{i}.self_attn.q_proj"
        m[f"wk.{i}"] = f"model.layers.{i}.self_attn.k_proj"
        m[f"wv.{i}"] = f"model.layers.{i}.self_attn.v_proj"
        m[f"wo.{i}"] = f"model.layers.{i}.self_attn.o_proj"
        m[f"w1.{i}"] = f"model.layers.{i}.mlp.gate_proj"
        m[f"w2.{i}"] = f"model.layers.{i}.mlp.down_proj"
        m[f"w3.{i}"] = f"model.layers.{i}.mlp.up_proj"
    m["lm_head"] = "lm_head"
    return m


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    mod = model
    for part in name.split("."):
        mod = getattr(mod, part)
    return mod


def untie_word_embeddings_if_needed(model: nn.Module) -> None:
    """
    C++ 导出里 embed 保持 FP32、lm_head 走 Q4_0；HF 若 tie_word_embeddings，
    lm_head 与 embed 共用一个 Parameter，只量化 lm_head 会误伤 embedding。
    """
    cfg = model.config
    if not getattr(cfg, "tie_word_embeddings", False):
        return
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()
    if input_emb is None or output_emb is None:
        return
    wi = input_emb.weight.data.clone()
    wo = output_emb.weight.data.clone()
    input_emb.weight = nn.Parameter(wi)
    output_emb.weight = nn.Parameter(wo)
    cfg.tie_word_embeddings = False


def replace_quantized_linears(
    model: nn.Module,
    sd: dict,
    layer_name_map: dict[str, str],
    group_size: int,
    mode: str,
    awq_results: dict | None,
    dtype: torch.dtype,
) -> None:
    device = next(model.parameters()).device
    for short, hf_prefix in layer_name_map.items():
        key = f"{hf_prefix}.weight"
        if key not in sd:
            continue
        w = sd[key].float()
        if mode == "fp16":
            continue
        if mode == "naive":
            w_new = q40_roundtrip_dequant(w, group_size)
        elif mode == "awq":
            assert awq_results is not None
            w_new = effective_weight_for_layer(w, short, awq_results, group_size)
        else:
            raise ValueError(mode)
        linear = get_module_by_name(model, hf_prefix)
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"{hf_prefix} is not Linear")
        linear.weight.data = w_new.to(device=device, dtype=dtype)


def compute_awq_results(
    model_dir: Path,
    device: str,
    calibration: str,
    group_size: int,
    n_layers: int,
    wikitext_calib_samples: int,
) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    act_scales, act_samples = collect_activation_scales(
        model,
        tokenizer,
        device,
        calibration=calibration,
        wikitext_n_samples=wikitext_calib_samples,
    )
    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    sd = load_safetensors(model_dir)
    layer_name_map = build_layer_name_map(n_layers)
    awq_results: dict = {}
    for short_name, hf_name in layer_name_map.items():
        w_key = f"{hf_name}.weight"
        if w_key not in sd:
            continue
        w = sd[w_key].float()
        if hf_name not in act_scales:
            awq_results[short_name] = {"alpha": 0.0, "scale": torch.ones(w.shape[1])}
            continue
        a_scale = act_scales[hf_name].cpu().float()
        a_samples = act_samples.get(hf_name)
        if a_samples is None or a_samples.shape[0] < 2:
            awq_results[short_name] = {"alpha": 0.0, "scale": torch.ones(w.shape[1])}
            continue
        best_alpha, best_scale, _, _ = grid_search_alpha(
            w.cpu(), a_samples, a_scale, group_size
        )
        awq_results[short_name] = {"alpha": best_alpha, "scale": best_scale.cpu()}
    return awq_results


def load_or_compute_awq_results(
    model_dir: Path,
    device: str,
    calibration: str,
    group_size: int,
    n_layers: int,
    cache_path: Path | None,
    recompute: bool,
    wikitext_calib_samples: int,
) -> dict:
    if cache_path and cache_path.is_file() and not recompute:
        blob = torch.load(cache_path, map_location="cpu")
        return blob["awq_results"]
    awq_results = compute_awq_results(
        model_dir,
        device,
        calibration,
        group_size,
        n_layers,
        wikitext_calib_samples,
    )
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "awq_results": awq_results,
                "group_size": group_size,
                "calibration": calibration,
                "wikitext_calib_samples": wikitext_calib_samples,
                "model_dir": str(model_dir),
            },
            cache_path,
        )
        print(f"  Saved AWQ scales cache: {cache_path}")
    return awq_results


def tokenize_wikitext_upto(
    tokenizer, ds_split, max_tokens: int
) -> torch.Tensor:
    """按行增量编码，凑满 max_tokens，避免整集拼成超长串再 tokenize。"""
    ids: list[int] = []
    for row in ds_split["text"]:
        text = row.strip() if isinstance(row, str) else ""
        if not text:
            continue
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        tid = enc.input_ids[0]
        take = min(tid.numel(), max_tokens - len(ids))
        if take > 0:
            ids.extend(tid[:take].tolist())
        if len(ids) >= max_tokens:
            break
    return torch.tensor([ids], dtype=torch.long)


@torch.inference_mode()
def eval_nll_chunks(
    model: nn.Module,
    input_ids: torch.Tensor,
    chunk_len: int,
    device: str,
) -> tuple[float, int]:
    """对 [1, T] 的 token，按 chunk 前向，返回 (sum_nll, n_tokens)。"""
    model.eval()
    total_ce = 0.0
    total_tok = 0
    T = input_ids.size(1)
    for start in range(0, T - 1, chunk_len):
        end = min(start + chunk_len, T)
        chunk = input_ids[:, start:end].to(device)
        if chunk.size(1) < 2:
            break
        out = model(chunk, labels=chunk)
        # HF CausalLM: loss is mean over non-masked positions
        n = chunk.numel() - 1  # shift 后有效预测数
        total_ce += out.loss.item() * n
        total_tok += n
    return total_ce, total_tok


def main():
    p = argparse.ArgumentParser(description="WikiText-2 subset PPL: FP16 vs naive Q4_0 vs AWQ")
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--modes", type=str, default="fp16,naive,awq",
                   help="Comma-separated: fp16,naive,awq")
    p.add_argument("--max_tokens", type=int, default=8192,
                   help="Eval 用的前 N 个 token（子集，跑得更快）")
    p.add_argument("--chunk_len", type=int, default=2048)
    p.add_argument("--wikitext_split", type=str, default="test",
                   choices=["test", "validation"])
    p.add_argument("--group_size", type=int, default=64)
    p.add_argument("--calibration", type=str, default="wikitext2",
                   choices=["builtin", "wikitext2"])
    p.add_argument(
        "--wikitext_calib_samples",
        type=int,
        default=128,
        help="calibration=wikitext2 时抽样条数，越小越快（如 16/32）；与完整导出默认 128 可不一致",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--cache_awq", type=str, default=None,
                   help="读写 AWQ 标定结果 (.pt)，避免重复跑 hooks")
    p.add_argument("--recompute_awq", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model_dir = Path(args.model_dir)
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)
    n_layers = cfg["num_hidden_layers"]
    layer_name_map = build_layer_name_map(n_layers)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if "awq" in modes:
        cache = Path(args.cache_awq) if args.cache_awq else None
        awq_results = load_or_compute_awq_results(
            model_dir,
            args.device,
            args.calibration,
            args.group_size,
            n_layers,
            cache,
            args.recompute_awq,
            args.wikitext_calib_samples,
        )
    else:
        awq_results = None

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n=== Loading WikiText-2 ({args.wikitext_split}) ===")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=args.wikitext_split)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    ids = tokenize_wikitext_upto(tokenizer, ds, args.max_tokens)
    print(f"  Tokenized length: {ids.size(1)}")

    sd = load_safetensors(model_dir)
    results: dict[str, tuple[float, float]] = {}

    for mode in modes:
        print(f"\n=== Mode: {mode} ===")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.float16,
            device_map=args.device,
            trust_remote_code=True,
        )
        if mode != "fp16":
            untie_word_embeddings_if_needed(model)
            replace_quantized_linears(
                model,
                sd,
                layer_name_map,
                args.group_size,
                mode,
                awq_results if mode == "awq" else None,
                torch.float16,
            )
        ce_sum, n_tok = eval_nll_chunks(model, ids, args.chunk_len, args.device)
        nll = ce_sum / n_tok
        ppl = math.exp(nll)
        results[mode] = (nll, ppl)
        print(f"  NLL (mean): {nll:.4f}")
        print(f"  PPL:        {ppl:.2f}")
        del model
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    print("\n=== Summary (同一 eval 子集、与导出脚本一致的 Q4_0) ===")
    print(f"| 模式 | NLL | PPL |")
    print(f"|------|-----|-----|")
    for mode in modes:
        nll, ppl = results[mode]
        print(f"| {mode} | {nll:.4f} | {ppl:.2f} |")
    if "fp16" in results and "naive" in results:
        dn = results["naive"][0] - results["fp16"][0]
        print(f"\n  ΔNLL(naive - fp16): {dn:+.4f}")
    if "fp16" in results and "awq" in results:
        da = results["awq"][0] - results["fp16"][0]
        print(f"  ΔNLL(awq - fp16):   {da:+.4f}")
    if "naive" in results and "awq" in results:
        d = results["naive"][0] - results["awq"][0]
        pct = (math.exp(results["naive"][0]) / math.exp(results["awq"][0]) - 1.0) * 100
        print(f"  ΔNLL(naive - awq):  {d:+.4f}  (AWQ 更低更好；约 PPL 相对差 {pct:+.1f}%)")


if __name__ == "__main__":
    main()
