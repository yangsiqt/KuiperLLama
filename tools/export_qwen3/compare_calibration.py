"""
Compare AWQ calibration quality: builtin 8 texts vs WikiText-2.
Memory-efficient: caps activation samples at 64 per layer during collection.
"""
import argparse
import json
import gc
from pathlib import Path

import torch
from export_int4_awq import (
    _BUILTIN_CALIBRATION_TEXTS,
    _load_wikitext2_calibration,
    grid_search_alpha,
    load_safetensors,
    quantize_q40,
)

MAX_SAMPLES_PER_LAYER = 64


def collect_for_texts(model, tokenizer, device, texts):
    act_scales = {}
    act_samples = {}
    hooks = []

    def make_hook(name):
        def fn(module, inp, out):
            x = inp[0].detach().float()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            scale = x.abs().mean(dim=0)
            act_scales[name] = (act_scales[name] + scale) / 2.0 if name in act_scales else scale

            x_cpu = x.cpu()
            if name in act_samples:
                cur = act_samples[name]
                if cur.shape[0] < MAX_SAMPLES_PER_LAYER:
                    remain = MAX_SAMPLES_PER_LAYER - cur.shape[0]
                    act_samples[name] = torch.cat([cur, x_cpu[:remain]], dim=0)
            else:
                act_samples[name] = x_cpu[:MAX_SAMPLES_PER_LAYER]
        return fn

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for i, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model(**inputs)
            if (i + 1) % 10 == 0 or i == len(texts) - 1:
                print(f"    {i+1}/{len(texts)}", flush=True)
    print()

    for h in hooks:
        h.remove()
    return act_scales, act_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="/home/dzy/models/Qwen3-8B")
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--wiki_samples", type=int, default=64)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)

    n_layers = min(args.layers, cfg["num_hidden_layers"])
    group_size = args.group_size

    layer_map = {}
    for i in range(n_layers):
        layer_map[f"wq.{i}"] = f"model.layers.{i}.self_attn.q_proj"
        layer_map[f"wk.{i}"] = f"model.layers.{i}.self_attn.k_proj"
        layer_map[f"wv.{i}"] = f"model.layers.{i}.self_attn.v_proj"
        layer_map[f"wo.{i}"] = f"model.layers.{i}.self_attn.o_proj"
        layer_map[f"w1.{i}"] = f"model.layers.{i}.mlp.gate_proj"
        layer_map[f"w2.{i}"] = f"model.layers.{i}.mlp.down_proj"
        layer_map[f"w3.{i}"] = f"model.layers.{i}.mlp.up_proj"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model ({model_dir.name}) in FP16 on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.float16, device_map=args.device, trust_remote_code=True
    )

    print(f"\n--- Builtin calibration (8 texts) ---")
    act_scales_b, act_samples_b = collect_for_texts(model, tokenizer, args.device, _BUILTIN_CALIBRATION_TEXTS)

    print(f"\n--- WikiText-2 calibration ({args.wiki_samples} texts) ---")
    wiki_texts = _load_wikitext2_calibration(n_samples=args.wiki_samples)
    act_scales_w, act_samples_w = collect_for_texts(model, tokenizer, args.device, wiki_texts)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Model released. Loading FP32 weights from safetensors...")
    sd = load_safetensors(model_dir)

    print(f"\n{'Layer':<10} {'Naive':>10} {'Builtin':>10} {'Wiki':>10} {'B-imp%':>8} {'W-imp%':>8} {'Winner':>8}")
    print("-" * 68)

    total_naive = total_b = total_w = 0.0
    count = 0

    for short_name, hf_name in layer_map.items():
        w_key = f"{hf_name}.weight"
        if w_key not in sd:
            continue
        w = sd[w_key].float()

        a_scale_b = act_scales_b.get(hf_name)
        a_samp_b  = act_samples_b.get(hf_name)
        a_scale_w = act_scales_w.get(hf_name)
        a_samp_w  = act_samples_w.get(hf_name)

        if any(x is None for x in [a_scale_b, a_samp_b, a_scale_w, a_samp_w]):
            continue

        _, _, naive_err = quantize_q40(w, group_size)
        _, _, b_err, b_base = grid_search_alpha(w.cpu(), a_samp_b.float(), a_scale_b.cpu().float(), group_size)
        _, _, w_err, w_base = grid_search_alpha(w.cpu(), a_samp_w.float(), a_scale_w.cpu().float(), group_size)

        b_imp = (1.0 - b_err / max(b_base, 1e-10)) * 100
        w_imp = (1.0 - w_err / max(w_base, 1e-10)) * 100
        winner = "wiki" if w_err < b_err else "builtin"

        print(f"{short_name:<10} {naive_err:>10.6f} {b_err:>10.6f} {w_err:>10.6f} {b_imp:>7.1f}% {w_imp:>7.1f}% {winner:>8}")

        total_naive += naive_err
        total_b += b_err
        total_w += w_err
        count += 1

    if count > 0:
        print("-" * 68)
        avg_n = total_naive / count
        avg_b = total_b / count
        avg_w = total_w / count
        b_pct = (1.0 - avg_b / avg_n) * 100
        w_pct = (1.0 - avg_w / avg_n) * 100
        print(f"{'AVERAGE':<10} {avg_n:>10.6f} {avg_b:>10.6f} {avg_w:>10.6f} {b_pct:>7.1f}% {w_pct:>7.1f}%")
        print(f"\n==> Builtin  avg error reduction vs naive: {b_pct:+.2f}%")
        print(f"==> WikiText avg error reduction vs naive: {w_pct:+.2f}%")
        diff = w_pct - b_pct
        if diff > 0:
            print(f"==> WikiText-2 is BETTER by {diff:.2f}pp")
        elif diff < 0:
            print(f"==> Builtin is BETTER by {-diff:.2f}pp")
        else:
            print(f"==> Both are equivalent")


if __name__ == "__main__":
    main()
