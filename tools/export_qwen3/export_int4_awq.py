"""
Qwen3 INT4 AWQ (Activation-Aware Weight Quantization) Export

Algorithm:
1. Load HF model in FP16 on GPU for calibration
2. Collect per-channel activation scales via forward hooks
3. Grid search optimal alpha per layer to minimize quantization error
4. Apply AWQ scaling to weights, quantize to INT4, export

File format (C++ compatible):
  [header: 8 int32 (ModelConfig)]
  [group_size: int32]
  [awq_flag: int32]  <-- 0=no AWQ, 1=AWQ
  Per quantized layer (in order: WQ*L, WK*L, WV*L, WO*L, W1*L, W2*L, W3*L, CLS):
    [packed_int4_weights: uint8, dim0 * dim1 / 2 bytes]
    [group_scales: float32, dim0 * dim1 / group_size floats]
    [awq_input_scales: float32, dim1 floats]  <-- 1/s, applied to input at runtime
  FP32 section:
    [embed_tokens: float32]
    [RMSNorm weights: attention(L) + ffn(L) + final(1)]
    [Q norm(L)]
    [K norm(L)]
"""
import struct
import argparse
import json
import gc
import sys
import builtins
from pathlib import Path

import torch
import numpy as np
from safetensors import safe_open

_orig_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _orig_print(*args, **kwargs)


def quantize_q40(w, group_size):
    """Q4_0 symmetric quantization: float -> int4 [-8, 7], packed 2 per byte."""
    assert w.numel() % group_size == 0
    w = w.float().reshape(-1, group_size)
    wmax = torch.abs(w).max(dim=1).values
    scale = wmax / 7.0
    scale = scale.clamp(min=1e-10)
    quant = (w / scale[:, None]).round().clamp(-8, 7)

    fp32val = (quant * scale[:, None]).reshape(-1)
    err = torch.abs(fp32val - w.reshape(-1)).max().item()

    int4val = (quant.to(torch.int8) + 8).to(torch.uint8)
    int4_flat = int4val.reshape(-1)
    low = int4_flat[0::2]
    high = int4_flat[1::2]
    packed = (high << 4) | low

    return packed, scale.float(), err


def quantize_q40_with_awq(w, awq_scale, group_size):
    """
    AWQ-enhanced Q4_0 quantization.
    w: (out_features, in_features) original FP32 weight
    awq_scale: (in_features,) per-input-channel scaling factor s
    Returns: packed_bytes, quant_scales, inv_awq_scale (1/s), max_error
    """
    w_scaled = w.float() * awq_scale.float().unsqueeze(0)
    packed, quant_scales, err = quantize_q40(w_scaled, group_size)
    inv_awq_scale = (1.0 / awq_scale.float()).contiguous()
    return packed, quant_scales, inv_awq_scale, err


def compute_awq_error(w, act_samples, awq_scale, group_size):
    """
    Compute quantization error WITH AWQ scaling.
    w: (out_features, in_features)
    act_samples: (n_samples, in_features) calibration activations
    awq_scale: (in_features,) scaling factor s
    Returns: MSE of output
    """
    w = w.float()
    act = act_samples.float()
    ref_out = act @ w.t()

    w_scaled = w * awq_scale.unsqueeze(0)
    packed, quant_scales, _ = quantize_q40(w_scaled, group_size)

    int4_flat_lo = (packed & 0x0F).to(torch.int8) - 8
    int4_flat_hi = ((packed >> 4) & 0x0F).to(torch.int8) - 8

    n_groups = w_scaled.numel() // group_size
    w_dq = torch.zeros_like(w_scaled.reshape(-1))
    for g in range(n_groups):
        start = g * group_size
        s = quant_scales[g].item()
        for k in range(group_size // 2):
            byte_idx = (start // 2) + k
            w_dq[start + 2 * k] = int4_flat_lo[byte_idx].float() * s
            w_dq[start + 2 * k + 1] = int4_flat_hi[byte_idx].float() * s
    w_dq = w_dq.reshape(w.shape)

    inv_s = 1.0 / awq_scale
    act_scaled = act * inv_s.unsqueeze(0)
    awq_out = act_scaled @ w_dq.t()

    mse = ((ref_out - awq_out) ** 2).mean().item()
    return mse


def compute_awq_error_fast(w, act_samples, awq_scale, group_size):
    """Fast vectorized AWQ error computation."""
    w = w.float()
    act = act_samples.float()
    ref_out = act @ w.t()

    w_scaled = w * awq_scale.unsqueeze(0)

    w_flat = w_scaled.reshape(-1, group_size)
    wmax = torch.abs(w_flat).max(dim=1).values
    scale = (wmax / 7.0).clamp(min=1e-10)
    quant = (w_flat / scale[:, None]).round().clamp(-8, 7)
    w_dq_flat = quant * scale[:, None]
    w_dq = w_dq_flat.reshape(w.shape)

    inv_s = 1.0 / awq_scale
    act_scaled = act * inv_s.unsqueeze(0)
    awq_out = act_scaled @ w_dq.t()

    mse = ((ref_out - awq_out) ** 2).mean().item()
    return mse


def grid_search_alpha(w, act_samples, act_scale, group_size, alphas=None):
    """
    Grid search for optimal alpha.
    s = act_scale^alpha, alpha in [0, 1]
    Returns: best_alpha, best_scale, best_error
    """
    if alphas is None:
        alphas = [i * 0.1 for i in range(11)]

    best_alpha = 0.0
    best_error = float('inf')
    best_scale = torch.ones_like(act_scale)

    baseline_s = torch.ones_like(act_scale)
    baseline_err = compute_awq_error_fast(w, act_samples, baseline_s, group_size)

    for alpha in alphas:
        if alpha == 0.0:
            s = torch.ones_like(act_scale)
        else:
            s = act_scale.float().clamp(min=1e-5).pow(alpha)
            s = s / s.mean() * 1.0

        err = compute_awq_error_fast(w, act_samples, s, group_size)
        if err < best_error:
            best_error = err
            best_alpha = alpha
            best_scale = s.clone()

    return best_alpha, best_scale, best_error, baseline_err


def _load_wikitext2_calibration(n_samples=128, min_words=20):
    """Load calibration texts from WikiText-2 dataset."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if t.strip() and len(t.split()) >= min_words]
    if len(texts) > n_samples:
        import random
        random.seed(42)
        texts = random.sample(texts, n_samples)
    return texts


_BUILTIN_CALIBRATION_TEXTS = [
    "The history of artificial intelligence began in the 1950s when researchers first explored the concept of machine intelligence. The Dartmouth Conference of 1956 is widely considered the founding event of AI as a field.",
    "Large language models like GPT and LLaMA use the transformer architecture with self-attention mechanisms to process and generate text. These models are trained on massive datasets.",
    "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。深度学习是目前最流行的方法之一。",
    "量化技术通过降低模型权重的数值精度来减少内存占用和计算开销。常见的方法包括INT8量化、INT4量化和混合精度推理。",
    "Neural networks consist of layers of interconnected nodes that process information. Each connection has a weight that is adjusted during training to minimize the loss function.",
    "The transformer architecture introduced in 2017 revolutionized natural language processing. Key innovations include multi-head self-attention and positional encoding.",
    "推理引擎的性能优化包括算子融合、内存池管理、KV缓存优化、以及利用GPU的并行计算能力进行高效的矩阵乘法运算。",
    "Modern GPU architectures like NVIDIA's Ampere and Hopper provide dedicated tensor cores that accelerate matrix multiplication operations commonly used in deep learning workloads.",
]


def collect_activation_scales(model, tokenizer, device, n_samples=8, calibration="builtin"):
    """
    Collect per-channel activation scales for each linear layer
    by running calibration data through the model.
    calibration: "builtin" (8 hardcoded texts) or "wikitext2" (128 samples from WikiText-2)
    """
    if calibration == "wikitext2":
        calibration_texts = _load_wikitext2_calibration(n_samples=128)
        print(f"  Using WikiText-2 calibration: {len(calibration_texts)} samples")
    else:
        calibration_texts = _BUILTIN_CALIBRATION_TEXTS
        print(f"  Using builtin calibration: {len(calibration_texts)} samples")

    act_scales = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            x = input[0].detach().float()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            scale = x.abs().mean(dim=0)
            if name in act_scales:
                act_scales[name] = (act_scales[name] + scale) / 2.0
            else:
                act_scales[name] = scale
        return hook_fn

    act_samples = {}
    # Cap rows per layer during collection so WikiText-2 (128 texts) does not OOM RAM.
    max_act_rows = 64

    def make_sample_hook(name):
        def hook_fn(module, input, output):
            x = input[0].detach().float()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            x_cpu = x.cpu()
            if name in act_samples:
                cur = act_samples[name]
                if cur.shape[0] < max_act_rows:
                    remain = max_act_rows - cur.shape[0]
                    act_samples[name] = torch.cat([cur, x_cpu[:remain]], dim=0)
            else:
                act_samples[name] = x_cpu[:max_act_rows]
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))
            hooks.append(module.register_forward_hook(make_sample_hook(name)))

    print(f"  Running {len(calibration_texts)} calibration samples...")
    model.eval()
    with torch.no_grad():
        for i, text in enumerate(calibration_texts):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model(**inputs)
            print(f"    Sample {i+1}/{len(calibration_texts)} done")

    for h in hooks:
        h.remove()

    for k in act_samples:
        if act_samples[k].shape[0] > 64:
            perm = torch.randperm(act_samples[k].shape[0])[:64]
            act_samples[k] = act_samples[k][perm]

    return act_scales, act_samples


def serialize_fp32(f, tensor):
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    f.write(struct.pack(f'{len(d)}f', *d))


def serialize_uint8(f, tensor):
    d = tensor.detach().cpu().view(-1).numpy().astype(np.uint8)
    f.write(d.tobytes())


def load_safetensors(model_dir):
    model_dir = Path(model_dir)
    state_dict = {}
    for st_file in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3 INT4 AWQ .bin")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--calibration", type=str, default="wikitext2",
                        choices=["builtin", "wikitext2"],
                        help="Calibration data: 'builtin' (8 hardcoded) or 'wikitext2' (128 samples)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    config_file = model_dir / "config.json"
    with open(config_file) as f:
        cfg = json.load(f)

    n_layers = cfg["num_hidden_layers"]
    hidden_size = cfg["hidden_size"]
    intermediate_size = cfg["intermediate_size"]
    num_attention_heads = cfg["num_attention_heads"]
    num_kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim", hidden_size // num_attention_heads)
    vocab_size = cfg["vocab_size"]
    max_seq_len = cfg.get("max_position_embeddings", 40960)

    dim = num_attention_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    print(f"Model: {model_dir.name}")
    print(f"  layers={n_layers}, hidden={hidden_size}, dim={dim}, kv_dim={kv_dim}")
    print(f"  intermediate={intermediate_size}, vocab={vocab_size}")
    print(f"  group_size={args.group_size}")

    # === Step 1: Load model in FP16 for calibration ===
    print("\n=== Step 1: Loading model in FP16 for calibration ===")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    print(f"  Model loaded on {device}")

    # === Step 2: Collect activation scales ===
    print("\n=== Step 2: Collecting activation scales ===")
    act_scales, act_samples = collect_activation_scales(model, tokenizer, device,
                                                         calibration=args.calibration)
    print(f"  Collected scales for {len(act_scales)} layers")

    # === Step 3: Grid search for optimal alpha per layer ===
    print("\n=== Step 3: Grid search for optimal alpha ===")

    layer_name_map = {}
    for i in range(n_layers):
        layer_name_map[f"wq.{i}"] = f"model.layers.{i}.self_attn.q_proj"
        layer_name_map[f"wk.{i}"] = f"model.layers.{i}.self_attn.k_proj"
        layer_name_map[f"wv.{i}"] = f"model.layers.{i}.self_attn.v_proj"
        layer_name_map[f"wo.{i}"] = f"model.layers.{i}.self_attn.o_proj"
        layer_name_map[f"w1.{i}"] = f"model.layers.{i}.mlp.gate_proj"
        layer_name_map[f"w2.{i}"] = f"model.layers.{i}.mlp.down_proj"
        layer_name_map[f"w3.{i}"] = f"model.layers.{i}.mlp.up_proj"
    layer_name_map["lm_head"] = "lm_head"

    awq_results = {}
    group_size = args.group_size

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("  Released FP16 model from GPU")

    print("\n  Loading FP32 weights from safetensors for grid search...")
    sd = load_safetensors(model_dir)

    for short_name, hf_name in layer_name_map.items():
        w_key = f"{hf_name}.weight"
        if w_key not in sd:
            print(f"  [SKIP] {short_name}: weight not found")
            continue

        w = sd[w_key].float()
        hf_linear_name = hf_name

        if hf_linear_name not in act_scales:
            print(f"  [SKIP] {short_name}: no activation data")
            awq_results[short_name] = {
                'alpha': 0.0,
                'scale': torch.ones(w.shape[1]),
                'improvement': 0.0
            }
            continue

        a_scale = act_scales[hf_linear_name].cpu().float()
        a_samples = act_samples.get(hf_linear_name, None)

        if a_samples is None or a_samples.shape[0] < 2:
            print(f"  [SKIP] {short_name}: insufficient activation samples")
            awq_results[short_name] = {
                'alpha': 0.0,
                'scale': torch.ones(w.shape[1]),
                'improvement': 0.0
            }
            continue

        best_alpha, best_scale, best_err, baseline_err = grid_search_alpha(
            w.cpu(), a_samples, a_scale, group_size
        )

        improvement = (1.0 - best_err / max(baseline_err, 1e-10)) * 100
        awq_results[short_name] = {
            'alpha': best_alpha,
            'scale': best_scale,
            'improvement': improvement
        }
        print(f"  {short_name}: alpha={best_alpha:.2f}, "
              f"error {baseline_err:.6f} -> {best_err:.6f} ({improvement:+.1f}%)")

    # === Step 4: Export with AWQ ===
    print("\n=== Step 4: Exporting AWQ INT4 model ===")
    output_path = args.output or f"Qwen3-{model_dir.name}-int4-awq.bin"
    out = open(output_path, 'wb')

    header = struct.pack('iiiiiiii',
                         dim, hidden_size, n_layers, num_attention_heads,
                         num_kv_heads, vocab_size, max_seq_len, intermediate_size)
    out.write(header)
    out.write(struct.pack('i', group_size))
    out.write(struct.pack('i', 1))  # awq_flag = 1

    total_err_awq = 0.0
    total_err_naive = 0.0
    n_quantized = 0

    def get(name):
        return sd[name]

    def write_awq_quantized(short_name, hf_weight_name, tensor):
        nonlocal total_err_awq, total_err_naive, n_quantized

        _, _, naive_err = quantize_q40(tensor, group_size)
        total_err_naive += naive_err

        result = awq_results.get(short_name)
        if result is not None and result['alpha'] > 0:
            awq_scale = result['scale']
            packed, quant_scales, inv_scale, awq_err = quantize_q40_with_awq(
                tensor, awq_scale, group_size
            )
            serialize_uint8(out, packed)
            serialize_fp32(out, quant_scales)
            serialize_fp32(out, inv_scale)
            total_err_awq += awq_err
            n_quantized += 1
            print(f"  [AWQ] {short_name}: {list(tensor.shape)}, "
                  f"alpha={result['alpha']:.2f}, improvement={result['improvement']:+.1f}%")
        else:
            packed, quant_scales, err = quantize_q40(tensor, group_size)
            serialize_uint8(out, packed)
            serialize_fp32(out, quant_scales)
            inv_scale = torch.ones(tensor.shape[1], dtype=torch.float32)
            serialize_fp32(out, inv_scale)
            total_err_awq += err
            n_quantized += 1
            print(f"  [Q4 ] {short_name}: {list(tensor.shape)}, no AWQ benefit")

    print("\nQuantizing and writing weights...")

    for i in range(n_layers):
        w = get(f"model.layers.{i}.self_attn.q_proj.weight")
        write_awq_quantized(f"wq.{i}", f"model.layers.{i}.self_attn.q_proj.weight", w)

    for i in range(n_layers):
        w = get(f"model.layers.{i}.self_attn.k_proj.weight")
        write_awq_quantized(f"wk.{i}", f"model.layers.{i}.self_attn.k_proj.weight", w)

    for i in range(n_layers):
        w = get(f"model.layers.{i}.self_attn.v_proj.weight")
        write_awq_quantized(f"wv.{i}", f"model.layers.{i}.self_attn.v_proj.weight", w)

    for i in range(n_layers):
        w = get(f"model.layers.{i}.self_attn.o_proj.weight")
        write_awq_quantized(f"wo.{i}", f"model.layers.{i}.self_attn.o_proj.weight", w)

    for i in range(n_layers):
        w = get(f"model.layers.{i}.mlp.gate_proj.weight")
        write_awq_quantized(f"w1.{i}", f"model.layers.{i}.mlp.gate_proj.weight", w)

    for i in range(n_layers):
        w = get(f"model.layers.{i}.mlp.down_proj.weight")
        write_awq_quantized(f"w2.{i}", f"model.layers.{i}.mlp.down_proj.weight", w)

    for i in range(n_layers):
        w = get(f"model.layers.{i}.mlp.up_proj.weight")
        write_awq_quantized(f"w3.{i}", f"model.layers.{i}.mlp.up_proj.weight", w)

    w = get("lm_head.weight")
    write_awq_quantized("lm_head", "lm_head.weight", w)

    if n_quantized > 0:
        print(f"\n  Avg naive Q4_0 max error: {total_err_naive / n_quantized:.6f}")
        print(f"  Avg AWQ   Q4_0 max error: {total_err_awq / n_quantized:.6f}")

    # === FP32 weights ===
    print("\nWriting FP32 weights...")

    emb = get("model.embed_tokens.weight")
    serialize_fp32(out, emb)
    print(f"  [FP32] embed_tokens: {list(emb.shape)}")

    for i in range(n_layers):
        w = get(f"model.layers.{i}.input_layernorm.weight")
        serialize_fp32(out, w)
    for i in range(n_layers):
        w = get(f"model.layers.{i}.post_attention_layernorm.weight")
        serialize_fp32(out, w)
    final_norm = get("model.norm.weight")
    serialize_fp32(out, final_norm)
    print(f"  [FP32] {2 * n_layers + 1} RMSNorm layers")

    for i in range(n_layers):
        w = get(f"model.layers.{i}.self_attn.q_norm.weight")
        serialize_fp32(out, w)
    print(f"  [FP32] {n_layers} Q norm layers")

    for i in range(n_layers):
        w = get(f"model.layers.{i}.self_attn.k_norm.weight")
        serialize_fp32(out, w)
    print(f"  [FP32] {n_layers} K norm layers")

    out.close()

    file_size = Path(output_path).stat().st_size
    print(f"\nExport complete: {output_path}")
    print(f"  File size: {file_size / 1024 / 1024:.1f} MB")
    print(f"\nRun inference:")
    print(f"  ./build/demo/qwen3_infer {output_path} {model_dir}/tokenizer.json 4 1")


if __name__ == "__main__":
    main()
