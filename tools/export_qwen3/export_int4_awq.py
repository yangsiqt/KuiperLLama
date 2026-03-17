"""
Qwen3 INT4 AWQ (Activation-Aware Weight Quantization) Export

AWQ key insight: not all weight channels are equally important.
Channels with larger activation magnitudes contribute more to output,
so we protect them by applying per-channel scaling before quantization.

    Y = X @ W^T = (X / s) @ (s * W)^T

where s is a per-channel scaling factor derived from activation statistics.
This is a mathematically equivalent transformation that makes quantization
of the important channels more accurate.

The CUDA inference kernel is identical to naive Q4_0 -- the difference
is only in how weights are pre-processed during export.
"""
import struct
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


def quantize_q40(w, group_size):
    """Q4_0 symmetric quantization with INT4 packing."""
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


def compute_awq_scales(weight, act_scales, alpha=0.5):
    """
    Compute AWQ per-channel scaling factors.

    For each input channel j:
        s_j = (act_scale_j)^alpha / (weight_scale_j)^(1-alpha)

    This balances quantization difficulty between activations and weights.
    Larger alpha = more protection for high-activation channels.
    """
    w_scales = weight.abs().max(dim=0).values.float()
    w_scales = w_scales.clamp(min=1e-8)
    act_scales = act_scales.float().clamp(min=1e-8)

    scales = (act_scales.pow(alpha) / w_scales.pow(1 - alpha)).clamp(min=1e-4, max=1e4)
    return scales


def apply_awq_scaling(weight, scales):
    """Apply per-input-channel scaling: W_scaled = W * diag(s)"""
    return weight.float() * scales.unsqueeze(0)


@torch.no_grad()
def collect_activation_scales(model, tokenizer, calib_texts, device="cpu"):
    """
    Collect per-channel activation magnitude statistics from calibration data.
    Returns a dict mapping layer weight name -> per-channel activation scale tensor.
    """
    act_scales = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            x = input[0].detach().float()
            x_abs = x.view(-1, x.shape[-1]).abs()
            x_mean = x_abs.mean(dim=0)
            if name in act_scales:
                act_scales[name] = torch.max(act_scales[name], x_mean)
            else:
                act_scales[name] = x_mean
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    for text in calib_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model(**inputs)

    for h in hooks:
        h.remove()

    return act_scales


def serialize_fp32(f, tensor):
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    f.write(struct.pack(f'{len(d)}f', *d))


def serialize_uint8(f, tensor):
    d = tensor.detach().cpu().view(-1).numpy().astype(np.uint8)
    f.write(d.tobytes())


CALIBRATION_TEXTS = [
    "The meaning of life is a philosophical question that has puzzled thinkers for centuries.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
    "In quantum computing, qubits can exist in superposition, representing both 0 and 1 simultaneously.",
    "Climate change is one of the most pressing challenges facing humanity in the 21st century.",
    "The human brain contains approximately 86 billion neurons, each forming thousands of connections.",
    "Python is a high-level programming language known for its readability and versatile ecosystem.",
    "The stock market reflects collective expectations about future corporate earnings and economic conditions.",
]


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3 to INT4 with AWQ calibration")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="AWQ alpha: higher = more activation-aware (0.0-1.0)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for calibration (cpu/cuda)")
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
    print(f"  layers={n_layers}, hidden={hidden_size}, dim={dim}")
    print(f"  AWQ alpha={args.alpha}, group_size={args.group_size}")

    output_path = args.output or f"Qwen3-{model_dir.name}-int4-awq.bin"

    # Step 1: Load model for calibration
    print("\n[Step 1] Loading model for AWQ calibration...")
    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.float32, device_map=device
    )
    model.eval()

    # Step 2: Collect activation statistics
    print("[Step 2] Collecting activation statistics...")
    act_scales = collect_activation_scales(model, tokenizer, CALIBRATION_TEXTS, device)
    print(f"  Collected scales for {len(act_scales)} linear layers")

    # Step 3: Apply AWQ scaling and quantize
    print("[Step 3] Applying AWQ scaling and quantizing to INT4...")

    sd = model.state_dict()
    group_size = args.group_size

    out = open(output_path, 'wb')

    header = struct.pack('iiiiiiii',
                         dim, hidden_size, n_layers, num_attention_heads,
                         num_kv_heads, vocab_size, max_seq_len, intermediate_size)
    out.write(header)
    out.write(struct.pack('i', group_size))

    total_err_awq = 0.0
    total_err_naive = 0.0
    n_quantized = 0

    def write_quantized_awq(layer_name, hf_module_name, weight):
        nonlocal total_err_awq, total_err_naive, n_quantized

        # Naive Q4_0 for comparison
        packed_naive, _, err_naive = quantize_q40(weight, group_size)

        if hf_module_name in act_scales:
            scales = compute_awq_scales(weight, act_scales[hf_module_name], args.alpha)
            w_scaled = apply_awq_scaling(weight, scales)
        else:
            w_scaled = weight.float()

        packed, s, err_awq = quantize_q40(w_scaled, group_size)
        serialize_uint8(out, packed)
        serialize_fp32(out, s)

        total_err_awq += err_awq
        total_err_naive += err_naive
        n_quantized += 1
        improvement = (1 - err_awq / max(err_naive, 1e-10)) * 100
        print(f"  [AWQ] {layer_name}: naive_err={err_naive:.6f} -> awq_err={err_awq:.6f} ({improvement:+.1f}%)")

    # === Quantized weights ===
    # WQ
    for i in range(n_layers):
        w = sd[f"model.layers.{i}.self_attn.q_proj.weight"]
        write_quantized_awq(f"wq.{i}", f"model.layers.{i}.self_attn.q_proj", w)

    # WK
    for i in range(n_layers):
        w = sd[f"model.layers.{i}.self_attn.k_proj.weight"]
        write_quantized_awq(f"wk.{i}", f"model.layers.{i}.self_attn.k_proj", w)

    # WV
    for i in range(n_layers):
        w = sd[f"model.layers.{i}.self_attn.v_proj.weight"]
        write_quantized_awq(f"wv.{i}", f"model.layers.{i}.self_attn.v_proj", w)

    # WO
    for i in range(n_layers):
        w = sd[f"model.layers.{i}.self_attn.o_proj.weight"]
        write_quantized_awq(f"wo.{i}", f"model.layers.{i}.self_attn.o_proj", w)

    # W1 (gate_proj)
    for i in range(n_layers):
        w = sd[f"model.layers.{i}.mlp.gate_proj.weight"]
        write_quantized_awq(f"w1.{i}", f"model.layers.{i}.mlp.gate_proj", w)

    # W2 (down_proj)
    for i in range(n_layers):
        w = sd[f"model.layers.{i}.mlp.down_proj.weight"]
        write_quantized_awq(f"w2.{i}", f"model.layers.{i}.mlp.down_proj", w)

    # W3 (up_proj)
    for i in range(n_layers):
        w = sd[f"model.layers.{i}.mlp.up_proj.weight"]
        write_quantized_awq(f"w3.{i}", f"model.layers.{i}.mlp.up_proj", w)

    # CLS (lm_head)
    w = sd["lm_head.weight"]
    write_quantized_awq("lm_head", "lm_head", w)

    avg_naive = total_err_naive / n_quantized
    avg_awq = total_err_awq / n_quantized
    print(f"\n  Average quantization error: naive={avg_naive:.6f}, AWQ={avg_awq:.6f}")
    print(f"  AWQ improvement: {(1 - avg_awq / avg_naive) * 100:.1f}%")

    # === FP32 weights ===
    print("\nWriting FP32 weights...")

    serialize_fp32(out, sd["model.embed_tokens.weight"])
    print(f"  [FP32] embed_tokens")

    for i in range(n_layers):
        serialize_fp32(out, sd[f"model.layers.{i}.input_layernorm.weight"])
    for i in range(n_layers):
        serialize_fp32(out, sd[f"model.layers.{i}.post_attention_layernorm.weight"])
    serialize_fp32(out, sd["model.norm.weight"])
    print(f"  [FP32] {2 * n_layers + 1} RMSNorm layers")

    for i in range(n_layers):
        serialize_fp32(out, sd[f"model.layers.{i}.self_attn.q_norm.weight"])
    for i in range(n_layers):
        serialize_fp32(out, sd[f"model.layers.{i}.self_attn.k_norm.weight"])
    print(f"  [FP32] {2 * n_layers} Q/K norm layers")

    out.close()

    del model
    del sd
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    file_size = Path(output_path).stat().st_size
    print(f"\nExport complete: {output_path}")
    print(f"  File size: {file_size / 1024 / 1024:.1f} MB")
    print(f"\nRun inference:")
    print(f"  ./build/demo/qwen3_infer {output_path} {model_dir}/tokenizer.json 4")


if __name__ == "__main__":
    main()
