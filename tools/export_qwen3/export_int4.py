"""
Qwen3 INT4 (Q4_0) Weight-Only Quantization Export
Directly reads HuggingFace safetensors and exports to .bin for C++ inference.
"""
import struct
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from safetensors import safe_open


def quantize_q40(w, group_size):
    """
    Q4_0 symmetric quantization: float -> int4 (range [-8, 7]).
    Packs two int4 values per byte: low nibble = even idx, high nibble = odd idx.
    Returns: packed_bytes (uint8 tensor), scales (fp32 tensor), max_error (float)
    """
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
    assert int4_flat.numel() % 2 == 0
    low = int4_flat[0::2]
    high = int4_flat[1::2]
    packed = (high << 4) | low

    return packed, scale.float(), err


def serialize_fp32(f, tensor):
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    f.write(struct.pack(f'{len(d)}f', *d))


def serialize_int8(f, tensor):
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    f.write(struct.pack(f'{len(d)}b', *d))


def serialize_uint8(f, tensor):
    d = tensor.detach().cpu().view(-1).numpy().astype(np.uint8)
    f.write(d.tobytes())


def load_safetensors(model_dir):
    """Load all weights from safetensors files in model directory."""
    model_dir = Path(model_dir)
    state_dict = {}
    for st_file in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3 to INT4 (Q4_0) .bin")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to HuggingFace Qwen3 model directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .bin file path")
    parser.add_argument("--group_size", type=int, default=64,
                        help="Quantization group size")
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

    output_path = args.output or f"Qwen3-{model_dir.name}-int4.bin"

    print("\nLoading safetensors...")
    sd = load_safetensors(model_dir)
    print(f"  Loaded {len(sd)} tensors")

    def get(name):
        t = sd[name]
        return t

    out = open(output_path, 'wb')

    # Header: ModelConfig (8 int32s for Qwen3)
    header = struct.pack('iiiiiiii',
                         dim, hidden_size, n_layers, num_attention_heads,
                         num_kv_heads, vocab_size, max_seq_len, intermediate_size)
    out.write(header)

    # group_size (int32) - read by Model::read_model_file when is_quant_model=true
    out.write(struct.pack('i', args.group_size))

    group_size = args.group_size
    total_err = 0.0
    n_quantized = 0

    def write_quantized(name, tensor):
        nonlocal total_err, n_quantized
        packed, scales, err = quantize_q40(tensor, group_size)
        serialize_uint8(out, packed)
        serialize_fp32(out, scales)
        total_err += err
        n_quantized += 1
        print(f"  [Q4] {name}: {list(tensor.shape)}, max_err={err:.6f}")

    # === Quantized weights (INT4 packed + FP32 scales) ===
    print("\nQuantizing and writing weights...")

    # WQ
    for i in range(n_layers):
        w = get(f"model.layers.{i}.self_attn.q_proj.weight")
        write_quantized(f"wq.{i}", w)

    # WK
    for i in range(n_layers):
        w = get(f"model.layers.{i}.self_attn.k_proj.weight")
        write_quantized(f"wk.{i}", w)

    # WV
    for i in range(n_layers):
        w = get(f"model.layers.{i}.self_attn.v_proj.weight")
        write_quantized(f"wv.{i}", w)

    # WO
    for i in range(n_layers):
        w = get(f"model.layers.{i}.self_attn.o_proj.weight")
        write_quantized(f"wo.{i}", w)

    # W1 (gate_proj)
    for i in range(n_layers):
        w = get(f"model.layers.{i}.mlp.gate_proj.weight")
        write_quantized(f"w1.{i}", w)

    # W2 (down_proj)
    for i in range(n_layers):
        w = get(f"model.layers.{i}.mlp.down_proj.weight")
        write_quantized(f"w2.{i}", w)

    # W3 (up_proj)
    for i in range(n_layers):
        w = get(f"model.layers.{i}.mlp.up_proj.weight")
        write_quantized(f"w3.{i}", w)

    # CLS (lm_head)
    w = get("lm_head.weight")
    write_quantized("lm_head", w)

    print(f"\n  Average max quantization error: {total_err / n_quantized:.6f}")

    # === FP32 weights (not quantized) ===
    print("\nWriting FP32 weights...")

    # Embedding
    emb = get("model.embed_tokens.weight")
    serialize_fp32(out, emb)
    print(f"  [FP32] embed_tokens: {list(emb.shape)}")

    # RMSNorm: attention norms, ffn norms, final norm (2*L+1)
    for i in range(n_layers):
        w = get(f"model.layers.{i}.input_layernorm.weight")
        serialize_fp32(out, w)
    for i in range(n_layers):
        w = get(f"model.layers.{i}.post_attention_layernorm.weight")
        serialize_fp32(out, w)
    final_norm = get("model.norm.weight")
    serialize_fp32(out, final_norm)
    print(f"  [FP32] {2 * n_layers + 1} RMSNorm layers")

    # Q norm
    for i in range(n_layers):
        w = get(f"model.layers.{i}.self_attn.q_norm.weight")
        serialize_fp32(out, w)
    print(f"  [FP32] {n_layers} Q norm layers")

    # K norm
    for i in range(n_layers):
        w = get(f"model.layers.{i}.self_attn.k_norm.weight")
        serialize_fp32(out, w)
    print(f"  [FP32] {n_layers} K norm layers")

    out.close()

    file_size = Path(output_path).stat().st_size
    print(f"\nExport complete: {output_path}")
    print(f"  File size: {file_size / 1024 / 1024:.1f} MB")
    print(f"\nRun inference:")
    print(f"  ./build/demo/qwen3_infer {output_path} {model_dir}/tokenizer.json 4")


if __name__ == "__main__":
    main()
