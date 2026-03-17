"""
Direct export: HuggingFace safetensors → .bin (no intermediate .pth)
Saves ~32GB disk space for 8B models.
"""
import struct
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM


def serialize_fp32(file, tensor):
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)


def main():
    parser = argparse.ArgumentParser(description="Direct export HF model → .bin")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="HuggingFace model directory (with safetensors + config.json)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .bin file path")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)

    print(f"Loading model from {model_dir}...")
    print(f"  hidden_size={cfg['hidden_size']}, layers={cfg['num_hidden_layers']}, "
          f"heads={cfg['num_attention_heads']}, kv_heads={cfg['num_key_value_heads']}")

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()

    dim = cfg["num_attention_heads"] * cfg.get("head_dim", 128)
    hidden_dim = cfg["hidden_size"]
    n_layers = cfg["num_hidden_layers"]
    n_heads = cfg["num_attention_heads"]
    n_kv_heads = cfg["num_key_value_heads"]
    vocab_size = cfg["vocab_size"]
    max_seq_len = cfg.get("max_position_embeddings", 40960)
    intermediate_size = cfg["intermediate_size"]

    weights = [
        *[model.model.layers[i].input_layernorm.weight for i in range(n_layers)],
        *[model.model.layers[i].post_attention_layernorm.weight for i in range(n_layers)],
        model.model.norm.weight,

        model.model.embed_tokens.weight,

        *[model.model.layers[i].self_attn.q_proj.weight for i in range(n_layers)],
        *[model.model.layers[i].self_attn.q_norm.weight for i in range(n_layers)],

        *[model.model.layers[i].self_attn.k_proj.weight for i in range(n_layers)],
        *[model.model.layers[i].self_attn.k_norm.weight for i in range(n_layers)],

        *[model.model.layers[i].self_attn.v_proj.weight for i in range(n_layers)],
        *[model.model.layers[i].self_attn.o_proj.weight for i in range(n_layers)],

        *[model.model.layers[i].mlp.gate_proj.weight for i in range(n_layers)],
        *[model.model.layers[i].mlp.down_proj.weight for i in range(n_layers)],
        *[model.model.layers[i].mlp.up_proj.weight for i in range(n_layers)],
        model.lm_head.weight,
    ]

    print(f"\nExporting {len(weights)} weight tensors to {args.output}...")
    header = struct.pack('iiiiiiii', dim, hidden_dim, n_layers, n_heads,
                         n_kv_heads, vocab_size, max_seq_len, intermediate_size)

    with open(args.output, 'wb') as out_file:
        out_file.write(header)
        for i, w in enumerate(tqdm(weights, desc="Writing weights")):
            serialize_fp32(out_file, w)

    import os
    size_gb = os.path.getsize(args.output) / (1024**3)
    print(f"\n✅ Exported to {args.output} ({size_gb:.2f} GB)")
    print(f"   Run: ./build/demo/qwen3_infer {args.output} {args.model_dir}/tokenizer.json")


if __name__ == "__main__":
    main()
