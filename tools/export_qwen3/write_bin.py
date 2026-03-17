"""
Interactive Script for Running Qwen3 Model
Author: Bound
Date: May 30, 2025
Version: 1.0
"""
import struct
import torch
import argparse

from load import model_load


def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)


def parse_args():
    parser = argparse.ArgumentParser(description="Export Qwen3 .pth to .bin for C++ inference.")
    parser.add_argument("-p", "--checkpoint", type=str, default="qwen3_8b_weights.pth",
                        help="Input .pth checkpoint file path")
    parser.add_argument("-o", "--output", type=str, default="Qwen3-8B.bin",
                        help="Output .bin file path for C++ inference")
    parser.add_argument("-n", "--model_name", type=str, default="/home/dzy/models/Qwen3-8B",
                        help="Model directory (for tokenizer & config.json)")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="Device for loading (cpu/cuda)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, tokenizer = model_load(model_name=args.model_name, device=device, checkpoint=args.checkpoint)
    model.eval()
    weights = [
        # 2 * rmsnorm + 1
        *[layer.input_layernorm.weight for layer in model.model.layers],
        *[layer.post_attention_layernorm.weight for layer in model.model.layers],
        model.model.norm.weight,

        model.model.embed_tokens.weight,

        *[layer.self_attn.q_proj.weight for layer in model.model.layers],
        *[layer.self_attn.q_norm.weight for layer in model.model.layers],

        *[layer.self_attn.k_proj.weight for layer in model.model.layers],
        *[layer.self_attn.k_norm.weight for layer in model.model.layers],

        *[layer.self_attn.v_proj.weight for layer in model.model.layers],
        *[layer.self_attn.o_proj.weight for layer in model.model.layers],

        *[layer.mlp.gate_proj.weight for layer in model.model.layers],
        *[layer.mlp.down_proj.weight for layer in model.model.layers],
        *[layer.mlp.up_proj.weight for layer in model.model.layers],
        model.lm_head.weight
    ]
    import numpy as np
    dim = model.config.num_attention_heads * model.config.head_dim
    hidden_dim = model.config.hidden_size
    n_layers = len(model.model.layers)
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    vocab_size = model.config.vocab_size
    max_seq_len = model.config.max_position_embeddings

    ## export
    file_path = args.output
    out_file = open(file_path, 'wb')

    header = struct.pack('iiiiiiii', dim, hidden_dim, n_layers, n_heads,
                         n_kv_heads, vocab_size, max_seq_len, model.config.intermediate_size)
    out_file.write(header)

    for w in weights:
        serialize_fp32(out_file, w)

    out_file.close()
    print(f"✅ Exported to {file_path}")
    print(f"   Run: ./build/demo/qwen3_infer {file_path} {args.model_name}/tokenizer.json")


if __name__ == "__main__":
    main()
