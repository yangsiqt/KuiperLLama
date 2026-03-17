"""
Qwen3 Model Configuration
Author: Bound
Date: May 1, 2025
Version: 1.0
Support: Qwen3-0.6B, Qwen3-8B, etc. (load from config.json)
"""

import json
import torch
from pathlib import Path


def load_config_from_json(config_path: str) -> "Qwen3Config":
    """从模型目录的 config.json 加载配置，支持任意规模的 Qwen3 模型"""
    path = Path(config_path)
    if path.is_dir():
        config_file = path / "config.json"
    else:
        config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found: {config_file}")
    with open(config_file, "r") as f:
        cfg = json.load(f)
    return Qwen3Config(
        vocab_size=cfg.get("vocab_size", 151936),
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        head_dim=cfg.get("head_dim", 128),
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        max_position_embeddings=cfg.get("max_position_embeddings", 40960),
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
        rope_theta=cfg.get("rope_theta", 1000000),
        torch_type=torch.bfloat16,
        eos_token_id=cfg.get("eos_token_id", 151645),
    )


class Qwen3Config:
    """Qwen3-0.6B 默认配置（用于向后兼容）"""
    def __init__(
        self,
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=3072,
        head_dim=128,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        max_position_embeddings=1024,
        rms_norm_eps=1e-6,
        rope_theta=1000000,
        torch_type=torch.bfloat16,
        eos_token_id=151645
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.torch_type = torch_type
        self.eos_token_id = eos_token_id