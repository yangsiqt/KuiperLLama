"""
Weight Loading Utility for Qwen3 Model
Author: Bound
Date: May 5, 2025
Version: 1.0
"""

import torch
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer

from config import Qwen3Config, load_config_from_json
from model import Qwen3ForCausalLM

def check_device_availability(device: torch.device):
    if device == "cuda":
        return torch.cuda.is_available()
    elif device == "mps":
        return torch.backends.mps.is_available() # MPS backend availability check
    return True

def weight_load(
    model: Qwen3ForCausalLM, 
    checkpoint_path: str, 
    device: torch.device
):
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint file is not exist!: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    named_parameters = list(model.named_parameters())
    
    progress_bar = tqdm(
        named_parameters,
        desc="Loading weights",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )
    
    for name, param in progress_bar:
        if name in checkpoint:
            if param.shape == checkpoint[name].shape:
                param.data.copy_(checkpoint[name])
                progress_bar.set_postfix_str(f"{name} loaded")
            else:
                progress_bar.write(f"Shape mismatch (skipped): {name} - "
                                  f"Model: {param.shape} ≠ Checkpoint: {checkpoint[name].shape}")
        else:
            progress_bar.write(f"Missing: {name}")
            
    progress_bar.close()
    print( "✅ Model loaded successfully.")
    del checkpoint
    return model


def model_load(
    model_name: str,
    device: torch.device,
    checkpoint: str,
    config=None,
):  
    if not check_device_availability(device):
        print(f"Specified device ({device}) is not available. Switching to CPU.")
        device = torch.device("cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if config is None:
        config = load_config_from_json(model_name)
        print(f"Loaded config from {model_name}/config.json: "
              f"hidden_size={config.hidden_size}, layers={config.num_hidden_layers}")
    model = Qwen3ForCausalLM(config=config).to(device=device, dtype=config.torch_type)
    print("Start loading model weight......")
    model = weight_load(model=model, checkpoint_path=checkpoint, device=device)
    
    print(f"Model is loaded on {device}.")
    return model, tokenizer



if __name__ == "__main__":
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description="Export Qwen3 model weights to .pth file")
    parser.add_argument("--model_name", type=str, default="/home/dzy/models/Qwen3-8B",
                       help="HuggingFace model path (e.g. /home/dzy/models/Qwen3-8B)")
    parser.add_argument("--output_file", type=str, default="qwen3_8b_weights.pth",
                       help="Output .pth file path")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    torch.save(model.state_dict(), args.output_file)
    print(f"Model weights saved to: {args.output_file}")

