import os
import sys
import argparse
import torch

def extract_weights(input_path: str, output_path: str = None, precision: str = "bfloat16"):
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_weights_only.pt"

    print(f"--> Loading full checkpoint: {input_path}")
    data = torch.load(input_path, map_location="cpu")

    # Extract model state dict
    if "model_state_dict" in data:
        weights = data["model_state_dict"]
        step = data.get("step", "N/A")
        print(f"--> Found SFT model state dict at step {step}")
    else:
        weights = data

    # Strip prefixes if saved from DDP
    clean_weights = {}
    for k, v in weights.items():
        clean_key = k[7:] if k.startswith("module.") else k
        if precision == "bfloat16" and v.is_floating_point():
            clean_weights[clean_key] = v.to(torch.bfloat16)
        elif precision == "float16" and v.is_floating_point():
            clean_weights[clean_key] = v.to(torch.float16)
        else:
            clean_weights[clean_key] = v

    torch.save(clean_weights, output_path)
    in_size = os.path.getsize(input_path) / (1024 * 1024 * 1024)
    out_size = os.path.getsize(output_path) / (1024 * 1024 * 1024)

    print("==================================================================")
    print(f" Extraction Complete!")
    print(f" Original Full Checkpoint (with Adam states): {in_size:.2f} GB")
    print(f" Extracted Downloadable Weights ({precision}): {out_size:.2f} GB")
    print(f" Saved to: {output_path}")
    print("==================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract clean downloadable model weights from large training checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to full checkpoint file (.pt)")
    parser.add_argument("--output", type=str, default=None, help="Output weights file path")
    parser.add_argument("--precision", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    extract_weights(args.checkpoint, args.output, args.precision)
