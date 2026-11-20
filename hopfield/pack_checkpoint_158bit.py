"""
hopfield/pack_checkpoint_158bit.py

Packs a 647 MB FP32 BitNet checkpoint into a ~40 MB true 1.58-bit ternary binary:
Stores {-1, 0, +1} weights as 2-bit signed packed integers (4 weights per byte)
along with per-tensor FP16 scale factors (gamma).
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from quantizers import hard_ternary


def pack_ternary_tensor(w_q: torch.Tensor):
    """Packs ternary tensor {-1, 0, 1} into 2-bit unsigned integers: 00=0, 01=+1, 11=-1."""
    # Map {-1: 3, 0: 0, 1: 1} (2-bit representation)
    val_map = torch.where(w_q == 1.0, 1, torch.where(w_q == -1.0, 3, 0)).to(torch.uint8)
    flat = val_map.flatten()
    
    pad_len = (4 - (len(flat) % 4)) % 4
    if pad_len > 0:
        flat = torch.cat([flat, torch.zeros(pad_len, dtype=torch.uint8, device=flat.device)])
    
    # Pack 4 ternary values per uint8 byte
    packed = (flat[0::4] | (flat[1::4] << 2) | (flat[2::4] << 4) | (flat[3::4] << 6))
    return packed, w_q.shape


def pack_checkpoint(ckpt_path: str, output_path: str):
    print(f"--> Loading full-precision checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))

    packed_dict = {}
    total_raw_bytes = 0
    total_packed_bytes = 0

    for name, param in state_dict.items():
        total_raw_bytes += param.numel() * param.element_size()
        if "weight" in name and param.dim() == 2:
            # Linear weight layer -> quantize to {-1, 0, 1} & pack
            gamma = param.abs().mean().clamp(min=1e-5)
            w_norm = param / gamma
            w_q = hard_ternary(w_norm, threshold=0.5)
            
            packed_data, orig_shape = pack_ternary_tensor(w_q)
            packed_dict[name] = {
                "packed": packed_data,
                "shape": orig_shape,
                "gamma": gamma.to(torch.float16),
                "is_ternary": True
            }
            total_packed_bytes += packed_data.numel() + 2
        else:
            # Embedding or RMSNorm -> keep in float16
            packed_dict[name] = param.to(torch.float16)
            total_packed_bytes += param.numel() * 2

    torch.save(packed_dict, output_path)
    print("=" * 60)
    print(f"Original Checkpoint Size : {total_raw_bytes / (1024**2):.2f} MB")
    print(f"Packed 1.58-bit Size     : {os.path.getsize(output_path) / (1024**2):.2f} MB")
    print(f"Compression Ratio        : {total_raw_bytes / os.path.getsize(output_path):.1f}x smaller!")
    print(f"Saved packed model to    : {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pack BitNet model into true 1.58-bit binary")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="bitnet_158bit_packed.pt")
    args = parser.parse_args()
    pack_checkpoint(args.checkpoint, args.output)
