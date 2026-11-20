"""
hopfield/train_hopfield_pure_ta_blazing.py

Blazing Fast Hopfield Init + Pure Vanilla Temperature Annealing (TA):
- No MIGE complexity, no floor gradients, no swarm overhead.
- True Pure TA (soft_ternary with analytical derivative).
- Fast Cosine Annealing in 500-1000 steps (~3 to 5 minutes on L40S).
- High throughput async batch pipeline.
"""
import os
import sys
import math
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["TOKENIZERS_PARALLELISM"] = "false"
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hopfield_bitnet import HopfieldBitNet125M
from quantizers import temp_anneal_quantize, hard_ternary
from train_hopfield_125m import get_token_streamer


def main():
    parser = argparse.ArgumentParser(description="Blazing Fast Hopfield + Pure TA")
    parser.add_argument("--steps", type=int, default=800, help="Very short steps (500 - 1000 max)")
    parser.add_argument("--batch_size", type=int, default=32, help="L40S optimal batch size")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", type=str, default="sample-10BT")
    parser.add_argument("--output_dir", type=str, default="checkpoints_pure_ta")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Blazing Fast Pure TA on: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) ===")

    # Create model configured for vanilla temp_anneal
    model = HopfieldBitNet125M(dim=768, n_layers=12, n_heads=12, hidden_dim=2048, method="temp_anneal").to(device)
    for m in model.modules():
        if hasattr(m, "method"):
            m.method = "temp_anneal"

    data_iter = get_token_streamer(args.dataset, args.subset, "train", args.seq_len, args.batch_size, device)

    # 1. Hopfield Analytical Single-Shot Calculation
    print("\n[Step 1] Analytical Hopfield Weight Calculation...")
    t0 = time.time()
    init_x, _ = next(data_iter)
    model.calculate_all_weights_single_shot(init_x)
    print(f"--> Done in {(time.time() - t0)*1000:.2f} ms")

    # 2. Pure Temperature Annealing (500 - 1000 steps)
    print(f"\n[Step 2] Pure TA Training for ONLY {args.steps} steps...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    for step in range(1, args.steps + 1):
        x, y = next(data_iter)
        tau = step / float(args.steps)
        # Smooth temperature drop from 1.0 down to 0.01
        T_val = 0.01 + 0.5 * (1.0 - 0.01) * (1.0 + math.cos(math.pi * tau))

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype):
            logits = model(x, T=T_val)
            loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        if step % 25 == 0 or step == 1 or step == args.steps:
            speed = (step * args.batch_size * args.seq_len) / max(time.time() - start_time, 1e-5)
            ppl = math.exp(min(loss.item(), 20.0))
            print(f"Step {step:04d}/{args.steps:04d} | Loss: {loss.item():.4f} | PPL: {ppl:.2f} | T: {T_val:.3f} | Speed: {speed:.0f} tok/s")

    # Save final model
    final_ckpt = os.path.join(args.output_dir, "pure_ta_final.pt")
    torch.save({
        "step": args.steps,
        "model_state_dict": model.state_dict(),
        "loss": loss.item()
    }, final_ckpt)
    print(f"\n--> Model saved to: {final_ckpt}")
    print(f"Finished in {(time.time() - start_time)/60:.2f} minutes!")


if __name__ == "__main__":
    main()
