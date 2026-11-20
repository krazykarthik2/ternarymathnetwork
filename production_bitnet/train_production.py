"""
production_bitnet/train_production.py

Production Distributed Training Pipeline for BitNet b1.58:
- No heuristics, no temperature schedules, no discretization shock.
- Full INT8 Activation x INT2 Ternary Weight quantization.
- 500-step linear warmup + Cosine Decay.
- FlashAttention-2 + Pinned memory async streaming.
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
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "true"

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../hopfield")))
from bitnet_production import ProductionBitNet125M
from train_hopfield_torch_maxspeed import FastStreamingDataset


def main():
    parser = argparse.ArgumentParser(description="Train Production BitNet b1.58")
    parser.add_argument("--steps", type=int, default=10000, help="Standard production steps (10k)")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--min_lr", type=float, default=1.5e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", type=str, default="sample-10BT")
    parser.add_argument("--output_dir", type=str, default="checkpoints_production")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Training Production BitNet b1.58 on: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) ===")

    model = ProductionBitNet125M().to(device)
    print(f"Model Parameters: {model.count_parameters():,}")

    dataset = FastStreamingDataset(args.dataset, args.subset, seq_len=args.seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.workers > 0 and torch.cuda.is_available()),
        prefetch_factor=4 if (args.workers > 0 and torch.cuda.is_available()) else None
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    # Linear Warmup followed by Cosine Annealing
    def get_lr(step):
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(max(1, args.steps - args.warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr / args.lr + (1.0 - args.min_lr / args.lr) * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    data_iter = iter(loader)

    for step in range(1, args.steps + 1):
        try:
            bx, by = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            bx, by = next(data_iter)

        x = bx.to(device, non_blocking=True)
        y = by.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=dtype):
            logits = model(x)
            loss = criterion(logits.view(-1, 50257), y.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % 50 == 0 or step == 1 or step == args.steps:
            speed = (step * args.batch_size * args.seq_len) / max(time.time() - start_time, 1e-5)
            ppl = math.exp(min(loss.item(), 20.0))
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Step {step:05d}/{args.steps:05d} | Loss: {loss.item():.4f} | PPL: {ppl:.2f} | LR: {current_lr:.2e} | Speed: {speed:.0f} tok/s")

        if step % 2500 == 0 or step == args.steps:
            ckpt_path = os.path.join(args.output_dir, f"bitnet_prod_step_{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "loss": loss.item()
            }, ckpt_path)
            print(f"--> Saved production checkpoint: {ckpt_path}")

    print(f"\n[Done] Production training finished in {(time.time() - start_time)/60:.2f} minutes.")


if __name__ == "__main__":
    main()
