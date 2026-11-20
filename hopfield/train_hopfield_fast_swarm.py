"""
hopfield/train_hopfield_fast_swarm.py

Ultra-Fast Training Pipeline:
1. Analytical Hopfield Initialization (Zero-Shot Closed Form, ~150 ms)
2. Swarm-Guided Genetic Boundary Probing (TSGO)
3. Short-Burst MIGE-TA (1,000 - 2,000 steps instead of 10,000+)
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

# Add parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hopfield_bitnet import HopfieldBitNet125M
from hybrid_fast_optimizer import TernarySwarmGeneticOptimizer
from train_hopfield_125m import get_token_streamer


def main():
    parser = argparse.ArgumentParser(description="Fast Hopfield + Swarm/TA Pipeline")
    parser.add_argument("--steps", type=int, default=1500, help="Short burst steps (1000-2000)")
    parser.add_argument("--swarm_interval", type=int, default=10, help="Interval for swarm genetic mutation")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--output_dir", type=str, default="checkpoints_fast_swarm")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Fast Hopfield + Swarm/TA Pipeline on: {device} ===")

    # Initialize model
    model = HopfieldBitNet125M(dim=768, n_layers=12, n_heads=12, hidden_dim=2048, method="mige_ta").to(device)
    print(f"Model Parameters: {model.count_parameters():,}")

    data_iter = get_token_streamer("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", args.seq_len, args.batch_size, device)

    # -------------------------------------------------------------
    # Step 1: Hopfield Closed-Form Analytical Initialization
    # -------------------------------------------------------------
    print("\n--> [Step 1] Initializing weights via Single-Shot Hopfield Calculation...")
    t0 = time.time()
    init_x, _ = next(data_iter)
    model.calculate_all_weights_single_shot(init_x)
    print(f"--> Hopfield initialization completed in {(time.time() - t0)*1000:.2f} ms")

    # -------------------------------------------------------------
    # Step 2: Swarm Genetic Boundary Optimizer + Fast MIGE-TA
    # -------------------------------------------------------------
    print(f"\n--> [Step 2] Launching Fast Hybrid Optimization ({args.steps} total steps)...")
    swarm_opt = TernarySwarmGeneticOptimizer(model, mutation_rate=0.03, temperature=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.05)

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    for step in range(1, args.steps + 1):
        x, y = next(data_iter)
        tau = step / float(args.steps)
        # Fast smooth cooling
        T_val = 0.01 + 0.5 * (1.0 - 0.01) * (1.0 + math.cos(math.pi * tau))
        eps_val = 0.05 + 0.5 * (0.35 - 0.05) * (1.0 + math.cos(math.pi * tau))

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype):
            logits = model(x, T=T_val, epsilon=eps_val)
            loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        # Interleaved Swarm Genetic Mutation
        if step % args.swarm_interval == 0:
            current_loss = loss.item()
            new_loss = swarm_opt.step_swarm_mutation(criterion, x, y, current_loss)
            if new_loss < current_loss:
                loss = torch.tensor(new_loss)

        if step % 25 == 0 or step == 1:
            speed = (step * args.batch_size * args.seq_len) / (time.time() - start_time)
            ppl = math.exp(min(loss.item(), 20.0))
            print(f"Step {step:04d}/{args.steps:04d} | Loss: {loss.item():.4f} | PPL: {ppl:.2f} | T: {T_val:.3f} | Speed: {speed:.0f} tok/s")

    # Save final model
    final_ckpt = os.path.join(args.output_dir, "hopfield_fast_final.pt")
    torch.save({
        "step": args.steps,
        "model_state_dict": model.state_dict(),
        "loss": loss.item()
    }, final_ckpt)
    print(f"\n--> Model saved to: {final_ckpt}")
    print(f"Total training time: {(time.time() - start_time)/60:.2f} min (Finished in a fraction of standard time!)")


if __name__ == "__main__":
    main()
