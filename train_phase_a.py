"""
BitNet b1.58 Multi-GPU / Distributed Pre-Training Engine (Phase A).
Supports:
- Model specifications: BitNet-125M (12 layers, d=768, h=12) & BitNet-350M (24 layers, d=1024, h=16)
- Resilient Streaming: Thread-safe generator without PyArrow GIL deadlock / thread-state collisions
- Resume Capability: Automatically resume training from any step checkpoint
"""
import os
import sys

# Prevent tokenizers and pyarrow multithreading deadlock during long streaming runs
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import time
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from bitnet_model import BitNet125M
from quantizers import hard_ternary
from device_utils import get_device, get_device_info

class HFStreamDataset(IterableDataset):
    """
    Thread-safe streaming dataset wrapper for HuggingFace datasets.
    Streams token batches directly into memory without downloading to disk.
    """
    def __init__(self, dataset_name="HuggingFaceFW/fineweb-edu", subset="sample-10BT", split="train", seq_len=512, tokenizer=None):
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        try:
            import datasets
        except ImportError:
            raise ImportError(
                "\n\n[FATAL] The 'datasets' library is not installed in this Python environment!\n"
                "Please run: pip install datasets\n"
            )

    def __iter__(self):
        from datasets import load_dataset
        print(f"[Dataset] Initializing stream: {self.dataset_name} (subset: {self.subset})...")
        if self.subset:
            ds = load_dataset(self.dataset_name, name=self.subset, split=self.split, streaming=True)
        else:
            ds = load_dataset(self.dataset_name, split=self.split, streaming=True)
        
        token_buffer = []
        for item in ds:
            text = item.get("text", "") or item.get("story", "")
            if not text:
                continue
            if self.tokenizer is not None:
                ids = self.tokenizer.encode(text)
            else:
                ids = [ord(c) % 256 for c in text]
            
            token_buffer.extend(ids)
            while len(token_buffer) >= self.seq_len + 1:
                chunk = token_buffer[:self.seq_len + 1]
                token_buffer = token_buffer[self.seq_len:]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y

def get_model(scale="125M", method="mige", vocab_size=50257, max_seq_len=512):
    if scale == "125M":
        return BitNet125M(vocab_size=vocab_size, dim=768, n_layers=12, n_heads=12,
                          hidden_dim=2048, max_seq_len=max_seq_len, method=method)
    elif scale == "350M":
        return BitNet125M(vocab_size=vocab_size, dim=1024, n_layers=24, n_heads=16,
                          hidden_dim=2816, max_seq_len=max_seq_len, method=method)
    else:
        raise ValueError(f"Unknown scale: {scale}")

def train_phase_a(args):
    device = get_device()
    print("=" * 90)
    print(f"PHASE A: BITNET b1.58 CORE SCALE BENCHMARK [{args.scale}]")
    print(f"Hardware: {get_device_info()['name']}")
    print(f"Dataset: {args.dataset} (Subset: {args.subset}) | Streaming Mode (Zero Disk)")
    print(f"Method: {args.method.upper()} | Steps: {args.steps:,} | Batch: {args.batch_size} | Seq Len: {args.seq_len}")
    print("=" * 90)

    try:
        from transformers import AutoTokenizer
        print("[Tokenizer] Loading GPT-2 BPE tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        vocab_size = tokenizer.vocab_size
        print(f"[Tokenizer] Loaded successfully. Vocab size: {vocab_size:,}")
    except Exception as e:
        raise RuntimeError(f"[FATAL] Failed to load tokenizer: {e}")

    model = get_model(scale=args.scale, method=args.method, vocab_size=vocab_size, max_seq_len=args.seq_len).to(device)
    print(f"Model instantiated. Total parameter count: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = os.path.join(args.output_dir, f"{args.scale}_{args.method}")
    os.makedirs(ckpt_dir, exist_ok=True)

    start_step = 1
    metrics = {"steps": [], "train_loss": [], "train_ppl": [], "discrete_loss": [], "discrete_ppl": [], "discretization_gap": []}

    # Resume from checkpoint if specified
    if args.resume_from:
        if os.path.exists(args.resume_from):
            print(f"[Checkpoint] Resuming from: {args.resume_from}")
            state = torch.load(args.resume_from, map_location=device)
            model.load_state_dict(state["model_state"])
            if "optimizer_state" in state:
                optimizer.load_state_dict(state["optimizer_state"])
            start_step = state.get("step", 0) + 1
            metrics = state.get("metrics", metrics)
            print(f"[Checkpoint] Successfully loaded. Resuming at step: {start_step}")
        else:
            print(f"[Checkpoint] Warning: Checkpoint {args.resume_from} not found. Starting fresh.")

    # num_workers=0 ensures clean single-threaded streaming execution with zero GIL crashes
    train_ds = HFStreamDataset(dataset_name=args.dataset, subset=args.subset, split="train", seq_len=args.seq_len, tokenizer=tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0)
    train_iter = iter(train_loader)

    device_type = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=(device.type == "cuda"))
    start_time = time.time()

    for step in range(start_step, args.steps + 1):
        x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        T_val = 0.01 + 0.5 * (1.0 - 0.01) * (1.0 + math.cos(math.pi * step / args.steps))
        eps_val = 0.05 + 0.5 * (0.35 - 0.05) * (1.0 + math.cos(math.pi * step / args.steps))
        floor_val = 0.05 + 0.5 * (0.30 - 0.05) * (1.0 + math.cos(math.pi * step / args.steps))

        if args.method == "mige":
            for m in model.modules():
                if hasattr(m, "method") and m.method == "mige":
                    m.floor_grad = floor_val

        optimizer.zero_grad()
        with torch.amp.autocast(device_type, enabled=(device.type == "cuda"), dtype=torch.bfloat16):
            logits = model(x, T=T_val, epsilon=eps_val)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % args.eval_interval == 0 or step == args.steps:
            train_loss_val = loss.item()
            train_ppl_val = math.exp(min(train_loss_val, 20.0))

            original_methods = {}
            for name, module in model.named_modules():
                if hasattr(module, "method"):
                    original_methods[name] = module.method
                    module.method = "discrete_eval"

            model.eval()
            with torch.no_grad():
                with torch.amp.autocast(device_type, enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                    disc_logits = model(x)
                    disc_loss = criterion(disc_logits.view(-1, vocab_size), y.view(-1)).item()
                    disc_ppl = math.exp(min(disc_loss, 20.0))

            for name, module in model.named_modules():
                if name in original_methods:
                    module.method = original_methods[name]
            model.train()

            gap_delta = disc_ppl - train_ppl_val
            metrics["steps"].append(step)
            metrics["train_loss"].append(train_loss_val)
            metrics["train_ppl"].append(train_ppl_val)
            metrics["discrete_loss"].append(disc_loss)
            metrics["discrete_ppl"].append(disc_ppl)
            metrics["discretization_gap"].append(gap_delta)

            elapsed = time.time() - start_time
            print(f"Step {step:6d}/{args.steps} | Train PPL: {train_ppl_val:7.2f} | Discrete PPL: {disc_ppl:7.2f} | Gap (Delta): {gap_delta:+6.2f} | Time: {elapsed:.1f}s")

            if step % args.save_interval == 0 or step == args.steps:
                ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")
                torch.save({
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "metrics": metrics,
                    "config": vars(args)
                }, ckpt_path)
                print(f"--> Saved checkpoint to: {ckpt_path}")

    with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=str, default="125M", choices=["125M", "350M"])
    parser.add_argument("--method", type=str, default="mige", choices=["ste", "temp_anneal", "mige"])
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", type=str, default="sample-10BT")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    train_phase_a(args)
