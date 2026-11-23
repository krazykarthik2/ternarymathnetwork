"""
production_slm_1b/tune_1b_dialogue.py

Stage 2: Dialogue & Instruction Alignment for 1B BitNet SLM:
- SFT with ChatML multi-turn conversational format.
- Multi-GPU Distributed Data Parallel (DDP) & Single-GPU Compatible.
- Automatic Checkpoint Recovery:
  * Saves optimizer state, scheduler (LR), step, and model weights every 500 steps.
  * Auto-resumes from latest checkpoint on failure.
  * Deletes older checkpoints automatically.
"""
import os
import sys
import glob
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset

os.environ["TOKENIZERS_PARALLELISM"] = "true"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../hopfield")))
from bitnet_1b_model import ProductionBitNet1B


def cleanup_old_sft_checkpoints(ckpt_dir: str, keep_latest: int = 1):
    """Deletes older SFT checkpoint files."""
    pattern = os.path.join(ckpt_dir, "sft_ckpt_step_*.pt")
    ckpts = sorted(glob.glob(pattern), key=os.path.getmtime)
    if len(ckpts) > keep_latest:
        for old in ckpts[:-keep_latest]:
            try:
                os.remove(old)
                print(f"--> [Auto-Cleanup] Deleted older SFT checkpoint: {os.path.basename(old)}")
            except Exception as e:
                print(f"[!] Warning deleting {old}: {e}")


def find_latest_sft_checkpoint(ckpt_dir: str):
    pattern = os.path.join(ckpt_dir, "sft_ckpt_step_*.pt")
    ckpts = sorted(glob.glob(pattern), key=os.path.getmtime)
    return ckpts[-1] if ckpts else None


class DialogueDataset(IterableDataset):
    """Streams conversational multi-turn dialogues formatted in ChatML."""
    def __init__(self, tokenizer, max_len=1024, rank=0, world_size=1):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        from datasets import load_dataset
        try:
            ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        except Exception:
            ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)

        for idx, item in enumerate(ds):
            if self.world_size > 1 and (idx % self.world_size != self.rank):
                continue

            messages = item.get("messages", [])
            if not messages:
                prompt = item.get("conversations", [])
                if not prompt:
                    continue
                messages = [{"role": m.get("from", "user"), "content": m.get("value", "")} for m in prompt]

            # Format into ChatML
            text = ""
            for m in messages:
                role = "user" if m.get("role") in ("user", "human") else "assistant"
                text += f"<|im_start|>{role}\n{m.get('content', '')}<|im_end|>\n"

            tokens = self.tokenizer.encode(text)
            if len(tokens) >= 16:
                # Clamp strictly to max_len to avoid any out-of-bounds positional indexing
                tokens = tokens[:self.max_len]
                yield (torch.tensor(tokens[:-1], dtype=torch.long),
                       torch.tensor(tokens[1:], dtype=torch.long))


def run_dialogue_tuning(base_ckpt="production_slm_1b/bitnet_1b_pretrained.pt", steps=5000,
                        batch_size=16, lr=1e-4, ckpt_dir="checkpoints_1b_sft",
                        output_file="production_slm_1b/bitnet_1b_dialogue_chat.pt"):
    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if is_distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_master = (rank == 0)

    if is_master:
        os.makedirs(ckpt_dir, exist_ok=True)
        print("==================================================================")
        print(f" Stage 2: 1B SLM Conversational Instruction Tuning (ChatML)")
        print(f" Distributed Mode: {'ENABLED (' + str(world_size) + ' GPUs)' if is_distributed else 'SINGLE GPU'}")
        print(f" Device          : {device} ({torch.cuda.get_device_name(local_rank) if torch.cuda.is_available() else 'CPU'})")
        print(f" Total Steps     : {steps:,}")
        print(f" Batch Size/GPU  : {batch_size} (Global Batch: {batch_size * world_size})")
        print(f" Checkpoint Dir  : {ckpt_dir} (Auto-saving every 500 steps)")
        print("==================================================================")

    model = ProductionBitNet1B(dim=2048, n_layers=24, n_heads=16, hidden_dim=5632).to(device)
    model.gradient_checkpointing = True

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)

    start_step = 1

    # Check for SFT Auto-Recovery
    latest_sft_ckpt = find_latest_sft_checkpoint(ckpt_dir)
    if latest_sft_ckpt and os.path.exists(latest_sft_ckpt):
        if is_master:
            print(f"--> [Auto-Resume] Found previous SFT checkpoint: {latest_sft_ckpt}")
        ckpt = torch.load(latest_sft_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt["step"] + 1
        if is_master:
            print(f"--> [Auto-Resume] Resuming SFT training from step {start_step}/{steps} (LR: {optimizer.param_groups[0]['lr']:.2e})!")
    elif os.path.exists(base_ckpt):
        if is_master:
            print(f"--> Loading pre-trained 1B foundation weights from: {base_ckpt}")
        ckpt = torch.load(base_ckpt, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        if is_master:
            print("--> Base foundation model loaded successfully!")

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    raw_model = model.module if is_distributed else model

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def collate_fn(batch):
        max_b = max(len(x) for x, y in batch)
        bx, by = [], []
        for x, y in batch:
            pad_x = F.pad(x, (0, max_b - len(x)), value=0)
            pad_y = F.pad(y, (0, max_b - len(y)), value=-100)
            bx.append(pad_x)
            by.append(pad_y)
        return torch.stack(bx), torch.stack(by)

    dataset = DialogueDataset(tokenizer, max_len=512, rank=rank, world_size=world_size)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    t0 = time.time()
    data_iter = iter(loader)

    if is_master:
        print("\n--> Starting Dialogue SFT Optimization...")
    for step in range(start_step, steps + 1):
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
            loss = criterion(logits.view(-1, raw_model.vocab_size), y.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if (step % 50 == 0 or step == start_step or step == steps) and is_master:
            ppl = math.exp(min(loss.item(), 20.0))
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"SFT Step {step:05d}/{steps:05d} | Chat Loss: {loss.item():.4f} | PPL: {ppl:.2f} | LR: {cur_lr:.2e} | Elapsed: {(time.time() - t0)/60:.1f}m")

        # Auto-Checkpointing every 500 steps
        if step % 500 == 0 or step == steps:
            if is_master:
                ckpt_path = os.path.join(ckpt_dir, f"sft_ckpt_step_{step}.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": loss.item()
                }, ckpt_path)
                print(f"--> [Checkpoint] Saved SFT step {step} to: {ckpt_path}")
                cleanup_old_sft_checkpoints(ckpt_dir, keep_latest=1)

    if is_master:
        torch.save(raw_model.state_dict(), output_file)
        print(f"\n--> Dialogue tuning finished! Saved final assistant SLM to: {output_file}")

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune 1B BitNet on Dialogue")
    parser.add_argument("--base_ckpt", type=str, default="production_slm_1b/bitnet_1b_pretrained.pt")
    parser.add_argument("--steps", type=int, default=5000, help="Default: 5000 SFT steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Default: 4 (with gradient checkpointing to prevent OOM)")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints_1b_sft")
    parser.add_argument("--output", type=str, default="production_slm_1b/bitnet_1b_dialogue_chat.pt")
    args = parser.parse_args()

    run_dialogue_tuning(base_ckpt=args.base_ckpt, steps=args.steps, batch_size=args.batch_size,
                        ckpt_dir=args.ckpt_dir, output_file=args.output)
