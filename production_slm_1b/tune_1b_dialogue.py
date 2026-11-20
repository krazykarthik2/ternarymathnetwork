"""
production_slm_1b/tune_1b_dialogue.py

Stage 2: Dialogue & Instruction Alignment for 1B BitNet SLM:
- SFT with ChatML multi-turn conversational format.
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
    def __init__(self, tokenizer, max_len=1024):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __iter__(self):
        from datasets import load_dataset
        try:
            ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        except Exception:
            ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)

        for item in ds:
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
                tokens = tokens[:self.max_len + 1]
                yield (torch.tensor(tokens[:-1], dtype=torch.long),
                       torch.tensor(tokens[1:], dtype=torch.long))


def run_dialogue_tuning(base_ckpt: str, steps: int = 5000, lr: float = 3e-4, batch_size: int = 16,
                        ckpt_dir: str = "checkpoints_1b_sft", output_file: str = "bitnet_1b_dialogue_chat.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("==================================================================")
    print(f" 1B BitNet SLM Dialogue Instruction Tuning Engine")
    print(f" Device          : {device}")
    print(f" Target Steps    : {steps:,}")
    print(f" Checkpoint Dir  : {ckpt_dir} (Auto-checkpointing & recovery enabled)")
    print("==================================================================")

    model = ProductionBitNet1B(dim=2048, n_layers=24, n_heads=16, hidden_dim=5632).to(device)

    start_step = 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.1)

    # Check for SFT resume checkpoint first
    latest_sft = find_latest_sft_checkpoint(ckpt_dir)
    if latest_sft and os.path.exists(latest_sft):
        print(f"--> [Auto-Resume] Found previous SFT checkpoint: {latest_sft}")
        sft_data = torch.load(latest_sft, map_location=device)
        model.load_state_dict(sft_data["model_state_dict"])
        optimizer.load_state_dict(sft_data["optimizer_state_dict"])
        scheduler.load_state_dict(sft_data["scheduler_state_dict"])
        start_step = sft_data["step"] + 1
        print(f"--> [Auto-Resume] Successfully recovered at Step {start_step:,} / {steps:,} (LR: {optimizer.param_groups[0]['lr']:.2e})!")
    elif os.path.exists(base_ckpt):
        print(f"--> Loading pre-trained 1B foundation weights from: {base_ckpt}")
        ckpt = torch.load(base_ckpt, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        print("--> Base foundation model loaded successfully!")

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

    dataset = DialogueDataset(tokenizer, max_len=512)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    t0 = time.time()
    data_iter = iter(loader)

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
            loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % 50 == 0 or step == start_step or step == steps:
            ppl = math.exp(min(loss.item(), 20.0))
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"SFT Step {step:05d}/{steps:05d} | Chat Loss: {loss.item():.4f} | PPL: {ppl:.2f} | LR: {cur_lr:.2e} | Elapsed: {(time.time() - t0)/60:.1f}m")

        # Auto-Checkpointing every 500 steps
        if step % 500 == 0 or step == steps:
            ckpt_path = os.path.join(ckpt_dir, f"sft_ckpt_step_{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": loss.item()
            }, ckpt_path)
            print(f"--> [Checkpoint] Saved SFT step {step} to: {ckpt_path}")
            cleanup_old_sft_checkpoints(ckpt_dir, keep_latest=1)

    torch.save(model.state_dict(), output_file)
    print(f"\n--> Dialogue tuning finished! Saved final assistant SLM to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune 1B BitNet on Dialogue")
    parser.add_argument("--base_ckpt", type=str, default="production_slm_1b/bitnet_1b_pretrained.pt")
    parser.add_argument("--steps", type=int, default=5000, help="Default: 5000 SFT steps")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints_1b_sft")
    parser.add_argument("--output", type=str, default="production_slm_1b/bitnet_1b_dialogue_chat.pt")
    args = parser.parse_args()

    run_dialogue_tuning(base_ckpt=args.base_ckpt, steps=args.steps, batch_size=args.batch_size,
                        ckpt_dir=args.ckpt_dir, output_file=args.output)
