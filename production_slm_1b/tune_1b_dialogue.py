"""
production_slm_1b/tune_1b_dialogue.py

Stage 2: Dialogue & Instruction Alignment for 1B BitNet SLM:
Loads pre-trained 1B weights from Stage 1.
Fine-tunes on conversational multi-turn dialogue (UltraChat / OpenHermes)
using ChatML standard:
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
"""
import os
import sys
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


def run_dialogue_tuning(base_ckpt: str, steps: int = 2500, lr: float = 3e-4, batch_size: int = 16,
                        output_file: str = "bitnet_1b_dialogue_chat.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================================")
    print(f" 1B BitNet SLM Dialogue Instruction Tuning")
    print(f" Device : {device}")
    print(f" Target Steps: {steps:,} | Batch Size: {batch_size}")
    print("==================================================================")

    model = ProductionBitNet1B(dim=2048, n_layers=24, n_heads=16, hidden_dim=5632).to(device)

    if os.path.exists(base_ckpt):
        print(f"--> Loading base pre-trained 1B weights from: {base_ckpt}")
        ckpt = torch.load(base_ckpt, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        print("--> Pre-trained 1B foundation successfully loaded!")
    else:
        print("[!] Base checkpoint not found, training directly.")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def collate_fn(batch):
        # Pad dynamically to max length in batch
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    t0 = time.time()
    data_iter = iter(loader)

    print("\n--> Starting Dialogue SFT Optimization...")
    for step in range(1, steps + 1):
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

        if step % 25 == 0 or step == 1 or step == steps:
            ppl = math.exp(min(loss.item(), 20.0))
            print(f"SFT Step {step:04d}/{steps:04d} | Chat Loss: {loss.item():.4f} | PPL: {ppl:.2f} | Elapsed: {(time.time() - t0)/60:.1f}m")

    torch.save(model.state_dict(), output_file)
    print(f"\n--> Dialogue tuning finished! Saved final assistant SLM to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune 1B BitNet on Dialogue")
    parser.add_argument("--base_ckpt", type=str, default="production_slm_1b/bitnet_1b_pretrained.pt")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output", type=str, default="production_slm_1b/bitnet_1b_dialogue_chat.pt")
    args = parser.parse_args()

    run_dialogue_tuning(base_ckpt=args.base_ckpt, steps=args.steps, batch_size=args.batch_size, output_file=args.output)
