import os
import math
import time
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
from bitnet_model import BitNet125M
from device_utils import get_device, get_device_info

def evaluate_checkpoints():
    device = get_device()
    print("=" * 80)
    print("RUNNING STANDARDIZED EVALUATION ON HELD-OUT TOKENS (NO RETRAINING)")
    print(f"Device: {get_device_info()['name']}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size

    # Load streaming validation data
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    # Skip 100k records to ensure completely unvisited held-out text
    ds_iter = iter(ds.skip(100000))

    seq_len = 512
    batch_size = 16
    num_eval_batches = 30
    batches = []
    token_buffer = []

    print(f"Pre-fetching {num_eval_batches} batches of {batch_size} sequences ({num_eval_batches * batch_size * seq_len:,} tokens)...")
    while len(batches) < num_eval_batches:
        item = next(ds_iter)
        text = item.get("text", "")
        if not text:
            continue
        ids = tokenizer.encode(text)
        token_buffer.extend(ids)
        while len(token_buffer) >= (seq_len + 1) * batch_size and len(batches) < num_eval_batches:
            batch_x, batch_y = [], []
            for _ in range(batch_size):
                chunk = token_buffer[:seq_len + 1]
                token_buffer = token_buffer[seq_len:]
                batch_x.append(chunk[:-1])
                batch_y.append(chunk[1:])
            batches.append((torch.tensor(batch_x, dtype=torch.long), torch.tensor(batch_y, dtype=torch.long)))

    print("Data ready. Evaluating checkpoints...")
    criterion = nn.CrossEntropyLoss()

    checkpoints = [
        ("STE Baseline", "checkpoints/125M_ste/checkpoint_step_10000.pt"),
        ("Temperature Annealing", "checkpoints/125M_temp_anneal/checkpoint_step_10000.pt"),
        ("MIGE (Adaptive Sobolev)", "checkpoints/125M_mige/checkpoint_step_10000.pt"),
    ]

    results = {}
    for name, path in checkpoints:
        t0 = time.time()
        model = BitNet125M(vocab_size=vocab_size, dim=768, n_layers=12, n_heads=12,
                           hidden_dim=2048, max_seq_len=seq_len, method="discrete_eval").to(device)
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)

        for m in model.modules():
            if hasattr(m, "method"):
                m.method = "discrete_eval"
        model.eval()

        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for bx, by in batches:
                bx, by = bx.to(device), by.to(device)
                with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                    logits = model(bx)
                    loss = criterion(logits.view(-1, vocab_size), by.view(-1))
                total_loss += loss.item() * by.numel()
                total_tokens += by.numel()

        elapsed = time.time() - t0
        avg_loss = total_loss / total_tokens
        ppl = math.exp(min(avg_loss, 20.0))

        # Weight distribution
        weights = []
        for n, p in model.named_parameters():
            if 'weight' in n and ('proj' in n or 'w1' in n or 'w2' in n or 'w3' in n):
                weights.append(p.detach().flatten())
        all_w = torch.cat(weights)
        gamma = all_w.abs().mean()
        w_norm = all_w / gamma
        near_neg1 = (w_norm < -0.5).float().mean().item()
        near_0 = ((w_norm >= -0.5) & (w_norm <= 0.5)).float().mean().item()
        near_pos1 = (w_norm > 0.5).float().mean().item()

        results[name] = {
            "test_loss": avg_loss,
            "perplexity": ppl,
            "eval_time_sec": elapsed,
            "ternary_weights": {
                "-1": near_neg1,
                "0": near_0,
                "+1": near_pos1
            }
        }
        print(f"--> [{name}] Test Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | Time: {elapsed:.2f}s | Split: [-1: {near_neg1*100:.1f}%, 0: {near_0*100:.1f}%, +1: {near_pos1*100:.1f}%]")

    with open("results/final_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSuccessfully saved results to results/final_evaluation_results.json")
    return results

if __name__ == "__main__":
    evaluate_checkpoints()
