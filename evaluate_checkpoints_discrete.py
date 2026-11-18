"""
Evaluates all completed 10,000-step BitNet checkpoints strictly under exact
hardware discrete integer weights snaps {-1, 0, +1} on held-out FineWeb-Edu tokens.
"""
import os
import json
import math
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
from bitnet_model import BitNet125M
from device_utils import get_device, get_device_info

def evaluate_suite():
    device = get_device()
    print("=" * 90)
    print("RUNNING ACCURATE HARDWARE DISCRETE EVALUATION ON HELD-OUT TOKENS")
    print(f"Device: {get_device_info()['name']}")
    print("=" * 90)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    ds_iter = iter(ds.skip(100000))

    seq_len = 512
    batch_size = 16
    num_eval_batches = 30
    batches = []
    token_buffer = []

    print(f"Fetching {num_eval_batches * batch_size * seq_len:,} held-out validation tokens...")
    while len(batches) < num_eval_batches:
        item = next(ds_iter)
        text = item.get("text", "")
        if not text:
            continue
        ids = tokenizer.encode(text)
        token_buffer.extend(ids)
        while len(token_buffer) >= (seq_len + 1) * batch_size and len(batches) < num_eval_batches:
            bx, by = [], []
            for _ in range(batch_size):
                chunk = token_buffer[:seq_len + 1]
                token_buffer = token_buffer[seq_len:]
                bx.append(chunk[:-1])
                by.append(chunk[1:])
            batches.append((torch.tensor(bx, dtype=torch.long), torch.tensor(by, dtype=torch.long)))

    criterion = nn.CrossEntropyLoss()
    
    models_to_test = {
        "MIGE-NBS (Normalized Bounded-Sobolev, Step 10,000)": "checkpoints/125M_mige_nbs/checkpoint_step_10000.pt",
        "Temperature Annealing (Step 10,000)": "checkpoints/125M_temp_anneal/checkpoint_step_10000.pt",
        "MIGE (Adaptive Sobolev, Step 10,000)": "checkpoints/125M_mige/checkpoint_step_10000.pt",
        "STE Baseline (Step 10,000)": "checkpoints/125M_ste/checkpoint_step_10000.pt",
    }
    
    results = {}
    eval_json_path = "results/final_evaluation_results.json"
    if os.path.exists(eval_json_path):
        with open(eval_json_path) as f:
            try:
                results = json.load(f)
            except Exception:
                results = {}

    for name, path in models_to_test.items():
        if not os.path.exists(path):
            print(f"[-] Checkpoint not found: {path} (skipping)")
            continue
        print(f"\n--> Evaluating: {name} [{path}]...")
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
        print(f"[{name}] Test Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | Time: {elapsed:.2f}s | Sparsity (0): {near_0*100:.1f}%")

    with open(eval_json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved accurate evaluation results to {eval_json_path}")

if __name__ == "__main__":
    evaluate_suite()
