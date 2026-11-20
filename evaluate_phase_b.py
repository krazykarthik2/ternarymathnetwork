"""
Phase B: Native Zero-Shot Downstream Task Evaluation Harness.
Self-contained, ultra-robust evaluator using Hugging Face datasets directly
(HellaSwag, PIQA, ARC-Easy, WinoGrande) via exact discrete log-likelihood ranking.
"""
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import json
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
from bitnet_model import BitNet125M
from device_utils import get_device, get_device_info

class DiscreteBitNetWrapper:
    """Wrapper that enforces true hardware ternary snaps {-1, 0, +1} for evaluation."""
    def __init__(self, checkpoint_path, scale="125M", device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        vocab_size = self.tokenizer.vocab_size

        if scale == "125M":
            self.model = BitNet125M(vocab_size=vocab_size, dim=768, n_layers=12, n_heads=12,
                                    hidden_dim=2048, method="discrete_eval").to(device)
        elif scale == "350M":
            self.model = BitNet125M(vocab_size=vocab_size, dim=1024, n_layers=24, n_heads=16,
                                    hidden_dim=2816, method="discrete_eval").to(device)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get("model_state", ckpt)
        self.model.load_state_dict(state_dict, strict=False)

        for m in self.model.modules():
            if hasattr(m, "method"):
                m.method = "discrete_eval"
        self.model.eval()

    def score_choice(self, prompt, choice):
        """Calculates normalized log-likelihood of completing prompt with choice."""
        full_text = prompt + (" " if prompt and not prompt.endswith(" ") else "") + str(choice)
        enc_full = self.tokenizer.encode(full_text)
        enc_prompt = self.tokenizer.encode(prompt)

        input_ids = torch.tensor([enc_full], dtype=torch.long, device=self.device)
        with torch.no_grad():
            with torch.amp.autocast(self.device.type, enabled=(self.device.type == "cuda"), dtype=torch.bfloat16):
                logits = self.model(input_ids)
            log_probs = F.log_softmax(logits[0, :-1], dim=-1)

            target_ids = input_ids[0, 1:]
            choice_start = max(0, len(enc_prompt) - 1)
            choice_targets = target_ids[choice_start:]
            choice_log_probs = log_probs[choice_start:]

            score = choice_log_probs.gather(1, choice_targets.unsqueeze(1)).sum().item()
            norm_score = score / max(1, len(choice_targets))
            return norm_score

def eval_hellaswag(model_wrapper, num_samples=500):
    print(f"\n--> Evaluating HellaSwag ({num_samples} samples)...")
    try:
        ds = load_dataset("Rowan/hellaswag", split="validation", streaming=True)
    except Exception:
        try:
            ds = load_dataset("parquet", data_files={"validation": "https://huggingface.co/datasets/Rowan/hellaswag/resolve/main/data/hellaswag-validation.parquet"}, split="validation", streaming=True)
        except Exception:
            ds = load_dataset("hellaswag", split="validation", streaming=True)

    correct = 0
    total = 0
    t0 = time.time()
    for item in ds:
        if total >= num_samples:
            break
        ctx = item.get("ctx", "")
        endings = item.get("endings", [])
        label = int(item.get("label", 0))

        scores = [model_wrapper.score_choice(ctx, end) for end in endings]
        pred = int(torch.tensor(scores).argmax().item())
        if pred == label:
            correct += 1
        total += 1

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    elapsed = time.time() - t0
    print(f"HellaSwag Acc: {acc:.2f}% ({correct}/{total}) | Time: {elapsed:.1f}s")
    return {"accuracy": acc, "correct": correct, "total": total}

def eval_piqa(model_wrapper, num_samples=500):
    print(f"\n--> Evaluating PIQA ({num_samples} samples)...")
    # Clean mirror URL guaranteed to bypass piqa.py script crash
    parquet_url = "https://huggingface.co/datasets/baber/piqa/resolve/main/piqa_validation.parquet"
    try:
        ds = load_dataset("parquet", data_files={"validation": parquet_url}, split="validation", streaming=True)
    except Exception:
        try:
            parquet_backup = "https://huggingface.co/datasets/ybisk/piqa/resolve/main/plain_text/validation-00000-of-00001.parquet"
            ds = load_dataset("parquet", data_files={"validation": parquet_backup}, split="validation", streaming=True)
        except Exception as e:
            print(f"[!] Warning: Could not load PIQA ({e})")
            return {"accuracy": 0.0, "correct": 0, "total": 0, "error": str(e)}

    correct = 0
    total = 0
    t0 = time.time()
    for item in ds:
        if total >= num_samples:
            break
        goal = item.get("goal", "")
        sol1 = item.get("sol1", "")
        sol2 = item.get("sol2", "")
        label = int(item.get("label", 0))

        s1 = model_wrapper.score_choice(goal, sol1)
        s2 = model_wrapper.score_choice(goal, sol2)
        pred = 0 if s1 > s2 else 1
        if pred == label:
            correct += 1
        total += 1

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    elapsed = time.time() - t0
    print(f"PIQA Acc: {acc:.2f}% ({correct}/{total}) | Time: {elapsed:.1f}s")
    return {"accuracy": acc, "correct": correct, "total": total}

def eval_arc_easy(model_wrapper, num_samples=500):
    print(f"\n--> Evaluating ARC-Easy ({num_samples} samples)...")
    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation", streaming=True)
    except Exception:
        try:
            ds = load_dataset("parquet", data_files={"validation": "https://huggingface.co/datasets/allenai/ai2_arc/resolve/refs%2Fconvert%2Fparquet/ARC-Easy/validation/0000.parquet"}, split="validation", streaming=True)
        except Exception:
            ds = load_dataset("ai2_arc", name="ARC-Easy", split="validation", streaming=True)

    correct = 0
    total = 0
    t0 = time.time()
    for item in ds:
        if total >= num_samples:
            break
        question = item.get("question", "")
        choices = item.get("choices", {})
        texts = choices.get("text", [])
        labels = choices.get("label", [])
        answer_key = item.get("answerKey", "")

        scores = [model_wrapper.score_choice(question, c) for c in texts]
        best_idx = int(torch.tensor(scores).argmax().item())
        pred_label = labels[best_idx] if best_idx < len(labels) else ""

        if str(pred_label).strip() == str(answer_key).strip():
            correct += 1
        total += 1

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    elapsed = time.time() - t0
    print(f"ARC-Easy Acc: {acc:.2f}% ({correct}/{total}) | Time: {elapsed:.1f}s")
    return {"accuracy": acc, "correct": correct, "total": total}

def eval_winogrande(model_wrapper, num_samples=500):
    print(f"\n--> Evaluating WinoGrande ({num_samples} samples)...")
    try:
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation", streaming=True)
    except Exception:
        try:
            ds = load_dataset("parquet", data_files={"validation": "https://huggingface.co/datasets/allenai/winogrande/resolve/refs%2Fconvert%2Fparquet/winogrande_xl/validation/0000.parquet"}, split="validation", streaming=True)
        except Exception:
            ds = load_dataset("winogrande", "winogrande_xl", split="validation", streaming=True)

    correct = 0
    total = 0
    t0 = time.time()
    for item in ds:
        if total >= num_samples:
            break
        sentence = item.get("sentence", "")
        opt1 = item.get("option1", "")
        opt2 = item.get("option2", "")
        answer = str(item.get("answer", "1"))
        label = 0 if answer == "1" else 1

        p1 = sentence.replace("_", opt1)
        p2 = sentence.replace("_", opt2)
        s1 = model_wrapper.score_choice("", p1)
        s2 = model_wrapper.score_choice("", p2)
        pred = 0 if s1 > s2 else 1
        if pred == label:
            correct += 1
        total += 1

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    elapsed = time.time() - t0
    print(f"WinoGrande Acc: {acc:.2f}% ({correct}/{total}) | Time: {elapsed:.1f}s")
    return {"accuracy": acc, "correct": correct, "total": total}

def evaluate_downstream(checkpoint_path, name, scale="125M", num_samples=500, output_json="results/phase_b_results.json"):
    device = get_device()
    print("=" * 80)
    print(f"PHASE B: ZERO-SHOT DOWNSTREAM HARNESS ({scale})")
    print(f"Model: {name} [{checkpoint_path}]")
    print(f"Hardware: {get_device_info()['name']}")
    print(f"Sample Budget: {num_samples} per task")
    print("=" * 80)

    wrapper = DiscreteBitNetWrapper(checkpoint_path, scale=scale, device=device)

    results = {}
    results["hellaswag"] = eval_hellaswag(wrapper, num_samples=num_samples)
    results["piqa"] = eval_piqa(wrapper, num_samples=num_samples)
    results["arc_easy"] = eval_arc_easy(wrapper, num_samples=num_samples)
    results["winogrande"] = eval_winogrande(wrapper, num_samples=num_samples)

    valid_accs = [
        res["accuracy"] for k, res in results.items()
        if "accuracy" in res and res["total"] > 0
    ]
    avg_acc = (sum(valid_accs) / len(valid_accs)) if valid_accs else 0.0
    results["average_acc"] = avg_acc

    print("\n" + "=" * 80)
    print(f"[{name}] SUMMARY: Average Zero-Shot Accuracy = {avg_acc:.2f}%")
    print("=" * 80)

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    all_results = {}
    if os.path.exists(output_json):
        try:
            with open(output_json, "r") as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}

    all_results[name] = results
    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f">> Saved Phase B results to: {output_json}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--scale", type=str, default="125M")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output_json", type=str, default="results/phase_b_results.json")
    args = parser.parse_args()

    name = args.name or os.path.basename(args.checkpoint)
    evaluate_downstream(args.checkpoint, name, scale=args.scale,
                        num_samples=args.num_samples, output_json=args.output_json)
    os._exit(0)
