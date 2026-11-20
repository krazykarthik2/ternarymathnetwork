"""
hopfield/evaluate_hopfield_phase_b.py

Phase B Evaluation Harness for Hopfield-BitNet-125M:
Directly performs:
1. Analytical Single-Shot Closed-Form Weight Calculation on token representations from data (Zero-Training).
2. True Hardware Ternary Snaps {-1, 0, +1} on all layers.
3. Zero-shot downstream task evaluations on:
   - HellaSwag
   - PIQA
   - ARC-Easy
   - WinoGrande
via exact discrete log-likelihood scoring.
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

# Add parent directory for shared modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hopfield_bitnet import HopfieldBitNet125M
from device_utils import get_device


class DiscreteHopfieldBitNetEvaluator:
    """Evaluates Hopfield-BitNet-125M enforcing true discrete ternary weights {-1, 0, 1}."""
    def __init__(self, checkpoint_path=None, samples_for_calculation=512, device=None):
        if device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device)

        print(f"--> Initializing DiscreteHopfieldBitNetEvaluator on: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        vocab_size = self.tokenizer.vocab_size

        self.model = HopfieldBitNet125M(vocab_size=vocab_size, dim=768, n_layers=12, n_heads=12,
                                        hidden_dim=2048, method="discrete").to(self.device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"--> Loading pre-trained checkpoint from: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
            self.model.load_state_dict(state_dict, strict=False)
        else:
            print("--> Zero iterative training: Calculating Hopfield ternary weights directly in single-shot...")
            self.calculate_weights_from_data(num_samples=samples_for_calculation)

        # Enforce discrete evaluation
        for m in self.model.modules():
            if hasattr(m, "method"):
                m.method = "discrete"
        self.model.eval()

    def calculate_weights_from_data(self, num_samples=512):
        """Calculates closed-form ternary weights from sample data tokens."""
        t0 = time.time()
        try:
            ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
            tokens = []
            for item in ds:
                text = item.get("text", "")
                if text:
                    tokens.extend(self.tokenizer.encode(text))
                if len(tokens) >= num_samples * 64:
                    break
            tokens = tokens[:num_samples * 64]
            sample_tensor = torch.tensor(tokens, dtype=torch.long, device=self.device).view(-1, 64)
        except Exception as e:
            print(f"[!] Warning: Fallback to synthetic sample tokens for analytical calculation ({e})")
            sample_tensor = torch.randint(0, self.tokenizer.vocab_size, (num_samples, 64), device=self.device)

        self.model.calculate_all_weights_single_shot(sample_tensor)
        print(f"--> Single-shot Hopfield weights calculated in {(time.time() - t0)*1000:.2f} ms")

    def score_choice(self, prompt, choice):
        """Calculates normalized log-likelihood of completing prompt with choice."""
        full_text = prompt + (" " if prompt and not prompt.endswith(" ") else "") + str(choice)
        enc_full = self.tokenizer.encode(full_text)
        enc_prompt = self.tokenizer.encode(prompt)

        input_ids = torch.tensor([enc_full], dtype=torch.long, device=self.device)
        with torch.no_grad():
            use_amp = (self.device.type == "cuda")
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            with torch.amp.autocast(self.device.type, enabled=use_amp, dtype=dtype):
                logits = self.model(input_ids)
            log_probs = F.log_softmax(logits[0, :-1], dim=-1)

            target_ids = input_ids[0, 1:]
            choice_start = max(0, len(enc_prompt) - 1)
            choice_targets = target_ids[choice_start:]
            choice_log_probs = log_probs[choice_start:]

            score = choice_log_probs.gather(1, choice_targets.unsqueeze(1)).sum().item()
            norm_score = score / max(1, len(choice_targets))
            return norm_score


def eval_hellaswag(evaluator, num_samples=250):
    print(f"\n--> Evaluating HellaSwag ({num_samples} samples)...")
    try:
        ds = load_dataset("Rowan/hellaswag", split="validation", streaming=True)
    except Exception:
        try:
            ds = load_dataset("parquet", data_files={"validation": "https://huggingface.co/datasets/Rowan/hellaswag/resolve/main/data/hellaswag-validation.parquet"}, split="validation", streaming=True)
        except Exception:
            ds = load_dataset("hellaswag", split="validation", streaming=True)

    correct, total = 0, 0
    t0 = time.time()
    for item in ds:
        if total >= num_samples:
            break
        ctx = item.get("ctx", "")
        endings = item.get("endings", [])
        label = int(item.get("label", 0))

        scores = [evaluator.score_choice(ctx, end) for end in endings]
        pred = int(torch.tensor(scores).argmax().item())
        if pred == label:
            correct += 1
        total += 1

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    elapsed = time.time() - t0
    print(f"HellaSwag Acc: {acc:.2f}% ({correct}/{total}) | Time: {elapsed:.1f}s")
    return {"accuracy": acc, "correct": correct, "total": total}


def eval_piqa(evaluator, num_samples=250):
    print(f"\n--> Evaluating PIQA ({num_samples} samples)...")
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

    correct, total = 0, 0
    t0 = time.time()
    for item in ds:
        if total >= num_samples:
            break
        goal = item.get("goal", "")
        sol1 = item.get("sol1", "")
        sol2 = item.get("sol2", "")
        label = int(item.get("label", 0))

        s1 = evaluator.score_choice(goal, sol1)
        s2 = evaluator.score_choice(goal, sol2)
        pred = 0 if s1 > s2 else 1
        if pred == label:
            correct += 1
        total += 1

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    elapsed = time.time() - t0
    print(f"PIQA Acc: {acc:.2f}% ({correct}/{total}) | Time: {elapsed:.1f}s")
    return {"accuracy": acc, "correct": correct, "total": total}


def eval_arc_easy(evaluator, num_samples=250):
    print(f"\n--> Evaluating ARC-Easy ({num_samples} samples)...")
    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation", streaming=True)
    except Exception:
        try:
            ds = load_dataset("parquet", data_files={"validation": "https://huggingface.co/datasets/allenai/ai2_arc/resolve/refs%2Fconvert%2Fparquet/ARC-Easy/validation/0000.parquet"}, split="validation", streaming=True)
        except Exception:
            ds = load_dataset("ai2_arc", name="ARC-Easy", split="validation", streaming=True)

    correct, total = 0, 0
    t0 = time.time()
    for item in ds:
        if total >= num_samples:
            break
        question = item.get("question", "")
        choices = item.get("choices", {})
        texts = choices.get("text", [])
        labels = choices.get("label", [])
        answer_key = item.get("answerKey", "")

        scores = [evaluator.score_choice(question, c) for c in texts]
        best_idx = int(torch.tensor(scores).argmax().item())
        pred_label = labels[best_idx] if best_idx < len(labels) else ""

        if str(pred_label).strip() == str(answer_key).strip():
            correct += 1
        total += 1

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    elapsed = time.time() - t0
    print(f"ARC-Easy Acc: {acc:.2f}% ({correct}/{total}) | Time: {elapsed:.1f}s")
    return {"accuracy": acc, "correct": correct, "total": total}


def eval_winogrande(evaluator, num_samples=250):
    print(f"\n--> Evaluating WinoGrande ({num_samples} samples)...")
    try:
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation", streaming=True)
    except Exception:
        try:
            ds = load_dataset("parquet", data_files={"validation": "https://huggingface.co/datasets/allenai/winogrande/resolve/refs%2Fconvert%2Fparquet/winogrande_xl/validation/0000.parquet"}, split="validation", streaming=True)
        except Exception:
            ds = load_dataset("winogrande", "winogrande_xl", split="validation", streaming=True)

    correct, total = 0, 0
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
        s1 = evaluator.score_choice("", p1)
        s2 = evaluator.score_choice("", p2)
        pred = 0 if s1 > s2 else 1
        if pred == label:
            correct += 1
        total += 1

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    elapsed = time.time() - t0
    print(f"WinoGrande Acc: {acc:.2f}% ({correct}/{total}) | Time: {elapsed:.1f}s")
    return {"accuracy": acc, "correct": correct, "total": total}


def main():
    parser = argparse.ArgumentParser(description="Hopfield-BitNet Phase B Downstream Evaluator")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional trained checkpoint path")
    parser.add_argument("--samples", type=int, default=250, help="Evaluation samples per benchmark")
    parser.add_argument("--calc_samples", type=int, default=512, help="Sample count for analytical single-shot weight calculation")
    parser.add_argument("--output", type=str, default="phase_b_hopfield_results.json")
    args = parser.parse_args()

    evaluator = DiscreteHopfieldBitNetEvaluator(checkpoint_path=args.checkpoint,
                                              samples_for_calculation=args.calc_samples)

    results = {}
    results["hellaswag"] = eval_hellaswag(evaluator, num_samples=args.samples)
    results["piqa"] = eval_piqa(evaluator, num_samples=args.samples)
    results["arc_easy"] = eval_arc_easy(evaluator, num_samples=args.samples)
    results["winogrande"] = eval_winogrande(evaluator, num_samples=args.samples)

    avg_acc = (results["hellaswag"]["accuracy"] + results["piqa"]["accuracy"] +
               results["arc_easy"]["accuracy"] + results["winogrande"]["accuracy"]) / 4.0
    results["average_accuracy"] = avg_acc

    print("\n" + "=" * 60)
    print(f"Phase B Hopfield-BitNet Results Summary:")
    print(f"  HellaSwag  : {results['hellaswag']['accuracy']:.2f}%")
    print(f"  PIQA       : {results['piqa']['accuracy']:.2f}%")
    print(f"  ARC-Easy   : {results['arc_easy']['accuracy']:.2f}%")
    print(f"  WinoGrande : {results['winogrande']['accuracy']:.2f}%")
    print(f"  AVERAGE    : {avg_acc:.2f}%")
    print("=" * 60)

    out_file = os.path.join(os.path.dirname(__file__), args.output)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to: {out_file}")


if __name__ == "__main__":
    main()
