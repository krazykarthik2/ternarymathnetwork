"""
hopfield/eval_checkpoint_phase_b.py

Stand-alone Phase B evaluation script for any saved checkpoint.
Loads checkpoint safely, snaps weights to true hardware ternary {-1, 0, +1},
and runs zero-shot log-likelihood evaluations on HellaSwag, PIQA, ARC-Easy, and WinoGrande.
"""
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import time
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hopfield_bitnet import HopfieldBitNet125M


class CheckpointEvaluator:
    def __init__(self, checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        print(f"--> Initializing Evaluator on: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        vocab_size = self.tokenizer.vocab_size

        self.model = HopfieldBitNet125M(vocab_size=vocab_size, dim=768, n_layers=12, n_heads=12,
                                        hidden_dim=2048, method="discrete").to(self.device)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        print(f"--> Loading trained weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
        self.model.load_state_dict(state_dict, strict=False)

        # Enforce true discrete ternary snap
        for m in self.model.modules():
            if hasattr(m, "method"):
                m.method = "discrete"
        self.model.eval()

    def score_choice(self, prompt, choice):
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
            return score / max(1, len(choice_targets))


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
        scores = [evaluator.score_choice(item.get("ctx", ""), end) for end in item.get("endings", [])]
        if int(torch.tensor(scores).argmax().item()) == int(item.get("label", 0)):
            correct += 1
        total += 1
    acc = (correct / total) * 100.0 if total > 0 else 0.0
    print(f"HellaSwag Acc: {acc:.2f}% ({correct}/{total}) | Time: {time.time() - t0:.1f}s")
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
            return {"accuracy": 0.0, "correct": 0, "total": 0, "error": str(e)}

    correct, total = 0, 0
    t0 = time.time()
    for item in ds:
        if total >= num_samples:
            break
        s1 = evaluator.score_choice(item.get("goal", ""), item.get("sol1", ""))
        s2 = evaluator.score_choice(item.get("goal", ""), item.get("sol2", ""))
        pred = 0 if s1 > s2 else 1
        if pred == int(item.get("label", 0)):
            correct += 1
        total += 1
    acc = (correct / total) * 100.0 if total > 0 else 0.0
    print(f"PIQA Acc: {acc:.2f}% ({correct}/{total}) | Time: {time.time() - t0:.1f}s")
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
        choices = item.get("choices", {})
        texts = choices.get("text", [])
        labels = choices.get("label", [])
        answer_key = item.get("answerKey", "")

        scores = [evaluator.score_choice(item.get("question", ""), c) for c in texts]
        best_idx = int(torch.tensor(scores).argmax().item())
        pred_label = labels[best_idx] if best_idx < len(labels) else ""
        if str(pred_label).strip() == str(answer_key).strip():
            correct += 1
        total += 1
    acc = (correct / total) * 100.0 if total > 0 else 0.0
    print(f"ARC-Easy Acc: {acc:.2f}% ({correct}/{total}) | Time: {time.time() - t0:.1f}s")
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
        p1 = sentence.replace("_", item.get("option1", ""))
        p2 = sentence.replace("_", item.get("option2", ""))
        label = 0 if str(item.get("answer", "1")) == "1" else 1

        s1 = evaluator.score_choice("", p1)
        s2 = evaluator.score_choice("", p2)
        pred = 0 if s1 > s2 else 1
        if pred == label:
            correct += 1
        total += 1
    acc = (correct / total) * 100.0 if total > 0 else 0.0
    print(f"WinoGrande Acc: {acc:.2f}% ({correct}/{total}) | Time: {time.time() - t0:.1f}s")
    return {"accuracy": acc, "correct": correct, "total": total}


def main():
    parser = argparse.ArgumentParser(description="Evaluate any saved checkpoint on Phase B")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--samples", type=int, default=250, help="Samples per benchmark")
    parser.add_argument("--output", type=str, default="phase_b_eval_results.json")
    args = parser.parse_args()

    evaluator = CheckpointEvaluator(checkpoint_path=args.checkpoint)

    results = {}
    results["hellaswag"] = eval_hellaswag(evaluator, num_samples=args.samples)
    results["piqa"] = eval_piqa(evaluator, num_samples=args.samples)
    results["arc_easy"] = eval_arc_easy(evaluator, num_samples=args.samples)
    results["winogrande"] = eval_winogrande(evaluator, num_samples=args.samples)

    avg_acc = (results["hellaswag"]["accuracy"] + results["piqa"]["accuracy"] +
               results["arc_easy"]["accuracy"] + results["winogrande"]["accuracy"]) / 4.0
    results["average_accuracy"] = avg_acc

    print("\n" + "=" * 60)
    print(f"Phase B Checkpoint Results:")
    print(f"  HellaSwag  : {results['hellaswag']['accuracy']:.2f}%")
    print(f"  PIQA       : {results['piqa']['accuracy']:.2f}%")
    print(f"  ARC-Easy   : {results['arc_easy']['accuracy']:.2f}%")
    print(f"  WinoGrande : {results['winogrande']['accuracy']:.2f}%")
    print(f"  AVERAGE    : {avg_acc:.2f}%")
    print("=" * 60)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
