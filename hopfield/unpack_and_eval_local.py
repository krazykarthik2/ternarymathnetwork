"""
hopfield/unpack_and_eval_local.py

Fast Local Evaluation Harness for 1.58-bit Packed Checkpoints:
1. Unpacks 2-bit compressed uint8 weights into exact hardware ternary {-1, 0, +1} * gamma.
2. Injects them into HopfieldBitNet125M.
3. Evaluates downstream Phase B zero-shot tasks (HellaSwag, PIQA, ARC-Easy, WinoGrande) locally.
"""
import os
import sys
import json
import time
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hopfield_bitnet import HopfieldBitNet125M


def unpack_ternary_tensor(packed: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Unpacks 2-bit unsigned integers (00=0, 01=+1, 11=-1) from uint8 bytes."""
    # Extract 4 2-bit values per byte
    v0 = packed & 0x03
    v1 = (packed >> 2) & 0x03
    v2 = (packed >> 4) & 0x03
    v3 = (packed >> 6) & 0x03

    unpacked_flat = torch.stack([v0, v1, v2, v3], dim=-1).flatten()
    numel = shape.numel()
    unpacked_flat = unpacked_flat[:numel]

    # Map {3: -1.0, 1: 1.0, 0: 0.0}
    w_q = torch.where(unpacked_flat == 1, 1.0, torch.where(unpacked_flat == 3, -1.0, 0.0))
    return w_q.view(shape)


def load_packed_model(packed_path: str, device="cpu") -> HopfieldBitNet125M:
    print(f"--> Loading and unpacking 1.58-bit compressed checkpoint: {packed_path}")
    t0 = time.time()
    packed_dict = torch.load(packed_path, map_location="cpu")

    model = HopfieldBitNet125M(vocab_size=50257, dim=768, n_layers=12, n_heads=12,
                               hidden_dim=2048, method="discrete").to(device)

    state_dict = {}
    for name, item in packed_dict.items():
        if isinstance(item, dict) and item.get("is_ternary", False):
            # Unpack 2-bit ternary weights and scale by gamma
            w_q = unpack_ternary_tensor(item["packed"], item["shape"]).to(device)
            gamma = item["gamma"].to(device).float()
            state_dict[name] = w_q * gamma
        else:
            state_dict[name] = item.to(device).float()

    model.load_state_dict(state_dict, strict=False)
    for m in model.modules():
        if hasattr(m, "method"):
            m.method = "discrete"
    model.eval()

    print(f"--> Unpacked and initialized 169M parameters in {(time.time() - t0)*1000:.2f} ms")
    return model


class LocalPackedEvaluator:
    def __init__(self, packed_path: str, device="cpu"):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = load_packed_model(packed_path, device=self.device)

    def score_choice(self, prompt, choice):
        full_text = prompt + (" " if prompt and not prompt.endswith(" ") else "") + str(choice)
        enc_full = self.tokenizer.encode(full_text)
        enc_prompt = self.tokenizer.encode(prompt)

        input_ids = torch.tensor([enc_full], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids)
            log_probs = F.log_softmax(logits[0, :-1], dim=-1)

            target_ids = input_ids[0, 1:]
            choice_start = max(0, len(enc_prompt) - 1)
            choice_targets = target_ids[choice_start:]
            choice_log_probs = log_probs[choice_start:]

            score = choice_log_probs.gather(1, choice_targets.unsqueeze(1)).sum().item()
            return score / max(1, len(choice_targets))


def run_local_eval(evaluator, samples=50):
    results = {}
    print(f"\n============================================================")
    print(f" Running Local CPU Phase B Evaluation ({samples} samples per task)")
    print(f"============================================================")

    # 1. HellaSwag
    print(f"\n[1/4] Evaluating HellaSwag...")
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
        if total >= samples:
            break
        scores = [evaluator.score_choice(item.get("ctx", ""), end) for end in item.get("endings", [])]
        if int(torch.tensor(scores).argmax().item()) == int(item.get("label", 0)):
            correct += 1
        total += 1
    acc_hella = (correct / total) * 100.0 if total > 0 else 0.0
    print(f"--> HellaSwag: {acc_hella:.2f}% ({correct}/{total}) in {time.time() - t0:.1f}s")
    results["hellaswag"] = acc_hella

    # 2. PIQA
    print(f"\n[2/4] Evaluating PIQA...")
    parquet_url = "https://huggingface.co/datasets/baber/piqa/resolve/main/piqa_validation.parquet"
    try:
        ds = load_dataset("parquet", data_files={"validation": parquet_url}, split="validation", streaming=True)
    except Exception:
        try:
            parquet_backup = "https://huggingface.co/datasets/ybisk/piqa/resolve/main/plain_text/validation-00000-of-00001.parquet"
            ds = load_dataset("parquet", data_files={"validation": parquet_backup}, split="validation", streaming=True)
        except Exception:
            ds = None

    if ds is not None:
        correct, total = 0, 0
        t0 = time.time()
        for item in ds:
            if total >= samples:
                break
            s1 = evaluator.score_choice(item.get("goal", ""), item.get("sol1", ""))
            s2 = evaluator.score_choice(item.get("goal", ""), item.get("sol2", ""))
            pred = 0 if s1 > s2 else 1
            if pred == int(item.get("label", 0)):
                correct += 1
            total += 1
        acc_piqa = (correct / total) * 100.0 if total > 0 else 0.0
        print(f"--> PIQA: {acc_piqa:.2f}% ({correct}/{total}) in {time.time() - t0:.1f}s")
        results["piqa"] = acc_piqa
    else:
        results["piqa"] = 0.0

    # 3. ARC-Easy
    print(f"\n[3/4] Evaluating ARC-Easy...")
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
        if total >= samples:
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
    acc_arc = (correct / total) * 100.0 if total > 0 else 0.0
    print(f"--> ARC-Easy: {acc_arc:.2f}% ({correct}/{total}) in {time.time() - t0:.1f}s")
    results["arc_easy"] = acc_arc

    # 4. WinoGrande
    print(f"\n[4/4] Evaluating WinoGrande...")
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
        if total >= samples:
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
    acc_wino = (correct / total) * 100.0 if total > 0 else 0.0
    print(f"--> WinoGrande: {acc_wino:.2f}% ({correct}/{total}) in {time.time() - t0:.1f}s")
    results["winogrande"] = acc_wino

    avg_acc = (results["hellaswag"] + results["piqa"] + results["arc_easy"] + results["winogrande"]) / 4.0
    results["average_accuracy"] = avg_acc

    print("\n" + "=" * 60)
    print(f"Local 1.58-bit Evaluation Summary:")
    print(f"  HellaSwag  : {results['hellaswag']:.2f}%")
    print(f"  PIQA       : {results['piqa']:.2f}%")
    print(f"  ARC-Easy   : {results['arc_easy']:.2f}%")
    print(f"  WinoGrande : {results['winogrande']:.2f}%")
    print(f"  Average    : {avg_acc:.2f}%")
    print("=" * 60)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate packed 1.58-bit model locally")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()

    evaluator = LocalPackedEvaluator(packed_path=args.checkpoint)
    run_local_eval(evaluator, samples=args.samples)


if __name__ == "__main__":
    main()
