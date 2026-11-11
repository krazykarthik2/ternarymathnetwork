"""
Phase B: Downstream Zero-Shot Accuracy Evaluation Harness.
Interfaces with lm-evaluation-harness to benchmark ternary BitNet checkpoints
on standard zero-shot tasks:
1. HellaSwag
2. PIQA
3. ARC-Easy / ARC-Challenge
4. WinoGrande

Provides standard PyTorch wrapper matching HuggingFace CausalLM interface
to load discrete {-1, 0, 1} quantized weights and evaluate zero-shot accuracies.
"""
import os
import argparse
import json
import torch
import torch.nn as nn
from bitnet_model import BitNet125M
from quantizers import hard_ternary
from device_utils import get_device

class BitNetForCausalLM(nn.Module):
    """HuggingFace-compatible wrapper for lm-evaluation-harness."""
    def __init__(self, checkpoint_path, scale="125M", vocab_size=50257):
        super().__init__()
        if scale == "125M":
            self.model = BitNet125M(vocab_size=vocab_size, dim=768, n_layers=12, n_heads=12,
                                    hidden_dim=2048, method="discrete_eval")
        elif scale == "350M":
            self.model = BitNet125M(vocab_size=vocab_size, dim=1024, n_layers=24, n_heads=16,
                                    hidden_dim=2816, method="discrete_eval")
        
        # Load weights
        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location="cpu")
            model_state = state.get("model_state", state)
            self.model.load_state_dict(model_state, strict=False)
            print(f"[CausalLM] Loaded weights from: {checkpoint_path}")

        # Enforce exact hardware discrete quantization
        for m in self.model.modules():
            if hasattr(m, "method"):
                m.method = "discrete_eval"

    def forward(self, input_ids, **kwargs):
        logits = self.model(input_ids)
        return type("CausalLMOutput", (), {"logits": logits})()

def evaluate_downstream(checkpoint_path, scale="125M", tasks=["hellaswag", "piqa", "arc_easy", "winogrande"]):
    device = get_device()
    print("=" * 80)
    print(f"PHASE B: ZERO-SHOT DOWNSTREAM EVALUATION ({scale})")
    print(f"Tasks: {tasks}")
    print("=" * 80)

    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        
        model_wrapper = BitNetForCausalLM(checkpoint_path, scale=scale)
        model_wrapper.to(device).eval()

        hf_lm = HFLM(pretrained=model_wrapper)
        results = lm_eval.simple_evaluate(
            model=hf_lm,
            tasks=tasks,
            num_fewshot=0,
            batch_size=16,
            device=str(device)
        )
        print("\n--- ZERO-SHOT ACCURACY RESULTS ---")
        for task, res in results["results"].items():
            acc = res.get("acc,none", res.get("acc", 0.0))
            print(f"{task:<20}: {acc * 100:.2f}%")
        return results
    except ImportError:
        print("[Notice] 'lm_eval' package not installed. Simulating harness pipeline check...")
        print("To run full suite on VM GPU: pip install lm-eval datasets")
        return {"status": "ready_for_gpu_eval"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--scale", type=str, default="125M")
    parser.add_argument("--tasks", nargs="+", default=["hellaswag", "piqa", "arc_easy", "winogrande"])
    args = parser.parse_args()
    evaluate_downstream(args.checkpoint, scale=args.scale, tasks=args.tasks)
