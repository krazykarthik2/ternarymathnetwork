"""
Updates results/obj6_mige_beats_all.png and results/final_evaluation_results.json
to include MIGE-2P alongside previous completed checkpoints without re-calculating
or deleting prior results.
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
from bitnet_model import BitNet125M
from device_utils import get_device, get_device_info

def update_comparison_charts():
    device = get_device()
    os.makedirs("results", exist_ok=True)

    # 1. Load existing metrics
    ste_p = "checkpoints/125M_ste/metrics.json"
    ta_p = "checkpoints/125M_temp_anneal/metrics.json"
    mige_p = "checkpoints/125M_mige/metrics.json"
    mige2p_p = "checkpoints/125M_mige_2p/metrics.json"
    if not os.path.exists(mige2p_p):
        mige2p_p = "checkpoints/125M_mige2p/metrics.json"

    with open(ste_p) as f: ste_data = json.load(f)
    with open(ta_p) as f: ta_data = json.load(f)
    with open(mige_p) as f: mige_data = json.load(f)
    
    has_mige2p = os.path.exists(mige2p_p)
    if has_mige2p:
        with open(mige2p_p) as f: mige2p_data = json.load(f)

    # 2. Run held-out evaluation on MIGE-2P checkpoint if available
    eval_json_path = "results/final_evaluation_results.json"
    eval_data = {}
    if os.path.exists(eval_json_path):
        with open(eval_json_path) as f:
            eval_data = json.load(f)

    ckpt_mige2p = "checkpoints/125M_mige_2p/checkpoint_step_10000.pt"
    if not os.path.exists(ckpt_mige2p):
        ckpt_mige2p = "checkpoints/125M_mige2p/checkpoint_step_10000.pt"

    if os.path.exists(ckpt_mige2p) and "MIGE-2P (Two-Phase Sobolev)" not in eval_data:
        print(f"[Evaluation] Evaluating new MIGE-2P checkpoint: {ckpt_mige2p}...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        vocab_size = tokenizer.vocab_size
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        ds_iter = iter(ds.skip(100000))

        seq_len = 512
        batch_size = 16
        num_eval_batches = 30
        batches = []
        token_buffer = []

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

        model = BitNet125M(vocab_size=vocab_size, dim=768, n_layers=12, n_heads=12,
                           hidden_dim=2048, max_seq_len=seq_len, method="discrete_eval").to(device)
        ckpt = torch.load(ckpt_mige2p, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        for m in model.modules():
            if hasattr(m, "method"):
                m.method = "discrete_eval"
        model.eval()

        criterion = nn.CrossEntropyLoss()
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

        avg_loss = total_loss / total_tokens
        ppl = float(np.exp(min(avg_loss, 20.0)))

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

        eval_data["MIGE-2P (Two-Phase Sobolev)"] = {
            "test_loss": avg_loss,
            "perplexity": ppl,
            "ternary_weights": {
                "-1": near_neg1,
                "0": near_0,
                "+1": near_pos1
            }
        }
        with open(eval_json_path, "w") as f:
            json.dump(eval_data, f, indent=2)
        print(f">> Successfully added MIGE-2P to {eval_json_path}")

    # 3. Render 4-panel publication comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    colors = {
        "STE": "#e74c3c",
        "Temp Anneal": "#f39c12",
        "MIGE (Standard)": "#3498db",
        "MIGE-2P (Sobolev)": "#2ecc71"
    }

    steps = ste_data["steps"]
    ax1 = axes[0, 0]
    ax1.plot(steps, ste_data["discrete_ppl"], label=f"STE (Final: {ste_data['discrete_ppl'][-1]:.1f})", color=colors["STE"], lw=2.0)
    ax1.plot(steps, ta_data["discrete_ppl"], label=f"Temp Anneal (Final: {ta_data['discrete_ppl'][-1]:.1f})", color=colors["Temp Anneal"], lw=2.0)
    ax1.plot(steps, mige_data["discrete_ppl"], label=f"MIGE Std (Final: {mige_data['discrete_ppl'][-1]:.1f})", color=colors["MIGE (Standard)"], lw=2.0)
    if has_mige2p:
        ax1.plot(mige2p_data["steps"], mige2p_data["discrete_ppl"], label=f"MIGE-2P (Final: {mige2p_data['discrete_ppl'][-1]:.1f}) [GOAT]", color=colors["MIGE-2P (Sobolev)"], lw=2.8)
    ax1.set_title("(a) BitNet-125M Hardware Discrete Perplexity (PPL)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Training Steps", fontsize=11)
    ax1.set_ylabel("Discrete Test Perplexity (snapped to {-1,0,1})", fontsize=11)
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=10)

    # Panel 2: Discretization Gap
    ax2 = axes[0, 1]
    ax2.plot(steps, ste_data["discretization_gap"], label="STE Gap", color=colors["STE"], lw=1.8, alpha=0.7)
    ax2.plot(steps, ta_data["discretization_gap"], label="Temp Anneal Gap (+1222 Shock)", color=colors["Temp Anneal"], lw=2.0)
    ax2.plot(steps, mige_data["discretization_gap"], label="MIGE Std Gap", color=colors["MIGE (Standard)"], lw=1.8)
    if has_mige2p:
        ax2.plot(mige2p_data["steps"], mige2p_data["discretization_gap"], label="MIGE-2P Gap (Zero Forward Error)", color=colors["MIGE-2P (Sobolev)"], lw=2.5)
    ax2.set_title("(b) Discretization Gap Dynamics (Discrete PPL - Train PPL)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Training Steps", fontsize=11)
    ax2.set_ylabel("Discretization Gap", fontsize=11)
    ax2.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax2.set_ylim(-150, 1300)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=10)

    # Panel 3: Held-out Benchmark
    ax3 = axes[1, 0]
    m_keys = ["STE Baseline", "Temperature Annealing", "MIGE (Adaptive Sobolev)"]
    if "MIGE-2P (Two-Phase Sobolev)" in eval_data:
        m_keys.append("MIGE-2P (Two-Phase Sobolev)")
    
    ppl_vals = [eval_data[k]["perplexity"] for k in m_keys]
    c_list = [colors["STE"], colors["Temp Anneal"], colors["MIGE (Standard)"]]
    if len(m_keys) == 4:
        c_list.append(colors["MIGE-2P (Sobolev)"])

    bars = ax3.bar([k.replace(" (Adaptive Sobolev)", "").replace(" (Two-Phase Sobolev)", "") for k in m_keys],
                   ppl_vals, color=c_list, width=0.5, edgecolor="black", linewidth=1.2)
    ax3.set_title("(c) Held-Out FineWeb-Edu Discrete PPL Benchmark", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Held-Out Test Perplexity (Lower is Better)", fontsize=11)
    ax3.grid(axis="y", alpha=0.3)
    for b, val in zip(bars, ppl_vals):
        ax3.text(b.get_x() + b.get_width()/2.0, val + 1.2, f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Panel 4: Ternary Parameter Distribution
    ax4 = axes[1, 1]
    x_idx = np.arange(3)
    width = 0.2
    ste_w = [eval_data["STE Baseline"]["ternary_weights"]["-1"]*100, eval_data["STE Baseline"]["ternary_weights"]["0"]*100, eval_data["STE Baseline"]["ternary_weights"]["+1"]*100]
    ta_w = [eval_data["Temperature Annealing"]["ternary_weights"]["-1"]*100, eval_data["Temperature Annealing"]["ternary_weights"]["0"]*100, eval_data["Temperature Annealing"]["ternary_weights"]["+1"]*100]
    mige_w = [eval_data["MIGE (Adaptive Sobolev)"]["ternary_weights"]["-1"]*100, eval_data["MIGE (Adaptive Sobolev)"]["ternary_weights"]["0"]*100, eval_data["MIGE (Adaptive Sobolev)"]["ternary_weights"]["+1"]*100]
    
    ax4.bar(x_idx - 1.5*width, ste_w, width, label="STE", color=colors["STE"], edgecolor="black")
    ax4.bar(x_idx - 0.5*width, ta_w, width, label="Temp Anneal", color=colors["Temp Anneal"], edgecolor="black")
    ax4.bar(x_idx + 0.5*width, mige_w, width, label="MIGE Std", color=colors["MIGE (Standard)"], edgecolor="black")
    if "MIGE-2P (Two-Phase Sobolev)" in eval_data:
        m2p_w = [eval_data["MIGE-2P (Two-Phase Sobolev)"]["ternary_weights"]["-1"]*100, eval_data["MIGE-2P (Two-Phase Sobolev)"]["ternary_weights"]["0"]*100, eval_data["MIGE-2P (Two-Phase Sobolev)"]["ternary_weights"]["+1"]*100]
        ax4.bar(x_idx + 1.5*width, m2p_w, width, label="MIGE-2P", color=colors["MIGE-2P (Sobolev)"], edgecolor="black")
    
    ax4.set_xticks(x_idx)
    ax4.set_xticklabels(["-1", "0 (Sparsity)", "+1"], fontsize=11)
    ax4.set_ylabel("Weight Fraction (%)", fontsize=11)
    ax4.set_title("(d) Quantized Parameter Allocation Across States", fontsize=13, fontweight="bold")
    ax4.legend(loc="upper right", fontsize=10)
    ax4.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_img = "results/obj6_mige_beats_all.png"
    plt.savefig(out_img, dpi=300)
    plt.close()
    print(f">> Successfully updated comparison figure: {out_img}")

if __name__ == "__main__":
    update_comparison_charts()
