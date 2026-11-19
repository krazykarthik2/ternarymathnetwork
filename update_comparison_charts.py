"""
Publication Chart Generator:
Renders results/obj6_mige_beats_all.png and updates results/final_evaluation_results.json
to include all completed 10,000-step BitNet-125M paradigms:
1. TSGO (Ternary Swarm-Guided Optimization)
2. Temperature Annealing
3. MIGE-1 (Adaptive Sobolev)
4. STE Baseline
5. MIGE-NBS (Normalized Bounded-Sobolev)
6. MIGE-v2 (Gaussian Proximity Halo - Historical Ablation)
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np

def update_comparison_charts():
    os.makedirs("results", exist_ok=True)

    # 1. Load evaluation results
    eval_json_path = "results/final_evaluation_results.json"
    if not os.path.exists(eval_json_path):
        eval_data = {}
    else:
        with open(eval_json_path) as f:
            try:
                eval_data = json.load(f)
            except Exception:
                eval_data = {}

    # 2. Load training metrics if available
    ste_p = "checkpoints/125M_ste/metrics.json"
    ta_p = "checkpoints/125M_temp_anneal/metrics.json"
    mige_p = "checkpoints/125M_mige/metrics.json"
    migenbs_p = "checkpoints/125M_mige_nbs/metrics.json"
    migev2_p = "checkpoints/125M_mige_v2/metrics.json"
    tsgo_p = "checkpoints/125M_tsgo/metrics.json"

    metrics = {}
    if os.path.exists(ste_p):
        with open(ste_p) as f: metrics["STE"] = json.load(f)
    if os.path.exists(ta_p):
        with open(ta_p) as f: metrics["Temp Anneal"] = json.load(f)
    if os.path.exists(mige_p):
        with open(mige_p) as f: metrics["MIGE-1"] = json.load(f)
    if os.path.exists(migenbs_p):
        with open(migenbs_p) as f: metrics["MIGE-NBS"] = json.load(f)
    if os.path.exists(migev2_p):
        with open(migev2_p) as f: metrics["MIGE-v2"] = json.load(f)
    if os.path.exists(tsgo_p):
        with open(tsgo_p) as f: metrics["TSGO"] = json.load(f)

    # 3. Render 4-panel publication comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    colors = {
        "TSGO (Ternary Swarm, Step 10,000)": "#8e44ad",
        "Temperature Annealing (Step 10,000)": "#f39c12",
        "MIGE (Adaptive Sobolev, Step 10,000)": "#3498db",
        "STE Baseline (Step 10,000)": "#e74c3c",
        "MIGE-NBS (Normalized Bounded-Sobolev, Step 10,000)": "#2ecc71",
        "MIGE-v2 (Gaussian Proximity Halo, Step 10,000)": "#95a5a6"
    }
    short_names = {
        "TSGO (Ternary Swarm, Step 10,000)": "TSGO (Ours)",
        "Temperature Annealing (Step 10,000)": "Temp Anneal",
        "MIGE (Adaptive Sobolev, Step 10,000)": "MIGE-1",
        "STE Baseline (Step 10,000)": "STE",
        "MIGE-NBS (Normalized Bounded-Sobolev, Step 10,000)": "MIGE-NBS",
        "MIGE-v2 (Gaussian Proximity Halo, Step 10,000)": "MIGE-v2"
    }

    # Panel 1: Hardware Discrete PPL Curve
    ax1 = axes[0, 0]
    has_curves = False
    for k, name in [("TSGO", "TSGO (Ternary Swarm, Step 10,000)"),
                    ("Temp Anneal", "Temperature Annealing (Step 10,000)"), 
                    ("MIGE-1", "MIGE (Adaptive Sobolev, Step 10,000)"),
                    ("STE", "STE Baseline (Step 10,000)"),
                    ("MIGE-NBS", "MIGE-NBS (Normalized Bounded-Sobolev, Step 10,000)"),
                    ("MIGE-v2", "MIGE-v2 (Gaussian Proximity Halo, Step 10,000)")]:
        if k in metrics:
            has_curves = True
            m = metrics[k]
            lw = 3.0 if k == "TSGO" else 2.0
            ax1.plot(m["steps"], m["discrete_ppl"], label=f"{k} (Final: {m['discrete_ppl'][-1]:.1f})", color=colors[name], lw=lw)

    if not has_curves and eval_data:
        keys = [k for k in colors.keys() if k in eval_data]
        vals = [eval_data[k]["perplexity"] for k in keys]
        c = [colors[k] for k in keys]
        labels = [short_names[k] for k in keys]
        bars = ax1.bar(labels, vals, color=c, edgecolor="black")
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width()/2.0, v + 2.0, f"{v:.1f}", ha="center", fontweight="bold")
    ax1.set_title("(a) BitNet-125M Hardware Discrete Perplexity (snapped {-1,0,+1})", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Training Steps", fontsize=11)
    ax1.set_ylabel("Discrete Test Perplexity (Lower is Better)", fontsize=11)
    if has_curves:
        ax1.set_yscale("log")
        ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Discretization Gap Dynamics
    ax2 = axes[0, 1]
    has_gap = False
    for k, name in [("TSGO", "TSGO (Ternary Swarm, Step 10,000)"),
                    ("Temp Anneal", "Temperature Annealing (Step 10,000)"), 
                    ("MIGE-1", "MIGE (Adaptive Sobolev, Step 10,000)"),
                    ("STE", "STE Baseline (Step 10,000)")]:
        if k in metrics:
            has_gap = True
            m = metrics[k]
            lw = 2.8 if k == "TSGO" else 1.8
            ax2.plot(m["steps"], m["discretization_gap"], label=f"{k} Gap", color=colors[name], lw=lw)
    if not has_gap:
        ax2.text(0.5, 0.5, "Discretization Gap Dynamics", 
                 ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax2.set_title("(b) Discretization Gap Dynamics (Discrete PPL - Continuous PPL)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Training Steps", fontsize=11)
    ax2.set_ylabel("Discretization Gap", fontsize=11)
    ax2.axhline(0, color="black", linestyle="--", alpha=0.5)
    if has_gap:
        ax2.legend(loc="upper right", fontsize=10)
        ax2.set_ylim(-150, 1300)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Held-out Benchmark (Step 10,000)
    ax3 = axes[1, 0]
    candidate_keys = [
        "TSGO (Ternary Swarm, Step 10,000)",
        "Temperature Annealing (Step 10,000)",
        "MIGE (Adaptive Sobolev, Step 10,000)",
        "STE Baseline (Step 10,000)",
        "MIGE-NBS (Normalized Bounded-Sobolev, Step 10,000)",
        "MIGE-v2 (Gaussian Proximity Halo, Step 10,000)"
    ]
    keys = [k for k in candidate_keys if k in eval_data]
    labels = [short_names[k] for k in keys]
    ppl_vals = [eval_data[k]["perplexity"] for k in keys]
    c_list = [colors[k] for k in keys]

    if keys:
        bars = ax3.bar(labels, ppl_vals, color=c_list, width=0.55, edgecolor="black", linewidth=1.2)
        ax3.set_title("(c) Held-Out FineWeb-Edu Benchmark (245k Tokens)", fontsize=13, fontweight="bold")
        ax3.set_ylabel("Discrete Test Perplexity (Lower is Better)", fontsize=11)
        ax3.grid(axis="y", alpha=0.3)
        for b, val in zip(bars, ppl_vals):
            ax3.text(b.get_x() + b.get_width()/2.0, val + 1.5, f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Panel 4: Ternary Parameter Distribution (Sparsity Allocation)
    ax4 = axes[1, 1]
    if keys:
        x_idx = np.arange(3)
        n_bars = len(keys)
        width = 0.8 / n_bars
        offsets = np.linspace(-0.4 + width/2, 0.4 - width/2, n_bars)

        for idx, k in enumerate(keys):
            w = eval_data[k]["ternary_weights"]
            w_pct = [w["-1"] * 100, w["0"] * 100, w["+1"] * 100]
            ax4.bar(x_idx + offsets[idx], w_pct, width, label=short_names[k], color=colors[k], edgecolor="black")

        ax4.set_xticks(x_idx)
        ax4.set_xticklabels(["State -1", "State 0 (Sparsity)", "State +1"], fontsize=11, fontweight="bold")
        ax4.set_ylabel("Weight Fraction (%)", fontsize=11)
        ax4.set_title("(d) Quantized Parameter Allocation Across States", fontsize=13, fontweight="bold")
        ax4.legend(loc="upper right", fontsize=10)
        ax4.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_img = "results/obj6_mige_beats_all.png"
    plt.savefig(out_img, dpi=300)
    plt.close()
    print(f">> Successfully regenerated publication chart: {out_img}")

if __name__ == "__main__":
    update_comparison_charts()
