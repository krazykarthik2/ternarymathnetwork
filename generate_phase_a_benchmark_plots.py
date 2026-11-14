"""
Generates the authoritative publication benchmark figure: results/obj6_mige_beats_all.png
Based on the completed full 10,000-step BitNet-125M pre-training runs on FineWeb-Edu (Phase A)
and held-out discrete evaluation.
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("results", exist_ok=True)

# Load metrics
with open("checkpoints/125M_ste/metrics.json") as f:
    ste_data = json.load(f)
with open("checkpoints/125M_temp_anneal/metrics.json") as f:
    ta_data = json.load(f)
with open("checkpoints/125M_mige/metrics.json") as f:
    mige_data = json.load(f)
with open("results/final_evaluation_results.json") as f:
    eval_data = json.load(f)

steps = ste_data["steps"]

# Setup 2x2 publication dashboard
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
colors = {"STE": "#e74c3c", "Temp Anneal": "#f39c12", "MIGE": "#2ecc71"}

# Panel 1: Discrete Perplexity Trajectory
ax1 = axes[0, 0]
ax1.plot(steps, ste_data["discrete_ppl"], label=f"STE (Final: {ste_data['discrete_ppl'][-1]:.1f})", color=colors["STE"], lw=2.2)
ax1.plot(steps, ta_data["discrete_ppl"], label=f"Temp Anneal (Final: {ta_data['discrete_ppl'][-1]:.1f})", color=colors["Temp Anneal"], lw=2.2)
ax1.plot(steps, mige_data["discrete_ppl"], label=f"MIGE (Final: {mige_data['discrete_ppl'][-1]:.1f})", color=colors["MIGE"], lw=2.5)
ax1.set_title("(a) BitNet-125M True Discrete Hardware Perplexity (PPL)", fontsize=13, fontweight="bold")
ax1.set_xlabel("Training Steps", fontsize=11)
ax1.set_ylabel("Discrete Test Perplexity (snapped to {-1,0,1})", fontsize=11)
ax1.set_yscale("log")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper right", fontsize=11)

# Panel 2: Discretization Gap Dynamics (Surrogate Loss vs Discrete True Loss)
ax2 = axes[0, 1]
ax2.plot(steps, ste_data["discretization_gap"], label="STE Gap", color=colors["STE"], lw=1.8, alpha=0.8)
ax2.plot(steps, ta_data["discretization_gap"], label="Temp Anneal Gap (Violent Early Mismatch)", color=colors["Temp Anneal"], lw=2.0)
ax2.plot(steps, mige_data["discretization_gap"], label="MIGE Gap (Stable Discrete Match)", color=colors["MIGE"], lw=2.2)
ax2.set_title("(b) Discretization Gap (Discrete PPL - Train PPL)", fontsize=13, fontweight="bold")
ax2.set_xlabel("Training Steps", fontsize=11)
ax2.set_ylabel("Discretization Gap", fontsize=11)
ax2.axhline(0, color="black", linestyle="--", alpha=0.5)
ax2.set_ylim(-150, 1300)
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper right", fontsize=11)

# Panel 3: Discrete Held-Out FineWeb-Edu Benchmark Comparison (Bar Chart)
ax3 = axes[1, 0]
methods = ["STE Baseline", "MIGE (Sobolev)", "Temperature Annealing"]
ppls = [
    eval_data["STE Baseline"]["perplexity"],
    eval_data["MIGE (Adaptive Sobolev)"]["perplexity"],
    eval_data["Temperature Annealing"]["perplexity"]
]
bar_colors = [colors["STE"], colors["MIGE"], colors["Temp Anneal"]]
bars = ax3.bar(methods, ppls, color=bar_colors, width=0.5, edgecolor="black", linewidth=1.2)
ax3.set_title("(c) Held-Out Benchmark Perplexity on Discrete {-1, 0, 1} Weights", fontsize=13, fontweight="bold")
ax3.set_ylabel("Held-Out Test Perplexity (Lower is Better)", fontsize=11)
ax3.grid(axis="y", alpha=0.3)
for bar, ppl in zip(bars, ppls):
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{ppl:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

# Panel 4: Ternary Weight Distribution Balance at Step 10,000
ax4 = axes[1, 1]
x_indices = np.arange(3)
width = 0.25
ste_w = [eval_data["STE Baseline"]["ternary_weights"]["-1"]*100, eval_data["STE Baseline"]["ternary_weights"]["0"]*100, eval_data["STE Baseline"]["ternary_weights"]["+1"]*100]
ta_w = [eval_data["Temperature Annealing"]["ternary_weights"]["-1"]*100, eval_data["Temperature Annealing"]["ternary_weights"]["0"]*100, eval_data["Temperature Annealing"]["ternary_weights"]["+1"]*100]
mige_w = [eval_data["MIGE (Adaptive Sobolev)"]["ternary_weights"]["-1"]*100, eval_data["MIGE (Adaptive Sobolev)"]["ternary_weights"]["0"]*100, eval_data["MIGE (Adaptive Sobolev)"]["ternary_weights"]["+1"]*100]

ax4.bar(x_indices - width, ste_w, width, label="STE", color=colors["STE"], edgecolor="black")
ax4.bar(x_indices, ta_w, width, label="Temp Anneal", color=colors["Temp Anneal"], edgecolor="black")
ax4.bar(x_indices + width, mige_w, width, label="MIGE", color=colors["MIGE"], edgecolor="black")
ax4.set_xticks(x_indices)
ax4.set_xticklabels(["-1 (Negative)", "0 (Zero / Sparsity)", "+1 (Positive)"], fontsize=11)
ax4.set_ylabel("Weight Fraction (%)", fontsize=11)
ax4.set_title("(d) Quantized Parameter Allocation Across States", fontsize=13, fontweight="bold")
ax4.legend(loc="upper right", fontsize=11)
ax4.grid(axis="y", alpha=0.3)

plt.tight_layout()
output_path = "results/obj6_mige_beats_all.png"
plt.savefig(output_path, dpi=300)
plt.close()
print(f"Generated updated benchmark plot: {output_path}")
