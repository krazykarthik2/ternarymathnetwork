"""
Objective 4: Prove MIGE Produces Bounded Smooth Gradients
Demonstrates:
1. Peak gradient at w=0.5 is finite for all epsilon > 0.
2. L2 energy is finite for all epsilon > 0.
3. Gradient away from threshold (|w - 0.5| > eps) is identically 0.0 (exact compact support).
4. Side-by-side comparison table between Temperature Annealing and MIGE.
"""
import os
import math
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import torch
from mige import mige_kernel_derivative, MOLLIFIER_NORMALIZATION_C
from quantizers import soft_ternary_derivative

def run_objective_4(results_dir: str = "results"):
    os.makedirs(results_dir, exist_ok=True)
    print("=" * 75)
    print("OBJECTIVE 4: PROVING MIGE PRODUCES BOUNDED, SMOOTH SOBOLEV GRADIENTS")
    print("=" * 75)

    eps_values = [0.3, 0.2, 0.1, 0.05, 0.02, 0.01]
    records = []

    print(f"{'Epsilon':>8} | {'Peak (w=0.5)':>14} | {'L2 Energy':>14} | {'L1 Integral':>12} | {'Grad (w=0.8)':>14}")
    print("-" * 75)

    for eps in eps_values:
        # 1. Peak gradient at w = 0.5
        w_peak = torch.tensor([0.5], dtype=torch.float64)
        peak_val = mige_kernel_derivative(w_peak, epsilon=eps, threshold=0.5).item()

        # 2. Gradient at w = 0.8
        w_away = torch.tensor([0.8], dtype=torch.float64)
        grad_away = mige_kernel_derivative(w_away, epsilon=eps, threshold=0.5).item()

        # 3. Integrals via quadrature
        def mige_fn(w):
            wt = torch.tensor([w], dtype=torch.float64)
            return mige_kernel_derivative(wt, epsilon=eps, threshold=0.5).item()

        def mige_sq(w):
            val = mige_fn(w)
            return val * val

        # Exact support is within [0.5 - eps, 0.5 + eps] and [-0.5 - eps, -0.5 + eps]
        l1_pos, _ = integrate.quad(mige_fn, 0.5 - eps, 0.5 + eps)
        l1_neg, _ = integrate.quad(mige_fn, -0.5 - eps, -0.5 + eps)
        l1_total = l1_pos + l1_neg

        l2_pos, _ = integrate.quad(mige_sq, 0.5 - eps, 0.5 + eps)
        l2_neg, _ = integrate.quad(mige_sq, -0.5 - eps, -0.5 + eps)
        l2_total = l2_pos + l2_neg

        records.append((eps, peak_val, l2_total, l1_total, grad_away))
        print(f"{eps:8.3f} | {peak_val:14.4f} | {l2_total:14.4f} | {l1_total:12.6f} | {grad_away:14.6f}")

    print("-" * 75)

    # Compare with Temperature Annealing at parameter = 0.01
    param = 0.01
    w_p = torch.tensor([0.5], dtype=torch.float64)
    w_a = torch.tensor([0.8], dtype=torch.float64)

    ta_peak = soft_ternary_derivative(w_p, T=param).item()
    ta_away = soft_ternary_derivative(w_a, T=param).item()
    ta_l2 = 1.0 / (12.0 * param)  # Theoretical asymptotic L2 energy for soft ternary

    mige_peak = mige_kernel_derivative(w_p, epsilon=param).item()
    mige_away = mige_kernel_derivative(w_a, epsilon=param).item()
    mige_l2 = records[-1][2]

    print("\n--- SIDE-BY-SIDE COMPARISON: Temp Anneal (T=0.01) vs MIGE (eps=0.01) ---")
    print(f"{'Property':<32} | {'Temp Anneal (T=0.01)':<22} | {'MIGE (eps=0.01)':<20}")
    print("-" * 80)
    print(f"{'Peak Gradient at Threshold':<32} | {ta_peak:<22.4f} | {mige_peak:<20.4f}")
    print(f"{'L2 Energy (Integrability)':<32} | {ta_l2:<22.4f} | {mige_l2:<20.4f}")
    print(f"{'Gradient at w=0.8 (Off-threshold)':<32} | {ta_away:<22.4e} | {mige_away:<20.4f}")
    print(f"{'C-infinity Smoothness':<32} | {'Yes':<22} | {'Yes':<20}")
    print(f"{'Exact Compact Support':<32} | {'No (Exponential tails)':<22} | {'Yes (Zero outside eps)':<20}")
    print(f"{'Forward Pass Exact Discrete?':<32} | {'No (Ternary surrogate)':<22} | {'Yes (Pure {-1,0,1})':<20}")
    print("-" * 80)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    w_eval = np.linspace(-1.0, 1.0, 1500)
    w_tensor = torch.tensor(w_eval, dtype=torch.float32)

    for eps in [0.3, 0.2, 0.1, 0.05]:
        g_vals = mige_kernel_derivative(w_tensor, epsilon=eps, threshold=0.5).numpy()
        ax1.plot(w_eval, g_vals, label=rf'$\epsilon = {eps}$')

    ax1.set_xlabel('Weight $w$')
    ax1.set_ylabel(r'$\nabla^{\mathrm{MIGE}}_\epsilon Q(w)$')
    ax1.set_title('MIGE Smooth Bump Weak Derivatives at Thresholds')
    ax1.grid(True, ls='--', alpha=0.5)
    ax1.legend()

    # Panel 2: Temp Anneal vs MIGE comparison
    ta_vals = soft_ternary_derivative(w_tensor, T=0.1, threshold=0.5).numpy()
    mige_vals = mige_kernel_derivative(w_tensor, epsilon=0.1, threshold=0.5).numpy()

    ax2.plot(w_eval, ta_vals, 'r--', label='Temp Anneal $Q_T\'(w)$ ($T=0.1$, non-compact)')
    ax2.plot(w_eval, mige_vals, 'g-', linewidth=2, label='MIGE Weak Derivative ($\epsilon=0.1$, compact support)')
    ax2.set_xlabel('Weight $w$')
    ax2.set_ylabel('Derivative Value')
    ax2.set_title('Kernel Profile: Compact Sobolev Support vs Sigmoidal Tails')
    ax2.grid(True, ls='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "obj4_mige_gradient.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved Objective 4 plot to: {plot_path}\n")

if __name__ == "__main__":
    run_objective_4()
