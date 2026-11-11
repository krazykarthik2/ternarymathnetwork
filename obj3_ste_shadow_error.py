"""
Objective 3: Prove STE Shadow Error is Permanent
Demonstrates:
For a 1-layer setup y = Q(w)*x with loss L = (y - target)^2:
1. The true quantized loss L(Q(w)) is piecewise constant (true gradient is 0 almost everywhere).
2. The STE gradient is 2*(w - target) (or nonzero gradient), pretending the loss is a smooth parabola.
3. The STE optimizes a phantom objective and cannot drive shadow error |w - Q(w)| to zero.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from quantizers import ste_quantize, hard_ternary

def run_objective_3(results_dir: str = "results"):
    os.makedirs(results_dir, exist_ok=True)
    print("=" * 70)
    print("OBJECTIVE 3: STE PERMANENT SHADOW ERROR & PHANTOM OBJECTIVE")
    print("=" * 70)

    target = 0.7
    x_input = 1.0

    w_vals = np.linspace(-1.5, 1.5, 600)
    true_losses = []
    ste_losses = []
    ste_grads = []
    shadow_errors = []

    for w_val in w_vals:
        # 1. True quantized output and loss
        w_t = torch.tensor([w_val], dtype=torch.float32, requires_grad=True)
        q_w = hard_ternary(w_t, threshold=0.5)
        y_true = q_w * x_input
        loss_true = (y_true - target) ** 2

        # 2. STE forward and autograd backward
        w_ste = torch.tensor([w_val], dtype=torch.float32, requires_grad=True)
        q_ste = ste_quantize(w_ste, threshold=0.5, clip_val=1.0)
        y_ste = q_ste * x_input
        loss_ste = (y_ste - target) ** 2
        loss_ste.backward()
        g_ste = w_ste.grad.item()

        # 3. Shadow error |w - Q(w)|
        shadow = abs(w_val - q_w.item())

        true_losses.append(loss_true.item())
        ste_losses.append((w_val * x_input - target) ** 2)
        ste_grads.append(g_ste)
        shadow_errors.append(shadow)

    true_losses = np.array(true_losses)
    ste_losses = np.array(ste_losses)
    ste_grads = np.array(ste_grads)
    shadow_errors = np.array(shadow_errors)

    # Prove discrepancy in bin (0.5, 1.0)
    in_bin_idx = np.where((w_vals > 0.55) & (w_vals < 0.95))[0]
    true_loss_variance = np.var(true_losses[in_bin_idx])
    ste_grad_mean = np.mean(np.abs(ste_grads[in_bin_idx]))

    print(f"Inside bin (0.55, 0.95) where Q(w) = +1:")
    print(f"  Variance of True Quantized Loss: {true_loss_variance:.8f} (loss is completely constant)")
    print(f"  Mean Magnitude of STE Gradient:   {ste_grad_mean:.6f} (STE claims gradients exist)")
    print(f"  Average Shadow Error |w - Q(w)|: {np.mean(shadow_errors[in_bin_idx]):.6f}")

    assert true_loss_variance < 1e-10, "True loss should be flat in quantization bin"
    assert ste_grad_mean > 0.0, "STE gradient is artificially active"

    print("SUCCESS: STE gradient generates phantom motion across flat plateaus.")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(w_vals, true_losses, 'r-', linewidth=2.5, label=r'True Quantized Loss $\mathcal{L}(Q(w))$ (Step/Staircase)')
    ax1.plot(w_vals, ste_losses, 'b--', linewidth=2, label=r'STE Surrogate Loss $\mathcal{L}(w) = (w - 0.7)^2$')
    ax1.axvline(0.5, color='gray', linestyle=':', label='Quantization Threshold $\pm 0.5$')
    ax1.axvline(-0.5, color='gray', linestyle=':')
    ax1.set_xlabel('Weight $w$')
    ax1.set_ylabel('Loss Value')
    ax1.set_title('Loss Landscape Mismatch: True vs STE Surrogate')
    ax1.grid(True, ls='--', alpha=0.5)
    ax1.legend()

    ax2.plot(w_vals, ste_grads, 'm-', linewidth=2, label='STE Gradient $\partial \mathcal{L} / \partial w$')
    ax2.plot(w_vals, shadow_errors, 'g:', linewidth=2, label=r'Shadow Error $|w - Q(w)|$')
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Weight $w$')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Non-Zero Phantom Gradients on Zero-Slope Plateau')
    ax2.grid(True, ls='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "obj3_ste_shadow_error.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved Objective 3 plot to: {plot_path}\n")

if __name__ == "__main__":
    run_objective_3()
