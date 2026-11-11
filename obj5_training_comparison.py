"""
Objective 5: True Discrete Evaluation Benchmark
Compares STE, Temperature Annealing, and MIGE on a multi-layer deep network
with BitNet absmean scaling, measuring:
1. Training Loss Trajectory
2. True Post-Training Discrete Evaluation Loss (Quantized Weights in {-1, 0, +1})
3. Weight Recovery Accuracy vs Ground Truth Discrete Topology
4. Gradient Variance Telemetry
"""
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from device_utils import get_device, get_device_info
from quantizers import ste_quantize, temp_anneal_quantize, hard_ternary
from mige import mige_quantize

def run_objective_5(results_dir: str = "results", num_steps: int = 800):
    os.makedirs(results_dir, exist_ok=True)
    device = get_device()
    print("=" * 80)
    print("OBJECTIVE 5: DISCRETE BENCHMARK & BITNET POST-TRAINING INFERENCE")
    print(f"Executing on: {get_device_info()['name']}")
    print("=" * 80)

    torch.manual_seed(42)
    np.random.seed(42)

    # 2-layer Teacher Network with exact ternary {-1, 0, +1} weights
    N, D_in, D_hidden, D_out = 1024, 128, 128, 64
    X = torch.randn(N, D_in, device=device)
    W1_true = torch.randint(-1, 2, (D_hidden, D_in), device=device).float()
    W2_true = torch.randint(-1, 2, (D_out, D_hidden), device=device).float()
    
    # Ground truth forward pass
    Y = torch.tanh(X @ W1_true.t()) @ W2_true.t()

    methods = ['ste', 'temp_anneal', 'mige']
    results = {}

    for method in methods:
        print(f"\n---> Training Network using: {method.upper()}")
        torch.manual_seed(1337)
        W1 = nn.Parameter(torch.randn(D_hidden, D_in, device=device) * 0.1)
        W2 = nn.Parameter(torch.randn(D_out, D_hidden, device=device) * 0.1)
        optimizer = torch.optim.AdamW([W1, W2], lr=4e-3, weight_decay=1e-4)

        train_losses = []
        grad_variances = []

        for step in range(num_steps):
            optimizer.zero_grad()
            g1 = W1.abs().mean().clamp(min=1e-4)
            g2 = W2.abs().mean().clamp(min=1e-4)

            if method == 'ste':
                q1 = ste_quantize(W1 / g1)
                q2 = ste_quantize(W2 / g2)
            elif method == 'temp_anneal':
                T = 0.01 + 0.5 * (1.0 - 0.01) * (1.0 + math.cos(math.pi * step / num_steps))
                q1 = temp_anneal_quantize(W1 / g1, T=T)
                q2 = temp_anneal_quantize(W2 / g2, T=T)
            else: # mige
                eps = 0.05 + 0.5 * (0.35 - 0.05) * (1.0 + math.cos(math.pi * step / num_steps))
                q1 = mige_quantize(W1 / g1, epsilon=eps, floor_grad=0.02)
                q2 = mige_quantize(W2 / g2, epsilon=eps, floor_grad=0.02)

            pred = torch.tanh(X @ q1.t()) @ q2.t()
            loss = nn.functional.mse_loss(pred, Y)
            loss.backward()

            # Record gradient variance
            all_grads = torch.cat([W1.grad.detach().flatten(), W2.grad.detach().flatten()])
            grad_var = torch.var(all_grads).item()

            optimizer.step()

            train_losses.append(loss.item())
            grad_variances.append(grad_var)

            if (step + 1) % 200 == 0 or step == num_steps - 1:
                print(f"Step {step+1:4d}/{num_steps} | Loss: {loss.item():.6f} | Grad Var: {grad_var:.6f}")

        # Post-training true discrete evaluation
        g1 = W1.abs().mean().clamp(min=1e-4)
        g2 = W2.abs().mean().clamp(min=1e-4)
        dq1 = hard_ternary(W1 / g1)
        dq2 = hard_ternary(W2 / g2)

        discrete_pred = torch.tanh(X @ dq1.t()) @ dq2.t()
        discrete_test_loss = nn.functional.mse_loss(discrete_pred, Y).item()

        # Weight recovery accuracy vs ground truth discrete topology
        match1 = (dq1 == W1_true).float().mean().item()
        match2 = (dq2 == W2_true).float().mean().item()
        avg_recovery = 0.5 * (match1 + match2)

        # Shockwave loss spikes (> 2 sigma)
        losses_arr = np.array(train_losses)
        window = 30
        spikes = 0
        for i in range(window, len(losses_arr)):
            mean_w = np.mean(losses_arr[i-window:i])
            std_w = np.std(losses_arr[i-window:i])
            if losses_arr[i] > mean_w + 2.0 * std_w:
                spikes += 1

        all_weights = torch.cat([(W1 / g1).detach().flatten(), (W2 / g2).detach().flatten()]).cpu().numpy()

        results[method] = {
            "train_losses": losses_arr,
            "grad_variances": np.array(grad_variances),
            "discrete_test_loss": discrete_test_loss,
            "weight_recovery": avg_recovery,
            "spikes": spikes,
            "weights": all_weights
        }

    # Summary Table
    print("\n" + "=" * 80)
    print(f"{'Method':<16} | {'Final Train Loss':>16} | {'Discrete Test Loss':>20} | {'Weight Recovery':>16}")
    print("-" * 80)
    for m in methods:
        print(f"{m.upper():<16} | {results[m]['train_losses'][-1]:16.6f} | {results[m]['discrete_test_loss']:20.6f} | {results[m]['weight_recovery']*100:15.2f}%")
    print("=" * 80)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Training loss curves
    ax = axes[0, 0]
    ax.plot(results['ste']['train_losses'], label='STE Baseline', alpha=0.7, color='blue')
    ax.plot(results['temp_anneal']['train_losses'], label='Temp Annealing', alpha=0.7, color='orange')
    ax.plot(results['mige']['train_losses'], label='MIGE (Sobolev)', alpha=0.9, color='green', linewidth=1.8)
    ax.set_yscale('log')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('MSE Loss (log scale)')
    ax.set_title('Training Loss Dynamics')
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend()

    # 2. Discrete Post-Training Test Loss (The true test)
    ax = axes[0, 1]
    labels = ['STE', 'Temp Anneal', 'MIGE']
    disc_losses = [results['ste']['discrete_test_loss'], results['temp_anneal']['discrete_test_loss'], results['mige']['discrete_test_loss']]
    bars = ax.bar(labels, disc_losses, color=['blue', 'orange', 'green'], alpha=0.8)
    ax.set_ylabel('Discrete MSE Loss (Hard {-1, 0, 1})')
    ax.set_title('True Post-Training Discrete Inference Loss')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval:.3f}', ha='center', va='bottom', fontweight='bold')
    ax.grid(True, axis='y', ls='--', alpha=0.5)

    # 3. Discrete Weight Recovery Accuracy
    ax = axes[1, 0]
    rec_accs = [results['ste']['weight_recovery']*100, results['temp_anneal']['weight_recovery']*100, results['mige']['weight_recovery']*100]
    bars = ax.bar(labels, rec_accs, color=['blue', 'orange', 'green'], alpha=0.8)
    ax.set_ylabel('Discrete Recovery Accuracy (%)')
    ax.set_title('Ternary Weight Topology Recovery')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1.0, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax.grid(True, axis='y', ls='--', alpha=0.5)

    # 4. Latent Weight Distribution
    ax = axes[1, 1]
    ax.hist(results['ste']['weights'], bins=50, alpha=0.4, label='STE', density=True, color='blue')
    ax.hist(results['temp_anneal']['weights'], bins=50, alpha=0.4, label='Temp Anneal', density=True, color='orange')
    ax.hist(results['mige']['weights'], bins=50, alpha=0.6, label='MIGE', density=True, color='green')
    ax.axvline(1.0, color='black', linestyle=':', label=r'Ternary Targets $\{-1, 0, 1\}$')
    ax.axvline(0.0, color='black', linestyle=':')
    ax.axvline(-1.0, color='black', linestyle=':')
    ax.set_xlabel(r'Normalized Latent Weight $w / \gamma$')
    ax.set_ylabel('Density')
    ax.set_title(r'Weight Clustering Around $\{-1, 0, +1\}$')
    ax.legend()
    ax.grid(True, ls='--', alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "obj5_training_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved Objective 5 plot to: {plot_path}\n")

    return results

if __name__ == "__main__":
    run_objective_5()
