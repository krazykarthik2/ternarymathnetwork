"""
Objective 6: BitNet b1.58 Language Model Pre-Training Benchmark
Directly implements Path B:
Trains BitNet b1.58 autoregressive Transformer models comparing:
1. Control A: Standard STE (Straight-Through Estimator)
2. Control B: Temperature Annealing (Soft quantizer annealed towards 0)
3. Experiment: MIGE (Mollified Integral Gradient Estimator via Sobolev Weak Derivatives)

Evaluates:
- Training Cross-Entropy Loss & Perplexity Trajectory
- True Discrete Post-Training Inference Loss & Perplexity (Weights snapped to {-1, 0, +1})
- Quantization Shockwaves (Instability Spikes > 2*std)
- Gradient Variance Dynamics across BitLinear layers
- Fast execution on local CPU/VM, scales dynamically to 125M on CUDA VM.
"""
import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from device_utils import get_device, get_device_info
from bitnet_model import BitNet125M
from dataset_utils import get_dataloaders
from quantizers import hard_ternary

def run_objective_6(results_dir: str = "results", num_steps: int = 400,
                    dim: int = 192, n_layers: int = 4, n_heads: int = 6, hidden_dim: int = 384,
                    seq_len: int = 64, batch_size: int = 32):
    os.makedirs(results_dir, exist_ok=True)
    device = get_device()
    print("=" * 80)
    print("OBJECTIVE 6: BITNET b1.58 LANGUAGE MODEL PRE-TRAINING EXPERIMENT")
    print(f"Hardware Environment: {get_device_info()['name']}")
    print("=" * 80)

    train_loader, val_loader, tokenizer = get_dataloaders(seq_len=seq_len, batch_size=batch_size)
    vocab_size = tokenizer.vocab_size

    methods = ["ste", "temp_anneal", "mige"]
    results = {}

    for method in methods:
        print(f"\n---> Initializing and Pre-Training BitNet b1.58 [{method.upper()}]")
        torch.manual_seed(42)
        model = BitNet125M(vocab_size=vocab_size, dim=dim, n_layers=n_layers,
                           n_heads=n_heads, hidden_dim=hidden_dim, max_seq_len=seq_len,
                           method=method).to(device)

        total_params = model.count_parameters()
        print(f"Total BitNet Parameters: {total_params:,} | Quantized Method: {method.upper()}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        train_iter = iter(train_loader)
        losses = []
        perplexities = []
        grad_variances = []

        start_time = time.time()
        for step in range(num_steps):
            try:
                x_batch, y_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x_batch, y_batch = next(train_iter)

            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Cosine schedules
            T_val = 0.02 + 0.5 * (1.0 - 0.02) * (1.0 + math.cos(math.pi * step / num_steps))
            eps_val = 0.05 + 0.5 * (0.35 - 0.05) * (1.0 + math.cos(math.pi * step / num_steps))

            optimizer.zero_grad()
            logits = model(x_batch, T=T_val, epsilon=eps_val)
            loss = criterion(logits.view(-1, vocab_size), y_batch.view(-1))
            loss.backward()

            # Record gradient variance across all BitLinear weight tensors
            grad_list = []
            for p in model.parameters():
                if p.grad is not None:
                    grad_list.append(p.grad.detach().flatten())
            all_grads = torch.cat(grad_list)
            grad_var = torch.var(all_grads).item()

            # Gradient clipping to maintain training stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_val = loss.item()
            ppl_val = math.exp(min(loss_val, 20.0))

            losses.append(loss_val)
            perplexities.append(ppl_val)
            grad_variances.append(grad_var)

            if (step + 1) % 100 == 0 or step == num_steps - 1:
                elapsed = time.time() - start_time
                print(f"Step {step+1:4d}/{num_steps} | Loss: {loss_val:.4f} | Perplexity: {ppl_val:8.2f} | Grad Var: {grad_var:.6f} | Elapsed: {elapsed:.1f}s")

        # Evaluate True Post-Training Discrete Inference (Hardware ternary test)
        # Snap every BitLinear layer to discrete_eval: W_q = hard_ternary(W / gamma)
        print(f"Evaluating True Discrete Hardware Inference for [{method.upper()}]...")
        for module in model.modules():
            if hasattr(module, 'method'):
                module.method = "discrete_eval"

        model.eval()
        val_loss_sum = 0.0
        val_tokens = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_logits = model(vx)
                v_loss = criterion(v_logits.view(-1, vocab_size), vy.view(-1))
                val_loss_sum += v_loss.item() * vy.numel()
                val_tokens += vy.numel()

        discrete_test_loss = val_loss_sum / val_tokens
        discrete_test_ppl = math.exp(min(discrete_test_loss, 20.0))
        print(f"[{method.upper()}] Discrete Test Loss: {discrete_test_loss:.4f} | Discrete Test Perplexity: {discrete_test_ppl:.2f}")

        # Compute Quantization Shockwave Spikes (> 2 std)
        losses_arr = np.array(losses)
        window = 25
        spikes = 0
        for i in range(window, len(losses_arr)):
            mean_w = np.mean(losses_arr[i-window:i])
            std_w = np.std(losses_arr[i-window:i])
            if losses_arr[i] > mean_w + 2.0 * std_w:
                spikes += 1

        results[method] = {
            "losses": losses_arr,
            "perplexities": np.array(perplexities),
            "grad_variances": np.array(grad_variances),
            "discrete_test_loss": discrete_test_loss,
            "discrete_test_ppl": discrete_test_ppl,
            "spikes": spikes,
            "final_train_loss": losses_arr[-1]
        }

    # Summary Benchmark Table
    print("\n" + "=" * 90)
    print(f"{'Method':<16} | {'Final Train Loss':>16} | {'Discrete Test Loss':>20} | {'Discrete Perplexity':>20} | {'Spikes':>8}")
    print("-" * 90)
    for m in methods:
        print(f"{m.upper():<16} | {results[m]['final_train_loss']:16.4f} | {results[m]['discrete_test_loss']:20.4f} | {results[m]['discrete_test_ppl']:20.2f} | {results[m]['spikes']:8d}")
    print("=" * 90)

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Training Perplexity Curves
    ax = axes[0, 0]
    ax.plot(results['ste']['perplexities'], label='BitNet (STE Baseline)', alpha=0.7, color='blue')
    ax.plot(results['temp_anneal']['perplexities'], label='BitNet (Temp Annealing)', alpha=0.7, color='orange')
    ax.plot(results['mige']['perplexities'], label='BitNet (MIGE Sobolev)', alpha=0.9, color='green', linewidth=1.8)
    ax.set_yscale('log')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Perplexity (log scale)')
    ax.set_title('BitNet b1.58 Language Modeling Perplexity Trajectory')
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend()

    # 2. Discrete Post-Training Test Perplexity (Hardware Inference)
    ax = axes[0, 1]
    labels = ['STE', 'Temp Anneal', 'MIGE']
    ppls = [results['ste']['discrete_test_ppl'], results['temp_anneal']['discrete_test_ppl'], results['mige']['discrete_test_ppl']]
    bars = ax.bar(labels, ppls, color=['blue', 'orange', 'green'], alpha=0.8)
    ax.set_ylabel('Hardware Discrete Perplexity (Lower is Better)')
    ax.set_title('Hardware Discrete Test Perplexity (snapped to {-1,0,1})')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')
    ax.grid(True, axis='y', ls='--', alpha=0.5)

    # 3. Gradient Variance Stability
    ax = axes[1, 0]
    ax.plot(results['ste']['grad_variances'], label='STE Grad Variance', alpha=0.6, color='blue')
    ax.plot(results['temp_anneal']['grad_variances'], label='Temp Anneal Grad Variance', alpha=0.6, color='orange')
    ax.plot(results['mige']['grad_variances'], label='MIGE Grad Variance', alpha=0.9, color='green')
    ax.set_yscale('log')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Gradient Variance (log scale)')
    ax.set_title('BitLinear Layer Gradient Telemetry')
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend()

    # 4. Quantization Shockwave Spikes
    ax = axes[1, 1]
    spikes_list = [results['ste']['spikes'], results['temp_anneal']['spikes'], results['mige']['spikes']]
    bars = ax.bar(labels, spikes_list, color=['blue', 'orange', 'green'], alpha=0.8)
    ax.set_ylabel('Number of Shockwave Spikes (> 2σ)')
    ax.set_title('Quantization Shockwaves (Instability Count)')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{int(yval)}', ha='center', va='bottom', fontweight='bold')
    ax.grid(True, axis='y', ls='--', alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "obj6_bitnet_language_model.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved Objective 6 plot to: {plot_path}\n")

    return results

if __name__ == "__main__":
    run_objective_6()
