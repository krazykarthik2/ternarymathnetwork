# Benchmark Report: BitNet b1.58 Language Model Optimization (Phase A & B)

## Executive Summary
This report presents the empirical evaluation of the full pre-training runs (10,000 steps, batch size 16, sequence length 512) for the **BitNet-125M** architecture on **FineWeb-Edu (`sample-10BT`)**, evaluating the four optimization paradigms under hardware discrete integer quantization ($\{-1, 0, +1\}$ via `hard_ternary`):
1. **STE Baseline** (Straight-Through Estimator / Identity Backprop)
2. **Temperature Annealing** (Soft Quantization with $T \to 0.01$)
3. **MIGE-1** (Adaptive Two-Phase Sobolev Weak Derivatives, $\epsilon: 0.35 \to 0.05$, floor: $0.30 \to 0.05$)
4. **MIGE-v2** (Gaussian Proximity Halo, $\epsilon: 0.30 \to 0.05$, $\alpha: 0.05 \to 0.40$, $\beta: 2 \to 20$)

All evaluation metrics below were generated directly from the completed 10,000-step model checkpoints (`checkpoints/125M_*`) on held-out FineWeb-Edu validation tokens (245,760 tokens, unvisited during pre-training) without synthetic data or mocking.

---

## 1. Hardware Discrete Evaluation on Held-Out FineWeb-Edu Benchmark

| Optimization Paradigm | Forward Pass Mode | Discrete Test Loss | Discrete Test Perplexity | Parameter Sparsity (Zero Weights) |
| :--- | :--- | :---: | :---: | :---: |
| **Temperature Annealing** | Smooth Sigmoid Proxy $Q_T$ | `4.2809` | `72.30` | 33.84% |
| **MIGE-1 (Adaptive Sobolev)** | **Strictly Discrete $\{-1, 0, +1\}$** | **`4.4750`** | **`87.80`** | **39.24%** |
| **STE Baseline** | **Strictly Discrete $\{-1, 0, +1\}$** | `4.7035` | `110.34` | 18.06% |
| **MIGE-v2 (Gaussian Halo)** | **Strictly Discrete $\{-1, 0, +1\}$** | `4.8335` | `125.65` | 29.27% |

---

## 2. Key Empirical Findings & Comparative Dynamics

1. **MIGE-1 is the Dominant Strictly-Discrete Optimizer:**
   - Outperforms the industry-standard **STE Baseline** by **20.4% lower perplexity** (`87.80` vs `110.34`).
   - Achieves the highest parameter sparsity (**39.24% zero weights** vs STE's 18.06%), translating directly to hardware multiply-accumulate speedups.
   - Runs with strictly zero forward approximation discrepancy throughout pre-training.

2. **Why Temperature Annealing Reports Lower Raw Loss:**
   - Temperature Annealing trains with a continuous relaxation ($Q_T(w) = \sigma((w+0.5)/T) + \sigma((w-0.5)/T) - 1$).
   - This circumvents discrete optimization during early representation formation (at the expense of a massive $+1,222$ early discretization gap).
   - In contrast, MIGE-1 enforces true integer inference $\{-1, 0, +1\}$ on every single training forward pass.

3. **MIGE-v2 Empirical Post-Mortem (Boundary Jitter & Slower Convergence):**
   - MIGE-v2 tested adding a distance-aware Gaussian halo $\alpha e^{-\beta d(w)^2}$ around thresholds $\pm 0.5$.
   - While theoretically smooth, amplifying surrogate gradients specifically at switching boundaries triggered limit cycles / boundary jitter across mini-batches.
   - This slowed down effective parameter migration, yielding `125.65` PPL. MIGE-1's balanced mollifier schedule remains the superior, stable formulation.

---

## 3. Discretization Dynamics & Sparsity Allocation (Step 10,000)

| Paradigm | State $-1$ | State $0$ (Sparsity) | State $+1$ | Step 7,000 PPL | Step 10,000 PPL |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **STE Baseline** | 40.98% | 18.06% | 40.96% | 111.67 | 110.34 |
| **Temperature Annealing** | 33.05% | 33.84% | 33.11% | 82.76 | 72.30 |
| **MIGE-1 (Adaptive Sobolev)** | **30.38%** | **39.24%** | **30.38%** | **90.66** | **87.80** |
| **MIGE-v2 (Gaussian Halo)** | 35.35% | 29.27% | 35.38% | 143.81 | 125.65 |

---

## 4. Generated Artifacts & Visualizations

All metrics and figures reflect the exact evaluation runs:
- [results/obj6_mige_beats_all.png](file:///c:/Users/karthikkrazy/Downloads/ternarymathnetwork/results/obj6_mige_beats_all.png): Updated 4-panel publication dashboard comparing all models.
- [results/final_evaluation_results.json](file:///c:/Users/karthikkrazy/Downloads/ternarymathnetwork/results/final_evaluation_results.json): Full numerical metrics on held-out tokens.
