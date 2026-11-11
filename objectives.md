# Experimental Objectives: Proving the Staircase Paradox in Quantized Networks

> [!IMPORTANT]
> Each objective is a self-contained experiment that produces a **numerical result or plot** proving one claim from the paper. They build on each other — do them in order.

---

## Objective 1: Prove the Staircase Paradox (Pure Math)

**What to show:** As staircase step count $n \to \infty$, the function converges but its arc length does NOT.

**Setup:**
- Define staircase $S_n(x) = \lfloor nx \rfloor / n$ on $[0, 1]$ for $n = 1, 2, 4, 8, 16, \ldots, 4096$
- Define diagonal $D(x) = x$

**Compute:**
1. $L_\infty$ error: $\max_x |S_n(x) - x|$ → should go to 0 as $1/n$
2. Arc length of $S_n$ (sum of horizontal + vertical segments) → should stay at **exactly 2.0** for all $n$
3. Arc length of diagonal = $\sqrt{2} \approx 1.414$

**Output:** A table with columns `[n, L_inf_error, arc_length]` and a plot showing $L_\infty \to 0$ while arc length is a flat line at 2.0. Print the ratio `arc_length / sqrt(2)` = 1.414 for all $n$.

---

## Objective 2: Prove Temperature Annealing Gradients Explode (The Nascent Delta)

**What to show:** The derivative of the soft ternary quantizer converges to Dirac deltas, not a usable function.

**Setup:**
- Define soft quantizer: $Q_T(w) = \sigma\!\left(\frac{w + 0.5}{T}\right) + \sigma\!\left(\frac{w - 0.5}{T}\right) - 1$
- Compute its derivative $Q_T'(w)$ analytically (sum of two scaled sech² bumps)
- Sweep $T = 1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001$

**Compute:**
1. **Peak gradient** at threshold $w = 0.5$: should be $\frac{1}{4T} \to \infty$
2. **$L_2$ energy** $\int |Q_T'(w)|^2 \, dw$: should scale as $\frac{1}{12T} \to \infty$
3. **$L_1$ integral** $\int Q_T'(w) \, dw$: should stay at exactly **2.0** for all $T$ (total jump is preserved)
4. **Gradient at $w = 0.8$** (away from threshold): should go to **0**

**Output:** Table with columns `[T, peak_grad, L2_energy, L1_integral, grad_at_0.8]`. Plot overlay of $Q_T'(w)$ for all $T$ values — showing the bumps getting taller and narrower, becoming delta spikes.

---

## Objective 3: Prove STE Shadow Error is Permanent

**What to show:** STE gradient points in a direction unrelated to the true quantized loss gradient.

**Setup:**
- Simple 1-layer linear network: $y = Q(w) \cdot x$, loss $= (y - \text{target})^2$
- Single scalar weight $w$, single input $x = 1.0$, target $= 0.7$
- Note: $Q(w)$ maps to $\{-1, 0, +1\}$, so the best achievable output is $1.0$ (closest to 0.7)

**Compute for $w$ swept from $-1.5$ to $+1.5$:**
1. **True quantized loss** $\mathcal{L}(Q(w))$: piecewise constant (flat within each bin)
2. **STE loss** $\mathcal{L}(w)$: smooth parabola $(w - 0.7)^2$
3. **STE gradient** at each $w$: $2(w - 0.7)$ — always nonzero, always "moving" $w$
4. **True gradient**: 0 everywhere (piecewise constant has no slope)

**The proof:** For any $w \in (0.5, 1.5)$, the quantized output is $Q(w) = 1$, the true loss is constant $(1 - 0.7)^2 = 0.09$, but STE reports a gradient of $2(w - 0.7)$ which varies across the bin — **the STE is optimizing a phantom objective**.

**Output:** Plot with two y-axes: true quantized loss (staircase) vs STE surrogate loss (parabola), overlaid with STE gradient arrows. Highlight the "shadow error" = $|w - Q(w)|$.

---

## Objective 4: Prove MIGE Produces Bounded Smooth Gradients

**What to show:** The mollified gradient is finite, smooth, and in $L^2$ for all bandwidths $\epsilon > 0$.

**Setup:**
- Define the bump mollifier: $\eta(x) = C \exp(-1/(1-x^2))$ for $|x| < 1$, else 0
- MIGE gradient: $\nabla^{\text{MIGE}}_\epsilon Q(w) = \frac{1}{\epsilon}[\eta(\frac{w+0.5}{\epsilon}) + \eta(\frac{w-0.5}{\epsilon})]$
- Sweep $\epsilon = 0.3, 0.2, 0.1, 0.05, 0.02$

**Compute:**
1. **Peak gradient** at $w = 0.5$: should be $C / \epsilon$ — large but **finite** for all $\epsilon > 0$
2. **$L_2$ energy** $\int |\nabla^{\text{MIGE}}|^2 \, dw$: finite for all $\epsilon$, scales as $O(1/\epsilon)$
3. **Gradient at $w = 0.8$** (outside support when $\epsilon < 0.3$): exactly **0**
4. **Smoothness**: the gradient curve has no kinks, no discontinuities

**Output:** Plot overlay of MIGE gradient for all $\epsilon$ values. Side-by-side comparison table:

| Metric | Temp. Anneal ($T=0.01$) | MIGE ($\epsilon=0.01$) |
|---|---|---|
| Peak gradient | 25.0 | ~225 (finite) |
| $L_2$ energy | $\infty$ (diverges) | finite |
| Gradient away from threshold | ~0 | exactly 0 |
| Gradient is $C^\infty$? | Yes | Yes |
| Forward pass exact? | No | **Yes** |

---

## Objective 5: Train a Small Network — Compare All Three Methods

**What to show:** MIGE converges faster and more stably than STE and temperature annealing on an actual learning task.

**Setup:**
- **Task:** Learn a small function approximation (e.g., 2-layer MLP, 128 hidden units, ternary weights, fitting a sine wave or MNIST)
- **Three training methods on the same architecture:**
  - **Method A (STE):** Forward = exact $Q(w)$, backward = identity (clipped)
  - **Method B (Temp Anneal):** Forward = $Q_T(w)$, backward = $Q_T'(w)$, anneal $T$ from 1.0 → 0.01 over training
  - **Method C (MIGE):** Forward = exact $Q(w)$, backward = MIGE gradient with adaptive $\epsilon$

**Track per training step:**
1. **Training loss** — MIGE should be lower / converge faster
2. **Gradient variance** across all weights — STE = high constant, Temp Anneal = increasing over time, MIGE = decreasing
3. **Loss spike count** (steps where loss jumps > 2σ from 50-step moving average) — Temp Anneal should have the most, MIGE should have the fewest
4. **Final weight distribution** — histogram of latent weights. MIGE should cluster tightest around $\{-1, 0, +1\}$

**Output:**
- 3 training curves (loss vs step) on one plot
- 3 gradient variance curves on one plot
- Bar chart of spike counts
- 3 weight histograms side by side

---

## File Structure

```
ternarymathnetwork/
├── obj1_staircase_paradox.py      # Objective 1
├── obj2_nascent_delta.py          # Objective 2
├── obj3_ste_shadow_error.py       # Objective 3
├── obj4_mige_gradient.py          # Objective 4
├── obj5_training_comparison.py    # Objective 5
├── mige.py                        # Shared: MIGE autograd function + ManhattanSGD optimizer
├── quantizers.py                  # Shared: Q(w), Q_T(w), STE autograd
├── run_all.py                     # Runs all 5 objectives, saves plots to results/
└── results/                       # Output plots and tables
```

---

## Success Criteria

| Objective | It's proven if... |
|---|---|
| 1 | Arc length = 2.0 for ALL $n$, while $L_\infty \to 0$ |
| 2 | $L_2$ energy diverges as $1/T$, peak diverges as $1/4T$, but $L_1$ integral stays at 2.0 |
| 3 | STE gradient is nonzero inside bins where the true loss is flat |
| 4 | MIGE $L_2$ energy is finite for all $\epsilon > 0$ (while Temp Anneal's is not at same scale) |
| 5 | MIGE has lower final loss, fewer spikes, and tighter weight clustering than STE and Temp Anneal |
