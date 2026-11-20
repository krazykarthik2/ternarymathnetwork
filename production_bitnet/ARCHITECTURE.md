# Production BitNet b1.58: The Industry Standard (Microsoft Research Architecture)

## 1. Why Heuristics (Hopfield Inversion, MIGE, TA) Fall Short of Production Grade
In commercial production (Microsoft BitNet b1.58, Llama-3, Qwen-2.5):
1. **Hopfield / Inversion**: Cannot resolve deep hierarchical compositional semantics across 12-24 sequential layers.
2. **Temperature Annealing ($Q_T$)**: Causes severe discretization shock when soft weights are snapped to hardware ternary after training.
3. **MIGE / Sobolev Perturbations**: Introduce surrogate gradient mismatch and boundary limit cycles.

---

## 2. The Official Microsoft BitNet b1.58 Formulation (Wang et al., 2024)

### 2.1 Weight Quantization (Absmean Scaling)
Weights $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ are normalized by their average absolute magnitude $\gamma$:
$$\gamma = \frac{1}{mn} \sum_{i=1}^m \sum_{j=1}^n |W_{ij}|$$
$$W_{\text{norm}} = \frac{W}{\gamma + \epsilon}$$
$$W_q = \operatorname{Clamp}\left(\operatorname{Round}(W_{\text{norm}}), -1, +1\right) \in \{-1, 0, +1\}$$

### 2.2 Activation Quantization (Absmax 8-bit Integer Quantization)
Activations $X$ are scaled to $[-128, 127]$ per token:
$$\beta = \max_{j} |X_{ij}|$$
$$\tilde{X} = \operatorname{Clamp}\left(\operatorname{Round}\left(X \times \frac{127}{\beta + \epsilon}\right), -128, 127\right)$$

### 2.3 Forward & Backward Pass (True STE)
- **Forward**:
  $$Y = \left(\tilde{X} \cdot W_q^T\right) \times \frac{\beta \cdot \gamma}{127}$$
  *(On hardware, $\tilde{X} \cdot W_q^T$ is purely integer INT8 $\times$ INT2 additions/subtractions with zero floating-point multiplications!)*
- **Backward**:
  $$\frac{\partial \mathcal{L}}{\partial W} \approx \frac{\partial \mathcal{L}}{\partial W_q} \cdot \mathbf{1}_{|W_{\text{norm}}| \le 1.0}$$
  The Straight-Through Estimator (STE) operates on the discrete forward pass directly. There is **zero discretization shock** because the model trains on the exact discrete lattice from Step 1 to the end.

---

## 3. Production Training Recipe
- **Architecture**: 125M LLaMA-style decoder (12 layers, 12 heads, $d=768$, hidden_dim=2048, SwiGLU, RMSNorm).
- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.95, \text{weight\_decay} = 0.01$).
- **Learning Rate**: $1.5 \times 10^{-3}$ with 500-step linear warmup followed by Cosine Decay down to $1.5 \times 10^{-4}$.
- **Attention**: FlashAttention-2 (`F.scaled_dot_product_attention`).
- **Precision**: PyTorch `bfloat16` AMP with TF32 enabled.
