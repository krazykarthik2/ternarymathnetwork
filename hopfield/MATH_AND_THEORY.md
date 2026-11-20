# Mathematical Foundations of Ternary Associative Networks (Hopfield-BitNet b1.58)

## 1. Classical vs. Modern Hopfield Networks: The Attention Bridge

### 1.1 Classical Discrete Hopfield Networks
A classical Hopfield network consists of $N$ recurrently connected binary or spin units $\mathbf{s} \in \{-1, +1\}^N$. 
Given $P$ target memory vectors $\{\boldsymbol{\xi}^\mu\}_{\mu=1}^P \in \{-1, +1\}^N$, the standard Hebbian learning prescription defines the weight matrix $W \in \mathbb{R}^{N \times N}$ in closed form (zero iterations):

$$W = \frac{1}{N} \sum_{\mu=1}^P \boldsymbol{\xi}^\mu (\boldsymbol{\xi}^\mu)^T - \frac{P}{N} I$$

The discrete dynamical update for unit $i$ is:
$$s_i^{(t+1)} = \operatorname{sgn}\left(\sum_{j=1}^N W_{ij} s_j^{(t)}\right)$$

Under symmetric weights $W = W^T$ and zero diagonal $W_{ii} = 0$, the network is guaranteed to monotonically minimize the Lyapunov energy function:
$$E(\mathbf{s}) = -\frac{1}{2} \mathbf{s}^T W \mathbf{s} = -\frac{1}{2} \sum_{i,j} W_{ij} s_i s_j$$

**Capacity Limit:** Under Hebbian storage, the maximum number of error-free recoverable patterns is bounded by:
$$P_{\max} \approx \frac{N}{2 \ln N} \approx 0.138 N$$
Exceeding this creates dense spurious attractors, driving the memory recall to noise.

---

### 1.2 Modern Continuous Hopfield Networks (Ramsauer et al., 2020)
Modern continuous Hopfield networks replace the quadratic energy with an exponential log-sum-exp (LSE) potential over continuous memory state $\mathbf{x} \in \mathbb{R}^d$ and stored patterns $X = [\mathbf{x}_1, \dots, \mathbf{x}_P] \in \mathbb{R}^{d \times P}$:

$$E(\mathbf{x}) = -\frac{1}{\beta} \ln \left( \sum_{\mu=1}^P \exp(\beta \mathbf{x}^T \mathbf{x}_\mu) \right) + \frac{1}{2} \|\mathbf{x}\|^2$$

By Concave-Convex Procedure (CCCP), the discrete update rule that guarantees monotonic descent in $E(\mathbf{x})$ is:
$$\mathbf{x}^{(t+1)} = X \operatorname{softmax}\left(\beta X^T \mathbf{x}^{(t)}\right)$$

**Direct Equivalence to Transformer Attention:**
Let queries be $Q = \mathbf{x}^{(t)}$, keys be $K = X$, and values be $V = X$. Then:
$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V \equiv \mathbf{x}^{(t+1)}$$
Modern Hopfield networks expand storage capacity exponentially: $P_{\max} \approx 2^{d/2}$.

---

## 2. Ternary Hopfield Formulation: State Space $\{-1, 0, +1\}$

In language modeling and ternary neural networks (such as BitNet b1.58), sparsity and zero-states are crucial. We generalize the state space to ternary spins $\mathbf{s} \in \mathcal{T}^N$, where $\mathcal{T} = \{-1, 0, +1\}$.

### 2.1 Ternary Hebbian Association
Let input patterns be $X \in \mathcal{T}^{P \times d_{\text{in}}}$ and target associative representations be $Y \in \mathcal{T}^{P \times d_{\text{out}}}$.
The unquantized correlation matrix $M$ is computed in a single pass (single-shot):
$$M = \frac{1}{P} Y^T X \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$$

To project $M$ into exact ternary weights $W \in \{-1, 0, +1\}^{d_{\text{out}} \times d_{\text{in}}}$, we define the optimal thresholding operator:
$$\Delta = \alpha \cdot \frac{1}{d_{\text{out}} d_{\text{in}}} \sum_{i,j} |M_{ij}|, \quad \alpha \in [0.5, 0.7]$$

$$W_{ij} = \mathcal{Q}_{\text{ternary}}(M_{ij}, \Delta) = \begin{cases} 
+1 & \text{if } M_{ij} > \Delta \\
0 & \text{if } |M_{ij}| \le \Delta \\
-1 & \text{if } M_{ij} < -\Delta
\end{cases}$$

### 2.2 Energy Function of Ternary Hopfield Layer
For a ternary Hopfield layer with state $\mathbf{s} \in \{-1, 0, +1\}^N$, threshold vector $\boldsymbol{\theta} \in \mathbb{R}^N$, and symmetric coupling $W$:
$$E(\mathbf{s}) = -\frac{1}{2} \mathbf{s}^T W \mathbf{s} + \boldsymbol{\theta}^T |\mathbf{s}|$$

The term $\boldsymbol{\theta}^T |\mathbf{s}|$ acts as an $L_0 / L_1$ regularization penalizing non-zero activations, inducing controllable sparsity:
$$s_i^{(t+1)} = \begin{cases}
+1 & \text{if } \sum_j W_{ij} s_j > \theta_i \\
0 & \text{if } |\sum_j W_{ij} s_j| \le \theta_i \\
-1 & \text{if } \sum_j W_{ij} s_j < -\theta_i
\end{cases}$$

---

## 3. Single-Shot Calculated Weights vs. Continuous Training

### 3.1 Closed-Form Associative Weight Calculation
Rather than starting from random Kaiming/Gaussian noise and optimizing via gradient descent, the Hopfield-BitLinear layer calculates its ternary weights directly from data statistics in **one shot**:
1. Pass representative tokens or context vectors $\{x_k\}_{k=1}^K$ through the embedding.
2. Compute empirical cross-covariance:
   $$C_{YX} = \frac{1}{K} \sum_{k=1}^K y_k x_k^T$$
3. Perform pseudo-inverse projection or Storkey associative learning:
   $$W_{\text{analytical}} = C_{YX} \left(C_{XX} + \lambda I\right)^{-1}$$
4. Quantize $W_{\text{analytical}}$ directly into ternary BitNet representation:
   $$W_{\text{ternary}} = \operatorname{RoundClip}\left(\frac{W_{\text{analytical}}}{\gamma}\right) \in \{-1, 0, +1\}$$
   where $\gamma = \frac{1}{mn} \sum |W_{ij}|$.

### 3.2 Gradient Flow and MIGE / STE Hybrid Optimization
When fine-tuning on downstream cross-entropy loss $\mathcal{L}_{\text{CE}}$, discrete weights cannot propagate gradients directly because $\frac{\partial \mathcal{Q}}{\partial W} = 0$ almost everywhere.
We integrate **MIGE (Mollified Identity Gradient Estimator)** and **STE**:
$$\frac{\partial \mathcal{L}}{\partial W_{\text{latent}}} \approx \frac{\partial \mathcal{L}}{\partial W_{\text{ternary}}} \cdot \psi_T(W_{\text{norm}})$$
where $\psi_T$ is the mollified gradient kernel maintaining stable parameter updates across discrete boundaries.

---

## 4. 125M Architecture Specification

Following the BitNet b1.58 standards in this repository:
- **Parameters**: ~125 Million
- **Layers**: 12 Decoder Blocks
- **Hidden Dimension ($d_{\text{model}}$)**: 768
- **Attention Heads**: 12 ($d_{\text{head}} = 64$)
- **FFN Intermediate Dimension**: 2048 (SwiGLU structure)
- **Quantization**: INT8 activations $\times$ INT2/Ternary $\{-1, 0, 1\}$ weights
- **Weight Acquisition**:
  1. Analytical Closed-Form Hopfield Associative Calculation (zero-shot).
  2. Optional BitNet fine-tuning via MIGE / STE on causal language modeling.
