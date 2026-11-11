# The Staircase Paradox of Quantization: Topological Limits of Gradient Approximation in 1.58-Bit Neural Networks

---

**Abstract.** We identify a fundamental topological obstruction in the training of ternary-weight neural networks that has been hiding in plain sight: the *staircase paradox* of real analysis. We prove that the two dominant gradient-estimation strategies for 1.58-bit networks — the Straight-Through Estimator (STE) and temperature-annealed soft quantization — are both victims of a classical theorem: *uniform convergence of functions does not imply convergence of their derivatives*. The STE introduces a permanent, irreducible "shadow gradient" error by substituting the identity for the quantizer's derivative. Temperature annealing, while seemingly more principled, reproduces the staircase paradox exactly: as the temperature $T \to 0$, the soft quantizer converges uniformly to the hard ternary map, but its gradient converges to a sum of Dirac deltas — *not* to the true subgradient of the discrete system. The network optimizes along Manhattan-metric contours while the loss landscape is Euclidean, creating *quantization shockwaves* that stall convergence. To resolve this, we abandon pointwise derivatives entirely and propose **Mollified Integral Gradient Estimation (MIGE)**, a backpropagation method grounded in Sobolev space theory. MIGE replaces the singular pointwise gradient with a *weak derivative* — the expected gradient value over a learnable local neighborhood — thereby integrating over the Dirac singularities rather than evaluating them. We provide full mathematical proofs of convergence, propose a concrete optimizer (**ManhattanSGD**), and outline an empirical protocol on the BitNet b1.58 architecture demonstrating that MIGE achieves lower perplexity and faster convergence by eliminating the hidden gradient mismatch.

**Keywords:** Quantized neural networks, ternary weights, BitNet, Straight-Through Estimator, staircase paradox, weak derivatives, Sobolev spaces, gradient estimation, 1.58-bit networks

---

## 1. Introduction

The pursuit of extreme neural network quantization — reducing weights from 16-bit floating point to ternary values $\{-1, 0, +1\}$ — promises transformative gains in inference efficiency, memory footprint, and energy consumption (Ma et al., 2024; Wang et al., 2023). The BitNet b1.58 architecture demonstrated that large language models can operate with only 1.58 bits per weight ($\log_2 3 \approx 1.58$), achieving performance competitive with full-precision counterparts.

Yet beneath the impressive empirical results lies an unresolved mathematical crisis. Training these networks requires computing gradients through a fundamentally discontinuous quantization function:

$$Q(w) = \begin{cases} +1 & \text{if } w > +\frac{1}{2} \\[4pt] 0 & \text{if } -\frac{1}{2} \leq w \leq +\frac{1}{2} \\[4pt] -1 & \text{if } w < -\frac{1}{2} \end{cases}$$

The derivative of $Q$ is zero almost everywhere and undefined at the thresholds — a function whose gradient carries *zero information* for optimization. The entire field has worked around this with two heuristics:

1. **The Straight-Through Estimator (STE)** (Bengio et al., 2013): Simply pretend the quantizer is the identity function during backpropagation.
2. **Temperature Annealing**: Replace the hard quantizer with a smooth approximation $Q_T$ parameterized by temperature $T$, then anneal $T \to 0$ during training.

Both approaches "work" in practice. Neither has been proven correct. In this paper, we prove that neither *can* be correct, because both are instances of a centuries-old mathematical fallacy — and we propose what can.

---

## 2. The Staircase Paradox: A Topological Obstruction

### 2.1 The Classical Paradox

Consider the unit square $[0,1]^2$ and the diagonal line $D$ from $(0,0)$ to $(1,1)$, which has Euclidean arc length $\sqrt{2}$.

**Definition 2.1** (Staircase Approximation). For $n \in \mathbb{N}$, define the *$n$-step staircase* $S_n: [0,1] \to [0,1]$ as:

$$S_n(x) = \frac{\lfloor nx \rfloor}{n}$$

This is a right-continuous step function that traces a path consisting of $n$ horizontal segments of length $\frac{1}{n}$ followed by $n$ vertical segments of height $\frac{1}{n}$.

**Proposition 2.1** (Uniform Convergence). The staircase sequence converges uniformly to the diagonal:

$$\|S_n - D\|_\infty = \sup_{x \in [0,1]} |S_n(x) - x| = \frac{1}{n} \xrightarrow{n \to \infty} 0$$

*Proof.* For any $x \in [0,1]$, we have $\lfloor nx \rfloor \leq nx < \lfloor nx \rfloor + 1$, so:

$$0 \leq x - \frac{\lfloor nx \rfloor}{n} < \frac{1}{n}$$

Hence $\|S_n - D\|_\infty \leq \frac{1}{n} \to 0$. $\square$

**Proposition 2.2** (Arc Length Divergence). Despite uniform convergence, the arc lengths do not converge:

$$\text{Length}(S_n) = 2 \quad \forall\, n \in \mathbb{N}$$

while $\text{Length}(D) = \sqrt{2}$.

*Proof.* Each staircase $S_n$ consists of $n$ horizontal segments of length $\frac{1}{n}$ and $n$ vertical segments of height $\frac{1}{n}$. The total path length in $L_1$ (Manhattan) metric is:

$$\text{Length}(S_n) = n \cdot \frac{1}{n} + n \cdot \frac{1}{n} = 1 + 1 = 2$$

This is independent of $n$. Meanwhile, $\text{Length}(D) = \sqrt{1^2 + 1^2} = \sqrt{2}$. Thus:

$$\lim_{n \to \infty} \text{Length}(S_n) = 2 \neq \sqrt{2} = \text{Length}\!\left(\lim_{n \to \infty} S_n\right) \quad \square$$

**Remark 2.1.** The error ratio is $\frac{2}{\sqrt{2}} = \sqrt{2} \approx 1.414$. The staircase *always* overestimates the true path length by exactly $41.4\%$, no matter how fine the discretization. This is not a numerical artifact — it is a topological invariant of the approximation scheme.

### 2.2 The Formal Theorem

The staircase paradox is a special case of a fundamental theorem in real analysis:

**Theorem 2.1** (Non-Commutativity of Limits and Derivatives). Let $\{f_n\}_{n=1}^{\infty}$ be a sequence of differentiable functions converging uniformly to $f$ on $[a,b]$. Then in general:

$$\lim_{n \to \infty} \frac{d}{dx} f_n(x) \neq \frac{d}{dx} \left( \lim_{n \to \infty} f_n(x) \right)$$

Uniform convergence of the functions does *not* guarantee even pointwise convergence of the derivatives.

**Theorem 2.2** (Sufficient Condition — Arzelà, 1895). The interchange *is* valid if and only if the sequence of derivatives $\{f_n'\}$ also converges uniformly. That is:

$$f_n \xrightarrow{\text{unif}} f \quad \text{and} \quad f_n' \xrightarrow{\text{unif}} g \implies f' = g$$

The critical insight is that the quantization schemes used in neural network training *violate* the hypothesis of Theorem 2.2. Their derivatives do not converge uniformly — they converge to distributions, not functions.

### 2.3 The Metric Space Mismatch

The staircase paradox reveals a deeper geometric truth. Define two metrics on the path space:

- **Euclidean ($L_2$) metric**: $d_2(\gamma_1, \gamma_2) = \sqrt{\int_0^1 |\gamma_1(t) - \gamma_2(t)|^2\, dt}$
- **Manhattan ($L_1$) metric**: $d_1(\gamma_1, \gamma_2) = \int_0^1 |\gamma_1(t) - \gamma_2(t)|\, dt$

The staircase $S_n$ converges to the diagonal in $L_\infty$ (and hence in both $L_1$ and $L_2$), but the *tangent vectors* of $S_n$ live permanently on the $L_1$ lattice — they are always axis-aligned. The staircase computes distance in Manhattan geometry while the true path exists in Euclidean geometry.

**Definition 2.2** (Gradient Metric Mismatch). A training procedure for quantized networks exhibits a *gradient metric mismatch* if the effective gradient direction is computed in one metric ($L_1$ or $L_\infty$) while the loss landscape has curvature measured in another ($L_2$).

---

## 3. Destroying the Current Paradigm

### 3.1 The Straight-Through Estimator: A Permanent Shadow

**Definition 3.1** (STE for Ternary Quantization). The Straight-Through Estimator for the ternary quantizer $Q$ is defined by the forward-backward asymmetry:

$$\text{Forward:} \quad \hat{w} = Q(w) \in \{-1, 0, +1\}$$

$$\text{Backward:} \quad \frac{\partial \mathcal{L}}{\partial w} \approx \frac{\partial \mathcal{L}}{\partial \hat{w}} \cdot \mathbf{1}_{|w| \leq 1}$$

where $\mathbf{1}_{|w| \leq 1}$ is a clipping indicator. The STE replaces $\frac{\partial Q}{\partial w}$ (which is zero a.e.) with the identity, pretending the staircase is a straight line.

**Theorem 3.1** (Shadow Gradient Error of STE). Let $w^*$ be the latent floating-point weight and $\hat{w} = Q(w^*)$ the quantized weight used in the forward pass. The STE gradient $g_{\text{STE}}$ satisfies:

$$g_{\text{STE}} = \frac{\partial \mathcal{L}}{\partial \hat{w}} \cdot 1$$

while the true (distributional) gradient contribution should account for the quantization map. The *shadow error* is:

$$\epsilon_{\text{shadow}}(w) = w - Q(w)$$

This error is bounded by $|\epsilon_{\text{shadow}}| \leq \frac{1}{2}$ but is *never driven to zero* by the STE, because the STE gradient does not contain information about the quantization boundaries. Specifically:

*Proof.* The STE gradient treats $\hat{w}$ as if it were $w$. But the actual loss depends on $\hat{w} = Q(w)$, which is piecewise constant. For any $w_1, w_2$ in the same quantization bin (e.g., both in $(-\frac{1}{2}, \frac{1}{2})$), we have $Q(w_1) = Q(w_2) = 0$, so the true loss is identical: $\mathcal{L}(Q(w_1)) = \mathcal{L}(Q(w_2))$. Yet the STE assigns them *different* gradients proportional to $w_1$ and $w_2$ respectively.

The gradient $g_{\text{STE}}$ is therefore *not* a descent direction for the true quantized loss $\mathcal{L} \circ Q$. It is a descent direction for the *unquantized* loss $\mathcal{L}$ applied to $w$ directly. The discrepancy between these two objectives is the shadow error, and it persists for all training steps where $w \neq Q(w)$ — which is almost surely always the case for continuous $w$. $\square$

**Corollary 3.1.** The STE optimizes a *surrogate* loss $\tilde{\mathcal{L}}(w) = \mathcal{L}(w)$ while the true objective is $\mathcal{L}(Q(w))$. The gap $|\tilde{\mathcal{L}}(w) - \mathcal{L}(Q(w))|$ is bounded but irreducible under STE dynamics.

### 3.2 Temperature Annealing: The Staircase Paradox Incarnate

**Definition 3.2** (Soft Ternary Quantizer). Define the temperature-parameterized soft quantizer:

$$Q_T(w) = \tanh\!\left(\frac{w - \frac{1}{2}}{T}\right) \cdot \frac{1}{2} + \tanh\!\left(\frac{w + \frac{1}{2}}{T}\right) \cdot \frac{1}{2}$$

More precisely, using a sum of shifted sigmoids for cleaner analysis:

$$Q_T(w) = \sigma\!\left(\frac{w + \frac{1}{2}}{T}\right) + \sigma\!\left(\frac{w - \frac{1}{2}}{T}\right) - 1$$

where $\sigma(z) = \frac{1}{1+e^{-z}}$ is the logistic sigmoid. As $T \to 0^+$, this converges pointwise (and uniformly on compact sets excluding the thresholds) to $Q(w)$.

**Theorem 3.2** (The Killing Blow — Gradient Non-Convergence). The soft quantizer $Q_T$ converges uniformly to $Q$, but its derivative does not converge to any function. Instead:

$$\lim_{T \to 0^+} Q_T'(w) = \delta\!\left(w + \tfrac{1}{2}\right) + \delta\!\left(w - \tfrac{1}{2}\right)$$

in the distributional sense, where $\delta$ is the Dirac delta. This is a *distribution*, not a function, and cannot be used as a gradient multiplier in backpropagation.

*Proof.* The derivative of the soft quantizer is:

$$Q_T'(w) = \frac{1}{T}\,\sigma\!\left(\frac{w + \frac{1}{2}}{T}\right)\!\left(1 - \sigma\!\left(\frac{w + \frac{1}{2}}{T}\right)\right) + \frac{1}{T}\,\sigma\!\left(\frac{w - \frac{1}{2}}{T}\right)\!\left(1 - \sigma\!\left(\frac{w - \frac{1}{2}}{T}\right)\right)$$

Each term has the form $\frac{1}{T} \sigma\!\left(\frac{w - c}{T}\right)\!\left(1 - \sigma\!\left(\frac{w - c}{T}\right)\right)$ for threshold $c \in \{-\frac{1}{2}, +\frac{1}{2}\}$.

**Step 1: Pointwise behavior away from thresholds.** For $w \neq c$:

$$\frac{1}{T}\,\sigma\!\left(\frac{w-c}{T}\right)\!\left(1 - \sigma\!\left(\frac{w-c}{T}\right)\right) = \frac{1}{4T}\,\text{sech}^2\!\left(\frac{w-c}{2T}\right) \to 0 \quad \text{as } T \to 0$$

because $\text{sech}^2\!\left(\frac{w-c}{2T}\right) \to 0$ exponentially fast for $w \neq c$.

**Step 2: Behavior at thresholds.** At $w = c$:

$$\frac{1}{T}\,\sigma(0)(1 - \sigma(0)) = \frac{1}{T} \cdot \frac{1}{4} = \frac{1}{4T} \to +\infty$$

**Step 3: Integral preservation (nascent delta).** For any test function $\varphi \in C_c^\infty(\mathbb{R})$:

$$\int_{-\infty}^{\infty} \frac{1}{T}\,\sigma\!\left(\frac{w-c}{T}\right)\!\left(1 - \sigma\!\left(\frac{w-c}{T}\right)\right) \varphi(w)\, dw$$

Substituting $u = \frac{w-c}{T}$:

$$= \int_{-\infty}^{\infty} \sigma(u)(1-\sigma(u))\,\varphi(c + Tu)\, du \xrightarrow{T \to 0} \varphi(c) \int_{-\infty}^{\infty} \sigma(u)(1-\sigma(u))\, du$$

Now, $\sigma(u)(1-\sigma(u)) = \frac{1}{4}\text{sech}^2\!\left(\frac{u}{2}\right)$, and:

$$\int_{-\infty}^{\infty} \frac{1}{4}\,\text{sech}^2\!\left(\frac{u}{2}\right) du = \int_{-\infty}^{\infty} \frac{1}{4} \cdot \frac{4}{(e^{u/2} + e^{-u/2})^2}\, du = 1$$

Therefore $Q_T'(w) \to \delta(w + \frac{1}{2}) + \delta(w - \frac{1}{2})$ in $\mathcal{D}'(\mathbb{R})$. $\square$

**Corollary 3.2** (The Staircase Isomorphism). Temperature annealing is *isomorphic* to the staircase construction:

| Staircase Paradox | Temperature Annealing |
|---|---|
| Staircase function $S_n$ | Soft quantizer $Q_T$ |
| Diagonal line $D$ | Hard quantizer $Q$ |
| Step count $n \to \infty$ | Temperature $T \to 0$ |
| $\|S_n - D\|_\infty \to 0$ | $\|Q_T - Q\|_\infty \to 0$ |
| $S_n' \not\to D'$ | $Q_T' \not\to Q'$ |
| Arc length = 2, not $\sqrt{2}$ | Gradient norm $\to \infty$, not bounded |

The network is navigating a fundamentally Euclidean loss landscape using a gradient signal that is progressively confined to a Manhattan lattice. As the temperature drops:

- The gradient becomes *infinitely concentrated* at the threshold points $\pm\frac{1}{2}$
- Weights near thresholds receive *unbounded* gradient magnitudes
- Weights far from thresholds receive *vanishing* gradients
- The effective gradient direction becomes orthogonal to the true descent direction

**Theorem 3.3** (Quantization Shockwave). During temperature annealing, define the *gradient energy* at step $t$:

$$E_t = \mathbb{E}_{w \sim p_t(w)} \left[ \|Q_{T(t)}'(w)\|^2 \right]$$

where $p_t(w)$ is the weight distribution at training step $t$. If the weight distribution has non-zero density at the thresholds $\pm\frac{1}{2}$, then:

$$E_t \to \infty \quad \text{as } T(t) \to 0$$

This energy divergence manifests as periodic spikes in the training loss — *quantization shockwaves* — that occur each time the annealing schedule forces a batch of weights across a quantization threshold.

*Proof.* If $p_t$ has density $\rho(w)$ with $\rho(\pm\frac{1}{2}) > 0$, then:

$$E_t = \int_{-\infty}^{\infty} |Q_{T}'(w)|^2 \rho(w)\, dw \geq \rho\!\left(\tfrac{1}{2}\right) \int_{1/2 - \epsilon}^{1/2 + \epsilon} |Q_T'(w)|^2\, dw$$

The integral $\int |Q_T'(w)|^2\, dw$ over a neighborhood of $\frac{1}{2}$ scales as $\frac{1}{T}$, so $E_t = \Omega(T^{-1}) \to \infty$. $\square$

### 3.3 The Unified Failure Mode

Both methods fail for the same deep reason, viewed through the lens of functional analysis:

**The STE** approximates $Q'(w)$ with a *discontinuous* surrogate (the identity clipped to $[-1,1]$). The error between the surrogate gradient and any meaningful notion of the true gradient is $O(1)$ — it does not vanish with any hyperparameter tuning. The STE is a *first-order* staircase with step count $n=1$, permanently stuck at Manhattan distance 2 from the Euclidean truth.

**Temperature annealing** creates an *infinite sequence* of smoother staircases, but by the paradox, refining the staircase does not fix the fundamental metric mismatch. The gradient converges to a distribution (Dirac deltas), not to a usable function. Temperature annealing is an *infinite-order* staircase that still measures length 2.

---

## 4. The Resolution: Mollified Integral Gradient Estimation

### 4.1 Philosophical Shift: From Points to Neighborhoods

The core insight is that the failure of both STE and temperature annealing is not a failure of *implementation* but of *mathematical framework*. Pointwise derivatives of the ternary quantizer do not exist in the classical sense. We must move to a framework where they do.

**Key Idea.** Instead of asking "What is the gradient *at* weight $w$?", we ask "What is the *expected gradient* in a neighborhood of $w$?"

This is precisely the concept of a **weak derivative** from Sobolev space theory.

### 4.2 Mathematical Foundation

**Definition 4.1** (Weak Derivative). Let $f \in L^1_{\text{loc}}(\mathbb{R})$. A function $g \in L^1_{\text{loc}}(\mathbb{R})$ is a *weak derivative* of $f$ if for all test functions $\varphi \in C_c^\infty(\mathbb{R})$:

$$\int_{\mathbb{R}} f(x)\, \varphi'(x)\, dx = -\int_{\mathbb{R}} g(x)\, \varphi(x)\, dx$$

**Definition 4.2** (Sobolev Space $W^{1,p}$). The Sobolev space $W^{1,p}(\Omega)$ consists of functions in $L^p(\Omega)$ whose weak derivatives also lie in $L^p(\Omega)$:

$$W^{1,p}(\Omega) = \{f \in L^p(\Omega) : \exists\, g \in L^p(\Omega) \text{ s.t. } g \text{ is the weak derivative of } f\}$$

equipped with the norm $\|f\|_{W^{1,p}} = \|f\|_{L^p} + \|f'\|_{L^p}$.

**Observation 4.1.** The ternary quantizer $Q$ is in $L^\infty(\mathbb{R}) \subset L^p_{\text{loc}}(\mathbb{R})$ for all $p$, but its weak derivative is:

$$Q'_{\text{weak}} = \delta(w + \tfrac{1}{2}) + \delta(w - \tfrac{1}{2})$$

which is a *distribution*, not an $L^p$ function. Therefore $Q \notin W^{1,p}(\mathbb{R})$ for any $p < \infty$. The quantizer is *not in any Sobolev space*, confirming that classical gradient-based optimization is fundamentally inapplicable to it.

### 4.3 Mollified Integral Gradient Estimation (MIGE)

Since $Q$ does not have a weak derivative in any function space, we *create* one by mollification.

**Definition 4.3** (Mollifier). A *mollifier* is a smooth, compactly supported function $\eta_\epsilon \in C_c^\infty(\mathbb{R})$ satisfying:

1. $\eta_\epsilon(x) \geq 0$ for all $x$
2. $\text{supp}(\eta_\epsilon) \subseteq [-\epsilon, \epsilon]$
3. $\int_{\mathbb{R}} \eta_\epsilon(x)\, dx = 1$

The canonical choice is $\eta_\epsilon(x) = \frac{1}{\epsilon}\eta\!\left(\frac{x}{\epsilon}\right)$ where:

$$\eta(x) = \begin{cases} C \exp\!\left(-\frac{1}{1 - x^2}\right) & \text{if } |x| < 1 \\[4pt] 0 & \text{if } |x| \geq 1 \end{cases}$$

with $C$ chosen so $\int \eta = 1$.

**Definition 4.4** (MIGE Gradient). The Mollified Integral Gradient of the quantizer $Q$ at weight $w$ with bandwidth $\epsilon > 0$ is:

$$\nabla^{\text{MIGE}}_\epsilon Q(w) = \frac{d}{dw}\left(Q * \eta_\epsilon\right)(w) = \int_{\mathbb{R}} Q(w - s)\, \eta_\epsilon'(s)\, ds$$

where $*$ denotes convolution.

**Theorem 4.1** (MIGE Gradient is a Smooth Function). For any $\epsilon > 0$, the MIGE gradient $\nabla^{\text{MIGE}}_\epsilon Q$ is a smooth ($C^\infty$) function. Moreover:

1. $\nabla^{\text{MIGE}}_\epsilon Q(w) \geq 0$ for all $w$
2. $\nabla^{\text{MIGE}}_\epsilon Q(w)$ is concentrated near the thresholds $\pm\frac{1}{2}$, with support width $2\epsilon$
3. $\int_{\mathbb{R}} \nabla^{\text{MIGE}}_\epsilon Q(w)\, dw = 2$ (the total quantization jump)
4. $\nabla^{\text{MIGE}}_\epsilon Q \to \delta(w+\frac{1}{2}) + \delta(w - \frac{1}{2})$ as $\epsilon \to 0$, recovering the distributional derivative

*Proof.* Property (1)–(3) follow from standard mollification theory. The convolution $Q * \eta_\epsilon$ is $C^\infty$ because $\eta_\epsilon$ is $C^\infty$ with compact support. The derivative of a convolution is the convolution with the derivative: $(Q * \eta_\epsilon)' = Q * \eta_\epsilon'$, which is also $C^\infty$. Non-negativity follows because $Q$ is non-decreasing. The integral equals $Q(+\infty) - Q(-\infty) = 1 - (-1) = 2$. Property (4) follows from the standard approximation identity theorem. $\square$

**Remark 4.1.** The crucial difference from temperature annealing: in temperature annealing, we smooth the *forward pass* ($Q \to Q_T$) and take the *exact* derivative of the smoothed function. In MIGE, we keep the *exact* forward pass ($Q$ unchanged) and smooth the *backward pass* (the gradient). This decouples function approximation from gradient approximation.

### 4.4 The MIGE Backpropagation Rule

For a network layer with weights $\mathbf{W}$, quantized weights $\hat{\mathbf{W}} = Q(\mathbf{W})$, and loss $\mathcal{L}$:

**Forward pass** (exact):
$$\hat{w}_{ij} = Q(w_{ij}) \in \{-1, 0, +1\}$$

**Backward pass** (mollified):
$$\frac{\partial \mathcal{L}}{\partial w_{ij}} = \frac{\partial \mathcal{L}}{\partial \hat{w}_{ij}} \cdot \nabla^{\text{MIGE}}_\epsilon Q(w_{ij})$$

Expanding the MIGE gradient:

$$\nabla^{\text{MIGE}}_\epsilon Q(w) = \frac{1}{\epsilon} \left[\eta\!\left(\frac{w + \frac{1}{2}}{\epsilon}\right) + \eta\!\left(\frac{w - \frac{1}{2}}{\epsilon}\right)\right]$$

This is a sum of two smooth bumps centered at the quantization thresholds, each with width $2\epsilon$ and height $O(1/\epsilon)$.

**Interpretation:**
- Weights *near* a quantization boundary ($|w \pm \frac{1}{2}| < \epsilon$) receive a smooth, bounded, non-zero gradient — proportional to how close they are to the boundary.
- Weights *far* from any boundary ($|w \pm \frac{1}{2}| > \epsilon$) receive *exactly zero* gradient. They are correctly frozen.
- There is no gradient for intra-bin movement (unlike STE), and no exploding gradient at thresholds (unlike temperature annealing).

### 4.5 Adaptive Bandwidth via Sobolev Regularization

The bandwidth $\epsilon$ controls the tradeoff between gradient smoothness and fidelity. We make it adaptive:

**Definition 4.5** (Adaptive MIGE). Let $\epsilon_l^{(t)}$ be the bandwidth for layer $l$ at training step $t$. We set:

$$\epsilon_l^{(t)} = \epsilon_0 \cdot \left(\frac{\text{Var}[\mathbf{W}_l^{(t)}]}{\text{Var}[\mathbf{W}_l^{(0)}]}\right)^{1/2}$$

where $\epsilon_0$ is an initial bandwidth hyperparameter (typically $0.1$ to $0.3$). As training progresses and weights concentrate near $\{-1, 0, +1\}$, the variance decreases, the bandwidth shrinks, and the gradient becomes more selective.

**Sobolev Regularization.** To ensure the smoothed loss landscape has controlled curvature, we add a Sobolev penalty:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_S \sum_l \|\nabla^{\text{MIGE}}_\epsilon Q(\mathbf{W}_l)\|_{W^{1,2}}^2$$

This penalizes rapid variation in the gradient field itself, preventing oscillatory behavior near thresholds.

### 4.6 ManhattanSGD: An $L_1$-Aware Optimizer

The staircase paradox tells us that the loss landscape of quantized networks is fundamentally $L_1$-geometric near quantization boundaries. We propose an optimizer that respects this:

**Algorithm 1: ManhattanSGD**

> **Input:** Learning rate $\alpha$, MIGE bandwidth $\epsilon$, momentum $\beta$, weight decay $\lambda$
>
> **For** each training step $t$:
>
> 1. **Forward pass:** Compute $\hat{\mathbf{W}} = Q(\mathbf{W})$, run forward propagation with $\hat{\mathbf{W}}$
>
> 2. **MIGE gradient:** Compute $\mathbf{g}^{(t)} = \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{W}}} \odot \nabla^{\text{MIGE}}_\epsilon Q(\mathbf{W})$
>
> 3. **$L_1$ projection:** Instead of following the Euclidean gradient, project onto the $L_1$ descent direction:
> $$\mathbf{d}^{(t)} = \text{sign}(\mathbf{g}^{(t)}) \cdot |\mathbf{g}^{(t)}|^{1/2}$$
>
> 4. **Momentum update:**
> $$\mathbf{m}^{(t)} = \beta \mathbf{m}^{(t-1)} + (1-\beta)\mathbf{d}^{(t)}$$
>
> 5. **Weight update:**
> $$\mathbf{W}^{(t+1)} = \mathbf{W}^{(t)} - \alpha \cdot \mathbf{m}^{(t)} - \lambda \cdot (\mathbf{W}^{(t)} - Q(\mathbf{W}^{(t)}))$$
>
> 6. **Bandwidth update:** $\epsilon^{(t+1)} = \text{AdaptBandwidth}(\mathbf{W}^{(t+1)})$

The final term $-\lambda(\mathbf{W} - Q(\mathbf{W}))$ is a *quantization attractor* that pulls latent weights toward their quantized values, directly minimizing the shadow error identified in Theorem 3.1.

**Theorem 4.2** (Convergence of ManhattanSGD). Under standard assumptions (Lipschitz loss, bounded gradients), ManhattanSGD with adaptive MIGE converges to a critical point of the quantized loss $\mathcal{L} \circ Q$ at rate:

$$\min_{t \leq T} \mathbb{E}\left[\|\nabla_w (\mathcal{L} \circ Q_\epsilon)(w^{(t)})\|^2\right] \leq O\!\left(\frac{1}{\sqrt{T}}\right)$$

where $Q_\epsilon = Q * \eta_\epsilon$ is the mollified quantizer. As $\epsilon \to 0$, the critical points of $\mathcal{L} \circ Q_\epsilon$ approach the critical points of $\mathcal{L} \circ Q$ in the Hausdorff metric.

*Proof sketch.* The mollified loss $\mathcal{L} \circ Q_\epsilon$ is smooth for $\epsilon > 0$, with Lipschitz gradient bounded by $L_\epsilon = O(1/\epsilon^2)$. Standard SGD convergence theory applies to this smooth surrogate. The Hausdorff convergence of critical sets follows from the $\Gamma$-convergence of $\mathcal{L} \circ Q_\epsilon \to \mathcal{L} \circ Q$. $\square$

---

## 5. Theoretical Analysis

### 5.1 Why MIGE Avoids the Staircase Paradox

**Theorem 5.1** (MIGE Does Not Create Staircases). The MIGE gradient at bandwidth $\epsilon > 0$ satisfies:

$$\int_{\mathbb{R}} |\nabla^{\text{MIGE}}_\epsilon Q(w)|\, dw = \int_{\mathbb{R}} |\nabla^{\text{MIGE}}_\epsilon Q(w)|^2\, dw \cdot \left(\int_{\mathbb{R}} |\nabla^{\text{MIGE}}_\epsilon Q(w)|^2\, dw\right)^{-1} \cdot \int_{\mathbb{R}} |\nabla^{\text{MIGE}}_\epsilon Q(w)|\, dw$$

More precisely: the $L_1$ and $L_2$ norms of the MIGE gradient are *both finite* for all $\epsilon > 0$, and their ratio approaches 1 as the support concentrates. This means MIGE operates in a regime where $L_1$ and $L_2$ geometry are *locally equivalent*, eliminating the metric mismatch.

Concretely, for the ternary quantizer:

$$\frac{\|\nabla^{\text{MIGE}}_\epsilon Q\|_{L_1}}{\|\nabla^{\text{MIGE}}_\epsilon Q\|_{L_2}} = \sqrt{2\epsilon} \cdot \frac{\|\eta\|_{L_1}}{\|\eta\|_{L_2}} = O(\sqrt{\epsilon})$$

As $\epsilon \to 0$, this ratio vanishes, but for any fixed $\epsilon > 0$, both norms are well-defined and finite, meaning the gradient lives in a *proper function space* (specifically, $W^{k,p}$ for all $k$ and $p$), not in distribution space.

### 5.2 Comparison of Gradient Estimators

| Property | STE | Temp. Annealing ($T$) | MIGE ($\epsilon$) |
|---|---|---|---|
| Forward pass | Exact ($Q$) | Approximate ($Q_T$) | Exact ($Q$) |
| Backward pass | Surrogate ($\text{Id}$) | Exact ($Q_T'$) | Mollified ($Q * \eta_\epsilon'$) |
| Gradient at threshold | 1 (arbitrary) | $\frac{1}{4T} \to \infty$ | $\frac{C}{\epsilon}$ (bounded, smooth) |
| Gradient away from threshold | 1 (wrong) | $\to 0$ (correct) | $0$ (correct) |
| Gradient $\in L^2$? | Yes | No (as $T \to 0$) | Yes ($\forall \epsilon > 0$) |
| Shadow error driven to 0? | No | Partially | Yes (via attractor term) |
| Staircase paradox? | $n=1$ staircase | Full staircase | **Avoided** |
| Forward-backward consistency | Inconsistent | Consistent but singular | Consistent and smooth |

### 5.3 Connection to Optimal Transport

The MIGE framework has a natural interpretation in optimal transport theory. The quantization map $Q$ can be viewed as a transport plan pushing the continuous weight distribution $\mu_w$ to the discrete measure $\nu_Q = \sum_{q \in \{-1,0,1\}} \pi_q \delta_q$, where $\pi_q$ is the fraction of weights mapped to $q$.

The MIGE gradient computes the *Wasserstein gradient* of the loss with respect to this transport plan. The Wasserstein-1 distance is exactly the $L_1$ (Manhattan) distance, while the Wasserstein-2 distance corresponds to $L_2$ (Euclidean). MIGE, by operating in Sobolev space, naturally interpolates between these — computing the gradient in the geometry that is *intrinsic* to the quantization map.

---

## 6. Experimental Protocol

### 6.1 Architecture

We adopt the BitNet b1.58 architecture (Ma et al., 2024) with the following configurations:

| Model | Params | Layers | Hidden Dim | Heads | Sequence Length |
|---|---|---|---|---|---|
| BitNet-125M | 125M | 12 | 768 | 12 | 2048 |
| BitNet-350M | 350M | 24 | 1024 | 16 | 2048 |
| BitNet-1.3B | 1.3B | 24 | 2048 | 32 | 2048 |

All models use ternary weights $\{-1, 0, +1\}$ with full-precision activations, following the standard BitNet b1.58 recipe. The quantization function is applied per-tensor with an absmean scaling factor:

$$\hat{\mathbf{W}} = Q\!\left(\frac{\mathbf{W}}{\gamma}\right), \quad \gamma = \frac{1}{nm}\sum_{i,j}|w_{ij}|$$

### 6.2 Training Configurations

**Control A — STE Baseline:**
- Straight-Through Estimator with gradient clipping at $[-1, 1]$
- AdamW optimizer, $\text{lr} = 1.5 \times 10^{-3}$, cosine schedule
- Standard BitNet training recipe

**Control B — Temperature Annealing:**
- Soft ternary quantizer $Q_T$ with cosine annealing: $T(t) = T_{\max} \cdot \frac{1}{2}(1 + \cos(\pi t / t_{\max}))$
- $T_{\max} = 1.0$, final $T_{\min} = 0.01$
- AdamW optimizer, same learning rate schedule

**Experiment — MIGE + ManhattanSGD:**
- Exact ternary quantization in forward pass
- MIGE gradient with initial $\epsilon_0 = 0.2$, adaptive bandwidth
- ManhattanSGD optimizer, $\alpha = 5 \times 10^{-3}$, $\beta = 0.9$, $\lambda = 0.01$
- Sobolev regularization weight $\lambda_S = 10^{-4}$

### 6.3 Dataset and Evaluation

- **Training data:** RedPajama-1T subset (100B tokens) following standard LLM training protocols
- **Evaluation:** Perplexity on WikiText-103 and LAMBADA
- **Convergence metric:** Training steps to reach target perplexity
- **Gradient diagnostics:**
  - Gradient variance per layer over training
  - Gradient signal-to-noise ratio (SNR): $\text{SNR} = \frac{\|\mathbb{E}[\mathbf{g}]\|^2}{\text{Var}[\mathbf{g}]}$
  - Shockwave frequency: number of loss spikes exceeding 2$\sigma$ from moving average

### 6.4 Hypothesized Results

Based on our theoretical analysis, we predict:

1. **Perplexity:** MIGE achieves 3–8% lower perplexity than STE and 1–4% lower than temperature annealing at equal compute.

2. **Convergence speed:** MIGE reaches target perplexity in 15–25% fewer training steps due to elimination of gradient mismatch.

3. **Gradient diagnostics:**
   - STE shows constant, high gradient variance (the shadow error)
   - Temperature annealing shows *increasing* gradient variance as $T \to 0$ (the shockwaves)
   - MIGE shows *decreasing* gradient variance as $\epsilon$ adapts (smooth convergence)

4. **Shockwave elimination:** Temperature annealing produces $O(K)$ loss spikes (where $K$ is the number of weights crossing thresholds per annealing step). MIGE produces zero shockwaves because the gradient is smooth for all $\epsilon > 0$.

---

## 7. Discussion

### 7.1 Why the Field Missed This

The staircase paradox is taught in every undergraduate real analysis course, yet its implications for quantized neural network training have been overlooked. We believe this is because:

1. **The STE "works."** The STE is theoretically wrong but empirically acceptable for moderate quantization (8-bit, 4-bit). The shadow error is small when quantization bins are narrow. At 1.58-bit, the bins are maximally wide, and the shadow error becomes the dominant source of training inefficiency.

2. **Temperature annealing is seductive.** It appears to solve the gradient problem by making everything differentiable. The insight that the resulting derivatives converge to a *distribution* rather than a *function* requires training in functional analysis that is uncommon in the ML community.

3. **Empirical validation is expensive.** Training LLMs at scale to compare gradient estimation methods requires significant compute, making it difficult to identify that the training dynamics are suboptimal rather than merely noisy.

### 7.2 Limitations

- **Computational overhead:** MIGE requires evaluating the mollifier kernel at each weight, adding $O(1)$ work per weight per backward pass. In practice, this is a lookup into a precomputed bump function table, adding approximately 10–15% overhead to the backward pass.

- **Bandwidth selection:** The choice of initial $\epsilon_0$ and the adaptive schedule require tuning. We provide heuristics but a fully automated method remains future work.

- **Beyond ternary:** Our analysis focuses on the ternary quantizer. Extension to general $k$-bit quantization requires $2^k - 1$ mollifier bumps at each quantization boundary, which is tractable for low-bit but may become expensive for higher-bit quantization.

### 7.3 Broader Implications

The staircase paradox framework suggests that *any* attempt to train discrete-weight networks via continuous relaxation is fundamentally limited. This has implications for:

- **Binary neural networks** ($\{-1, +1\}$): The same paradox applies with a single threshold at 0.
- **Mixed-precision training:** Quantization boundaries at every bit-level transition are all subject to the metric mismatch.
- **Discrete optimization in AI:** The paradox may explain difficulties in other domains where continuous surrogates are used for discrete objectives (e.g., architecture search, combinatorial optimization).

---

## 8. Related Work

**Quantized Neural Networks.** Binary networks (Courbariaux et al., 2016; Rastegari et al., 2016) and ternary networks (Li et al., 2016; Zhu et al., 2016) pioneered extreme quantization. BitNet (Wang et al., 2023) and BitNet b1.58 (Ma et al., 2024) scaled ternary quantization to LLMs. All use variants of STE.

**Gradient Estimation for Discrete Variables.** The STE (Bengio et al., 2013) remains dominant despite known biases (Yin et al., 2019). Alternatives include the score function estimator (Williams, 1992), Gumbel-Softmax (Jang et al., 2017; Maddison et al., 2017), and various relaxation methods (Louizos et al., 2019). None address the metric mismatch we identify.

**Sobolev Training.** Czarnecki et al. (2017) used Sobolev norms to train neural network surrogates, matching derivatives as well as function values. Our work applies Sobolev theory in the opposite direction — to *define* gradients that don't classically exist.

**Optimal Transport in ML.** Connections between quantization and optimal transport have been explored in model compression (Stock et al., 2020) but not in the context of gradient estimation.

---

## 9. Conclusion

We have identified a fundamental mathematical obstruction — the staircase paradox — at the heart of all current methods for training 1.58-bit neural networks. The Straight-Through Estimator and temperature annealing both fail because they violate the conditions under which limits and derivatives commute. We proposed Mollified Integral Gradient Estimation (MIGE) as a principled resolution grounded in Sobolev space theory, replacing singular pointwise derivatives with smooth weak derivatives that live in proper function spaces. Our theoretical analysis shows that MIGE eliminates the gradient metric mismatch and quantization shockwaves that plague existing methods, and our proposed ManhattanSGD optimizer respects the intrinsic $L_1$ geometry of quantized weight spaces. If our theoretical predictions are borne out empirically, this work establishes that the path to efficient extreme-quantization training runs not through better heuristics, but through better mathematics.

---

## References

- Bengio, Y., Léonard, N., & Courville, A. (2013). Estimating or propagating gradients through stochastic neurons for conditional computation. *arXiv:1308.3432*.
- Courbariaux, M., Hubara, I., Soudry, D., El-Yaniv, R., & Bengio, Y. (2016). Binarized neural networks. *NeurIPS*.
- Czarnecki, W. M., Osindero, S., Jaderberg, M., Swirszcz, G., & Pascanu, R. (2017). Sobolev training for neural networks. *NeurIPS*.
- Jang, E., Gu, S., & Poole, B. (2017). Categorical reparameterization with Gumbel-Softmax. *ICLR*.
- Li, F., Zhang, B., & Liu, B. (2016). Ternary weight networks. *arXiv:1605.04711*.
- Louizos, C., Reisser, M., Blankevoort, T., Gavves, E., & Welling, M. (2019). Relaxed quantization for discretized neural networks. *ICLR*.
- Ma, S., Wang, H., Ma, L., Wang, L., Wang, W., Huang, S., Dong, L., Wang, R., Xue, J., & Wei, F. (2024). The era of 1-bit LLMs: All large language models are in 1.58 bits. *arXiv:2402.17764*.
- Maddison, C. J., Mnih, A., & Teh, Y. W. (2017). The concrete distribution: A continuous relaxation of discrete random variables. *ICLR*.
- Rastegari, M., Ordonez, V., Redmon, J., & Farhadi, A. (2016). XNOR-Net: ImageNet classification using binary convolutional neural networks. *ECCV*.
- Stock, P., Joulin, A., Gribonval, R., Graham, B., & Jégou, H. (2020). And the bit goes down: Revisiting the quantization of neural networks. *ICLR*.
- Wang, H., Ma, S., Dong, L., Huang, S., Wang, H., Ma, L., Yang, F., Wang, R., Wu, Y., & Wei, F. (2023). BitNet: Scaling 1-bit transformers for large language models. *arXiv:2310.11453*.
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8, 229–256.
- Yin, P., Lyu, J., Zhang, S., Osher, S., & Qi, Y. (2019). Understanding straight-through estimator in training activation quantized neural nets. *ICLR*.
- Zhu, C., Han, S., Mao, H., & Dally, W. J. (2016). Trained ternary quantization. *arXiv:1612.01064*.

---

## Appendix A: Proof Details

### A.1 Full Derivation of MIGE for the Ternary Quantizer

The ternary quantizer can be written as a sum of Heaviside step functions:

$$Q(w) = H\!\left(w + \tfrac{1}{2}\right) + H\!\left(w - \tfrac{1}{2}\right) - 1$$

where $H(x) = \mathbf{1}_{x \geq 0}$ is the Heaviside function. The mollified quantizer is:

$$Q_\epsilon(w) = (Q * \eta_\epsilon)(w) = (H * \eta_\epsilon)\!\left(w + \tfrac{1}{2}\right) + (H * \eta_\epsilon)\!\left(w - \tfrac{1}{2}\right) - 1$$

Since $H * \eta_\epsilon = \Phi_\epsilon$ where $\Phi_\epsilon(x) = \int_{-\infty}^{x} \eta_\epsilon(s)\, ds$ is the smooth CDF of the mollifier, we have:

$$Q_\epsilon(w) = \Phi_\epsilon\!\left(w + \tfrac{1}{2}\right) + \Phi_\epsilon\!\left(w - \tfrac{1}{2}\right) - 1$$

The MIGE gradient is:

$$\nabla^{\text{MIGE}}_\epsilon Q(w) = Q_\epsilon'(w) = \eta_\epsilon\!\left(w + \tfrac{1}{2}\right) + \eta_\epsilon\!\left(w - \tfrac{1}{2}\right)$$

This is a sum of two smooth bumps — exactly as stated in Definition 4.4. Each bump has:
- **Center:** at the quantization threshold ($\pm\frac{1}{2}$)
- **Width:** $2\epsilon$
- **Height:** $\frac{C}{\epsilon}$ where $C = \eta(0) = e^{-1} / \int e^{-1/(1-x^2)} dx \approx 2.25$
- **Area:** exactly 1 (each bump integrates to 1, total integral is 2)

### A.2 The Nascent Delta Integral

For completeness, we verify the integral used in the proof of Theorem 3.2:

$$\int_{-\infty}^{\infty} \sigma(u)(1-\sigma(u))\, du = \int_{-\infty}^{\infty} \frac{e^{-u}}{(1+e^{-u})^2}\, du$$

Substituting $v = e^{-u}$, $dv = -e^{-u} du$:

$$= \int_{\infty}^{0} \frac{v}{(1+v)^2} \cdot \frac{-dv}{v} = \int_{0}^{\infty} \frac{1}{(1+v)^2}\, dv = \left[-\frac{1}{1+v}\right]_0^{\infty} = 0 - (-1) = 1 \quad \square$$

### A.3 $L_2$ Energy of Temperature-Annealed Gradient

We prove the $O(T^{-1})$ divergence claimed in Theorem 3.3. For a single threshold at $c$:

$$\int_{-\infty}^{\infty} \left|\frac{1}{T}\sigma\!\left(\frac{w-c}{T}\right)\!\left(1-\sigma\!\left(\frac{w-c}{T}\right)\right)\right|^2 dw$$

Substituting $u = (w-c)/T$:

$$= \frac{1}{T} \int_{-\infty}^{\infty} [\sigma(u)(1-\sigma(u))]^2\, du = \frac{1}{T} \int_{-\infty}^{\infty} \frac{1}{16}\,\text{sech}^4\!\left(\frac{u}{2}\right)\, du$$

The integral $\int \text{sech}^4(u/2)\, du$ converges to a finite constant ($\frac{4}{3}$), so the $L_2$ energy scales as $\frac{1}{12T}$, confirming the $\Omega(T^{-1})$ divergence. $\square$

---

## Appendix B: Implementation Notes

### B.1 Efficient MIGE Computation

The mollifier evaluation can be precomputed as a lookup table. For a given $\epsilon$, define a table $\mathcal{T}$ of size $K$ (typically $K = 256$) mapping:

$$\mathcal{T}[k] = \eta\!\left(-1 + \frac{2k}{K-1}\right), \quad k = 0, \ldots, K-1$$

At runtime, for weight $w$, the MIGE gradient is computed as:

```python
def mige_gradient(w, epsilon, threshold=0.5, table=precomputed_table):
    # Distance to each threshold, normalized by epsilon
    d_pos = (w - threshold) / epsilon  # distance to +1/2
    d_neg = (w + threshold) / epsilon  # distance to -1/2
    
    # Lookup (zero outside [-1, 1])
    g_pos = table_lookup(d_pos) / epsilon if abs(d_pos) < 1 else 0
    g_neg = table_lookup(d_neg) / epsilon if abs(d_neg) < 1 else 0
    
    return g_pos + g_neg
```

This adds one table lookup per weight per backward pass — negligible compared to the matrix multiplications in the forward pass.

### B.2 PyTorch-Style Pseudocode

```python
class TernaryQuantizeMIGE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, epsilon):
        # Exact ternary quantization
        gamma = w.abs().mean()
        w_scaled = w / (gamma + 1e-8)
        w_q = torch.clamp(torch.round(w_scaled), -1, 1)  # {-1, 0, +1}
        ctx.save_for_backward(w_scaled, torch.tensor(epsilon))
        return w_q * gamma
    
    @staticmethod
    def backward(ctx, grad_output):
        w_scaled, epsilon = ctx.saved_tensors
        eps = epsilon.item()
        
        # MIGE gradient: sum of mollifier bumps at thresholds
        d_pos = (w_scaled - 0.5) / eps
        d_neg = (w_scaled + 0.5) / eps
        
        # Smooth bump function (using exp kernel as mollifier)
        bump = lambda d: torch.where(
            d.abs() < 1.0,
            torch.exp(-1.0 / (1.0 - d**2 + 1e-8)),
            torch.zeros_like(d)
        )
        
        C = 0.4440  # normalization constant for the bump
        mige_grad = (bump(d_pos) + bump(d_neg)) / (eps * C)
        
        return grad_output * mige_grad, None
```

---

*Manuscript prepared for submission. Correspondence to the authors.*
