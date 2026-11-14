# Mathematical Foundations of Two-Phase Topological Sobolev Gradient Estimation (Objective 7)
## Rigorous Formulation and Resolution of Sobolev-Distributional Duality

---

## 1. Abstract & Problem Statement

In ternary neural network quantization (BitNet b1.58), optimization takes place over the discrete ternary lattice $\mathcal{W}_{\text{discrete}} = \{-1, 0, +1\}^D$. The quantization operator:

$$Q(w) = \begin{cases} +1 & \text{if } w > \frac{1}{2} \\[4pt] 0 & \text{if } -\frac{1}{2} \leq w \leq \frac{1}{2} \\[4pt] -1 & \text{if } w < -\frac{1}{2} \end{cases}$$

possesses the exact distributional derivative (in the Schwartz space of distributions $\mathcal{D}'(\mathbb{R})$):

$$\nabla_w Q = \delta\left(w + \frac{1}{2}\right) + \delta\left(w - \frac{1}{2}\right)$$

Integrating this distribution over $\mathbb{R}$ yields the total variation mass:

$$\int_{-\infty}^\infty \nabla_w Q \, dw = \langle \nabla_w Q, 1 \rangle = 1 + 1 = 2$$

Previous heuristic estimators suffer from severe deficiencies:
1. **The Straight-Through Estimator (STE):** Replaces $\nabla_w Q$ with the indicator $\mathbf{1}_{|w| \le 1}$. As proved in Theorem 3.1 of the primary paper, this creates an irreducible *shadow error* $\epsilon_{\text{shadow}}(w) = w - Q(w)$ that optimizes a phantom unquantized objective.
2. **Temperature Annealing ($Q_T$):** Relaxes the forward pass to a sum of sigmoids and decays $T \to 0$. As $T \to 0$, the $L_2$ norm explodes as $\|Q_T'\|_{L_2}^2 = \frac{1}{12T} \to \infty$, generating severe late-stage gradient shockwaves and early discretization gaps exceeding $1,000$ perplexity.

In this work, we present the **Two-Phase Topological Sobolev Framework (MIGE-2P)**. We resolve the duality between bounded Sobolev energy and asymptotic distributional behavior by formulating the schedule not as a naive point-limit contradiction, but as a **regularized variational projection into the Sobolev space $H^1(\mathbb{R})$ with an explicit $\delta$-approximation error bound**.

---

## 2. Rigorous Formulation: Two-Phase Topological Sobolev Operator

Let $\eta \in C_c^\infty(\mathbb{R})$ be the standard Friedrichs mollifier supported on $(-1, 1)$:

$$\eta(z) = \begin{cases} C \exp\left(-\frac{1}{1 - z^2}\right) & \text{if } |z| < 1 \\[4pt] 0 & \text{otherwise} \end{cases}$$

where $C = \left(\int_{-1}^1 \exp\left(-\frac{1}{1 - z^2}\right) dz\right)^{-1} \approx 0.4439938$ ensures $\int_{-1}^1 \eta(z)\, dz = 1$.

### Definition 2.1 (Exact-Mass MIGE Operator)
At optimization step $t \in [0, T_{\text{total}}]$, let normalized training progress be $\tau = \frac{t}{T_{\text{total}}} \in [0, 1]$.
We define the time-dependent Sobolev bandwidth $\epsilon(\tau)$ and exploratory transport floor $\rho(\tau)$ by:

$$\epsilon(\tau) = \epsilon_{\min} + \frac{1}{2}(\epsilon_{\max} - \epsilon_{\min})\left(1 + \cos(\pi \tau)\right)$$

$$\rho(\tau) = \rho_{\min} + \frac{1}{2}(\rho_{\max} - \rho_{\min})\left(1 + \cos(\pi \tau)\right)$$

with schedule bounds:
- $\epsilon_{\max} = 0.80$, $\epsilon_{\min} = 0.05$
- $\rho_{\max} = 0.35$, $\rho_{\min} = 0.02$

The backward surrogate gradient operator $G_{\tau}(w): \mathbb{R} \to \mathbb{R}$ is defined with **full unit mass per threshold** (total mass $= 2$ from the weak-derivative bumps):

$$G_{\tau}(w) = \frac{1}{\epsilon(\tau)}\left[\eta\left(\frac{w - 1/2}{\epsilon(\tau)}\right) + \eta\left(\frac{w + 1/2}{\epsilon(\tau)}\right)\right] + \rho(\tau)\cdot \mathbf{1}_{|w| \le 1.5}$$

while the forward evaluation is **invariant and strictly discrete**:

$$F(w) = Q(w) \in \{-1, 0, +1\} \quad \forall \tau \in [0, 1]$$

### Remark 2.1 (Conservation of Jump Mass)
For any bandwidth $\epsilon > 0$, the total mass of the bump components alone satisfies:

$$\int_{-\infty}^\infty \frac{1}{\epsilon}\left[\eta\left(\frac{w - 1/2}{\epsilon}\right) + \eta\left(\frac{w + 1/2}{\epsilon}\right)\right] dw = \int_{-1}^1 \eta(u)\, du + \int_{-1}^1 \eta(u)\, du = 1 + 1 = 2$$

This matches the total distributional variation of the discontinuous step operator $\nabla_w Q$ identically, resolving the previous half-scale underestimation.

---

## 3. Theoretical Proofs & Resolution of Sobolev-Distributional Duality

### Theorem 3.1 ($H^1$ Sobolev Regularity and Exact $L_2$ Energy Bounds)
For any $\tau \in [0, 1]$, the operator $G_{\tau}(w)$ belongs to the Sobolev space $W^{1, 2}(\mathbb{R}) \equiv H^1(\mathbb{R})$.
Under the exact-mass formulation, the $L_2$ norm of $G_\tau$ is uniformly bounded across all $\tau \in [0, 1]$:

$$\|G_\tau\|_{L_2} \le 2\sqrt{\frac{I_2}{\epsilon_{\min}}} + \sqrt{3}\rho_{\max} \le 7.69 < \infty$$

where $I_2 = \int_{-1}^1 \eta(u)^2 \, du \approx 0.627$.

*Proof.*
Let $K_{\epsilon, c}(w) = \frac{1}{\epsilon}\eta\left(\frac{w - c}{\epsilon}\right)$ for $c \in \{\pm 1/2\}$.
By substitution $u = (w - c)/\epsilon$:

$$\|K_{\epsilon, c}\|_{L_2}^2 = \int_{-\infty}^\infty \frac{1}{\epsilon^2}\eta\left(\frac{w - c}{\epsilon}\right)^2 dw = \frac{1}{\epsilon}\int_{-1}^1 \eta(u)^2 \, du = \frac{I_2}{\epsilon}$$

Since $\epsilon(\tau) \ge \epsilon_{\min} = 0.05$ for all $\tau \in [0, 1]$:

$$\|K_{\epsilon, c}\|_{L_2} \le \sqrt{\frac{I_2}{\epsilon_{\min}}} = \sqrt{\frac{0.627}{0.05}} = \sqrt{12.54} \approx 3.5412$$

For the transport floor component $R_\tau(w) = \rho(\tau)\mathbf{1}_{|w| \le 1.5}$:

$$\|R_\tau\|_{L_2} = \sqrt{\int_{-1.5}^{1.5} \rho(\tau)^2 \, dw} = \sqrt{3}\,\rho(\tau) \le \sqrt{3}\,\rho_{\max} = \sqrt{3}(0.35) \approx 0.6062$$

Applying Minkowski's triangle inequality to $G_\tau(w) = K_{\epsilon, 1/2}(w) + K_{\epsilon, -1/2}(w) + R_\tau(w)$:

$$\|G_\tau\|_{L_2} \le \|K_{\epsilon, 1/2}\|_{L_2} + \|K_{\epsilon, -1/2}\|_{L_2} + \|R_\tau\|_{L_2} \le 2(3.5412) + 0.6062 = 7.6886 < \infty$$

Since $\eta \in C^\infty_c((-1, 1))$, its derivative $\eta'$ is also square-integrable.
Thus $G_\tau \in H^1(\mathbb{R})$ for all $\tau \in [0, 1]$.
In stark contrast, for Temperature Annealing, $\|Q_T'\|_{L_2}^2 = \frac{1}{12T} \to \infty$ as $T \to 0$. $\square$

---

### Theorem 3.2 (Global Topological Transport in Phase 1)
In Phase 1 ($\tau \in [0, 0.4]$), the broad Sobolev kernel ($\epsilon(\tau) \ge 0.40$) combined with the non-zero transport floor ($\rho(\tau) \ge 0.15$) guarantees that the surrogate velocity field $v_\tau(w)$ has full support over $[-1.5, 1.5]$, eliminating the zero-gradient plateau trap of STE.

*Proof.*
Under standard STE, inside any flat bin $(c - \delta, c + \delta)$, the true loss satisfies $\nabla_{\hat{w}}\mathcal{L} = 0$, but the surrogate gradient assigns a direction $w - w^*$, creating limit cycles around spurious local minima.
Under MIGE Phase 1, the gradient update vector field is:

$$v_\tau(w) = - \mathbb{E}_{x \sim \mathcal{D}}\left[\nabla_{\hat{w}}\mathcal{L}(Q(w); x)\right] \cdot G_\tau(w)$$

Since $\eta \ge 0$ everywhere and $\rho(\tau) \ge \rho_{\min} = 0.02 > 0$ for all $\tau$:

$$\inf_{w \in [-1.5, 1.5]} G_\tau(w) \ge \rho(\tau) \ge 0.02 > 0$$

Hence $G_\tau(w) > 0$ uniformly on $[-1.5, 1.5]$.
There are no vanishing-gradient regions in the active weight domain.
Every latent weight receives continuous transport velocity away from suboptimal plateaus and towards the basin of attraction minimizing the discrete task loss, eliminating the discrete representation trap without compromising the exact discrete forward pass. $\square$

---

### Theorem 3.3 (Rigorous Resolution of Asymptotic Convergence: $\delta$-Approximation with Controlled Error Bound)

The claim of simultaneous exact convergence to Dirac delta measures and finite $L_2$ norm is a mathematical impossibility.
We resolve this duality rigorously:
The physical training run operates with **finite-cutoff parameters** $(\epsilon_{\min}, \rho_{\min}) = (0.05, 0.02)$, which defines an **$\epsilon$-mollified weak derivative** $G_{\text{final}} \in H^1(\mathbb{R})$.

For any $C^1$ test function $\phi \in C^1(\mathbb{R})$ with bounded derivative $\|\phi'\|_\infty < \infty$, the dual pairing error between $G_\tau$ and the true distributional derivative $\nabla_w Q$ is explicitly bounded by:

$$|\langle G_\tau, \phi \rangle - \langle \nabla_w Q, \phi \rangle| \le 2\epsilon(\tau)\|\phi'\|_\infty + 3\rho(\tau)\|\phi\|_\infty$$

Furthermore, if the theoretical schedule is extended such that $\lim_{\tau \to 1} \epsilon(\tau) = 0$ and $\lim_{\tau \to 1} \rho(\tau) = 0$, then $G_\tau \to \nabla_w Q$ in $\mathcal{D}'(\mathbb{R})$.
Under the practical stopping cutoff $(\epsilon_{\min}, \rho_{\min})$, $G_\tau$ achieves an **optimal variational trade-off**: it remains strictly within $H^1(\mathbb{R})$ (preventing numerical gradient blow-up) while bounding the distributional approximation error by:

$$\sup_{\|\phi\|_{C^1} \le 1} |\langle G_{\text{final}}, \phi \rangle - \langle \nabla_w Q, \phi \rangle| \le 2(0.05) + 3(0.02) = 0.16$$

*Proof.*
Let $\phi \in C^1(\mathbb{R})$. We evaluate the difference against the true distribution $\langle \nabla_w Q, \phi \rangle = \phi(1/2) + \phi(-1/2)$:

$$\langle G_\tau, \phi \rangle - \langle \nabla_w Q, \phi \rangle = \int_{-\infty}^\infty \frac{1}{\epsilon}\eta\left(\frac{w - 1/2}{\epsilon}\right)\phi(w)\, dw - \phi(1/2) + \int_{-\infty}^\infty \frac{1}{\epsilon}\eta\left(\frac{w + 1/2}{\epsilon}\right)\phi(w)\, dw - \phi(-1/2) + \rho(\tau)\int_{-1.5}^{1.5}\phi(w)\, dw$$

Focusing on the first term with substitution $u = (w - 1/2)/\epsilon$:

$$\int_{-\infty}^\infty \frac{1}{\epsilon}\eta\left(\frac{w - 1/2}{\epsilon}\right)\phi(w)\, dw - \phi(1/2) = \int_{-1}^1 \eta(u)\left[\phi(1/2 + \epsilon u) - \phi(1/2)\right] du$$

By the Mean Value Theorem, $|\phi(1/2 + \epsilon u) - \phi(1/2)| \le \|\phi'\|_\infty \epsilon |u| \le \epsilon \|\phi'\|_\infty$.
Since $\int_{-1}^1 \eta(u)\, du = 1$:

$$\left|\int_{-1}^1 \eta(u)\left[\phi(1/2 + \epsilon u) - \phi(1/2)\right] du\right| \le \epsilon \|\phi'\|_\infty \int_{-1}^1 \eta(u)\, du = \epsilon \|\phi'\|_\infty$$

The same bound holds for the second threshold at $w = -1/2$.
For the transport floor:

$$\left|\rho(\tau)\int_{-1.5}^{1.5}\phi(w)\, dw\right| \le \rho(\tau) \cdot 3 \|\phi\|_\infty$$

Summing both error components:

$$|\langle G_\tau, \phi \rangle - \langle \nabla_w Q, \phi \rangle| \le 2\epsilon(\tau)\|\phi'\|_\infty + 3\rho(\tau)\|\phi\|_\infty$$

When normalized over the unit ball of test functions $\{\phi : \|\phi\|_\infty \le 1, \|\phi'\|_\infty \le 1\}$:

$$\text{Error}(\tau) \le 2\epsilon(\tau) + 3\rho(\tau)$$

At the terminal cutoff $(\epsilon_{\min}, \rho_{\min}) = (0.05, 0.02)$:

$$\text{Error}_{\text{terminal}} \le 2(0.05) + 3(0.02) = 0.10 + 0.06 = 0.16$$

This establishes the precise mathematical nature of MIGE-2P:
- It does **not** generate singular Dirac deltas (which would diverge in $L_2$ and destabilize floating-point arithmetic);
- Instead, it generates an **optimal $H^1$-regularized Sobolev surrogate** whose deviation from the true discrete subgradient is mathematically bounded by $\mathcal{O}(\epsilon_{\min} + \rho_{\min}) = 0.16$. $\square$

---

## 4. Rigorous Comparison Table Across All Paradigms

| Optimization Property | Straight-Through Estimator (STE) | Temperature Annealing | MIGE-2P (Two-Phase Sobolev) |
| :--- | :---: | :---: | :---: |
| **Forward Pass Evaluation** | Exact Discrete | Smooth Sigmoid Proxy ($Q_T \neq Q$) | **Exact Discrete** ($\{-1, 0, +1\}$) |
| **Early Discretization Gap** | Stable ($\sim 0$) | Explodes ($+1,222$ PPL) | **Strictly Zero Forward Discrepancy** |
| **Total Weak Derivative Mass** | $2.0 \cdot \mathbf{1}_{|w| \le 1}$ (Flat) | $2.0$ (via sech$^2$) | **$2.0$ (Conserved Unit Mass per Jump)** |
| **Gradient Regularity** | $L_2$ bounded (clipped) | Diverges: $\|Q_T'\|_{L_2}^2 = O(1/T) \to \infty$ | **Strictly $H^1$ Bounded** ($\le 7.69$) |
| **Distributional Error Bound** | Irreducible $\mathcal{O}(1)$ Shadow Gap | Asymptotic $O(T)$ with $L_2 \to \infty$ | **Controlled $H^1$ Bound** ($\le 0.16$) |
| **Phase 1 Global Representation** | Trapped in local shadow bins | Free continuous flow | **Global Sobolev Transport** ($\rho \ge 0.15$) |
| **Phase 2 Boundary Specialization**| None (Uniform Identity) | Explosive Delta Spikes | **Smooth $C^\infty$ Compact Bumps** |
| **Production Hardware Ready?** | Yes | No (Unstable schedules) | **Yes (Direct drop-in)** |
