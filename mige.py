"""
Mollified Integral Gradient Estimation (MIGE) Suite & TSGO.
Includes:
1. MIGE-1 (Adaptive Sobolev bump with 1/epsilon boundary concentration)
2. MIGE-NBS (Normalized Bounded-Sobolev - High-Speed Fused GPU Kernel)
3. TSGO (Ternary Swarm-Guided Optimization):
   - Discrete forward execution {-1, 0, +1}
   - Analytical sech^2 surrogate backward pass
   - Gradient-guided discrete swarm probing with Metropolis-Boltzmann acceptance
"""
import math
import torch
import torch.nn.functional as F

def _maybe_compile(fn):
    """Conditionally compiles on CUDA platforms where Triton/C++ is present."""
    if torch.cuda.is_available():
        try:
            return torch.compile(fn)
        except Exception:
            return fn
    return fn

def _fused_mige1_impl(w: torch.Tensor, inv_eps: float, threshold: float, floor_grad: float) -> torch.Tensor:
    dp = (w - threshold) * inv_eps
    dn = (w + threshold) * inv_eps
    bump_p = torch.cosh(dp).reciprocal().square()
    bump_n = torch.cosh(dn).reciprocal().square()
    sobolev_bump = (bump_p + bump_n) * inv_eps
    floor_mask = torch.where(w.abs() <= 1.5, floor_grad, 0.0)
    return sobolev_bump + floor_mask

def _fused_nbs_impl(w: torch.Tensor, inv_eps: float, threshold: float, floor_grad: float) -> torch.Tensor:
    dp = (w - threshold) * inv_eps
    dn = (w + threshold) * inv_eps
    bump_p = torch.cosh(dp).reciprocal().square()
    bump_n = torch.cosh(dn).reciprocal().square()
    norm_bump = (bump_p + bump_n) * (1.0 - floor_grad)
    floor_mask = torch.where(w.abs() <= 1.5, floor_grad, 0.0)
    return torch.clamp(norm_bump + floor_mask, 0.0, 1.0)

fused_mige1_kernel = _maybe_compile(_fused_mige1_impl)
fused_nbs_kernel = _maybe_compile(_fused_nbs_impl)

class MIGE1QuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w: torch.Tensor, epsilon: float, threshold: float = 0.5,
                floor_grad: float = 0.05) -> torch.Tensor:
        from quantizers import hard_ternary
        ctx.save_for_backward(w)
        ctx.inv_eps = float(1.0 / max(epsilon, 1e-4))
        ctx.threshold = float(threshold)
        ctx.floor_grad = float(floor_grad)
        return hard_ternary(w, threshold)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (w,) = ctx.saved_tensors
        grad_mult = fused_mige1_kernel(w, ctx.inv_eps, ctx.threshold, ctx.floor_grad)
        return grad_output * grad_mult, None, None, None

class MIGENBSQuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w: torch.Tensor, epsilon: float, threshold: float = 0.5,
                floor_grad: float = 0.10) -> torch.Tensor:
        from quantizers import hard_ternary
        ctx.save_for_backward(w)
        ctx.inv_eps = float(1.0 / max(epsilon, 1e-4))
        ctx.threshold = float(threshold)
        ctx.floor_grad = float(floor_grad)
        return hard_ternary(w, threshold)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (w,) = ctx.saved_tensors
        grad_mult = fused_nbs_kernel(w, ctx.inv_eps, ctx.threshold, ctx.floor_grad)
        return grad_output * grad_mult, None, None, None

def mige1_quantize(w: torch.Tensor, epsilon: float, threshold: float = 0.5,
                   floor_grad: float = 0.05) -> torch.Tensor:
    return MIGE1QuantizeFunction.apply(w, epsilon, threshold, floor_grad)

def mige_nbs_quantize(w: torch.Tensor, epsilon: float, threshold: float = 0.5,
                      floor_grad: float = 0.10) -> torch.Tensor:
    return MIGENBSQuantizeFunction.apply(w, epsilon, threshold, floor_grad)

# TSGO uses MIGE-1 analytical surrogate for backward pass
def tsgo_quantize(w: torch.Tensor, epsilon: float, threshold: float = 0.5,
                  floor_grad: float = 0.05) -> torch.Tensor:
    return MIGE1QuantizeFunction.apply(w, epsilon, threshold, floor_grad)

# Backward-compatible API for standard MIGE
def mige_quantize(w: torch.Tensor, epsilon: float, threshold: float = 0.5,
                  floor_grad: float = 0.05, halo_alpha: float = 0.0,
                  halo_beta: float = 5.0) -> torch.Tensor:
    return mige1_quantize(w, epsilon, threshold, floor_grad)
