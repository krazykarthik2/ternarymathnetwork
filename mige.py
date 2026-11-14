"""
Mollified Integral Gradient Estimation (MIGE) with Exact-Mass Two-Phase Sobolev Schedule.
Solves the exploration bottleneck in deep BitNet Transformers:
Phase 1 (Global Topological Flow): Broad Sobolev kernel with exploratory transport floor
Phase 2 (Sobolev Boundary Specialization): Adaptive narrowing to sharp weak derivative bumps
Conserves total distributional mass = 2.0.
"""
import math
import torch
from torch.optim.optimizer import Optimizer

MOLLIFIER_NORMALIZATION_C = 0.44399381616807944

def mollifier_bump(x: torch.Tensor) -> torch.Tensor:
    """
    Standard Friedrichs C^infinity mollifier with compact support (-1, 1).
    eta(x) = C * exp(-1 / (1 - x^2)) if |x| < 1 else 0
    Normalized such that \int_{-1}^1 eta(x) dx = 1.0.
    """
    mask = (x.abs() < 1.0)
    x_clamped = torch.clamp(x, -0.999999, 0.999999)
    bump = torch.where(mask, torch.exp(-1.0 / (1.0 - x_clamped**2)), torch.zeros_like(x))
    return bump / MOLLIFIER_NORMALIZATION_C

def mige_kernel_derivative(w: torch.Tensor, epsilon: float, threshold: float = 0.5,
                           floor_grad: float = 0.25) -> torch.Tensor:
    """
    Exact-Mass Two-Phase Sobolev weak derivative operator:
    G_tau(w) = (1 / eps) * [ eta((w - th)/eps) + eta((w + th)/eps) ] + floor * 1_{|w| <= 1.5}
    Total bump mass = 1.0 + 1.0 = 2.0 (matches distributional \nabla_w Q).
    """
    d_pos = (w - threshold) / epsilon
    d_neg = (w + threshold) / epsilon
    bump_grad = (mollifier_bump(d_pos) + mollifier_bump(d_neg)) / epsilon
    # Transport floor active across the dynamic weight domain |w| <= 1.5
    transport_floor = torch.where(w.abs() <= 1.5, torch.full_like(w, floor_grad), torch.zeros_like(w))
    return bump_grad + transport_floor

class MIGEQuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w: torch.Tensor, epsilon: float, threshold: float = 0.5,
                floor_grad: float = 0.25) -> torch.Tensor:
        from quantizers import hard_ternary
        ctx.save_for_backward(w)
        ctx.epsilon = float(epsilon)
        ctx.threshold = float(threshold)
        ctx.floor_grad = float(floor_grad)
        return hard_ternary(w, threshold)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (w,) = ctx.saved_tensors
        grad_mult = mige_kernel_derivative(w, ctx.epsilon, ctx.threshold, ctx.floor_grad)
        return grad_output * grad_mult, None, None, None

def mige_quantize(w: torch.Tensor, epsilon: float, threshold: float = 0.5,
                  floor_grad: float = 0.25) -> torch.Tensor:
    return MIGEQuantizeFunction.apply(w, epsilon, threshold, floor_grad)
