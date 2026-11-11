"""
Mollified Integral Gradient Estimation (MIGE) with Adaptive Two-Phase Sobolev Schedule.
Solves the exploration bottleneck in deep BitNet Transformers:
Phase 1 (Global Topological Flow): Broad Sobolev kernel with exploratory transport floor
Phase 2 (Sobolev Boundary Specialization): Adaptive narrowing to sharp weak derivative bumps
"""
import math
import torch
from torch.optim.optimizer import Optimizer

MOLLIFIER_NORMALIZATION_C = 0.44399381616807944

def mollifier_bump(x: torch.Tensor) -> torch.Tensor:
    mask = (x.abs() < 1.0)
    x_clamped = torch.clamp(x, -0.999999, 0.999999)
    bump = torch.where(mask, torch.exp(-1.0 / (1.0 - x_clamped**2)), torch.zeros_like(x))
    return bump / MOLLIFIER_NORMALIZATION_C

def mige_kernel_derivative(w: torch.Tensor, epsilon: float, threshold: float = 0.5,
                           floor_grad: float = 0.25) -> torch.Tensor:
    """
    Two-scale weak derivative:
    Primary Sobolev weak derivative localized at thresholds + exploratory transport channel.
    """
    d_pos = (w - threshold) / epsilon
    d_neg = (w + threshold) / epsilon
    bump_grad = (mollifier_bump(d_pos) + mollifier_bump(d_neg)) / epsilon
    # Scale bump so maximum peak is well-conditioned and combine with transport flow
    return 0.5 * bump_grad + floor_grad

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
