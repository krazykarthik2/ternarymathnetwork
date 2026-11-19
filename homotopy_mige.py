"""
Homotopy Continuation / Curriculum Sobolev Quantization (MIGE-HC):
Phase 1 (Steps 1 to switch_step):
  - Forward: Smooth homotopy relaxation Q_T(w) with zero gradient mismatch
  - Backward: Exact analytical derivative Q'_T(w)
  - Result: Smooth loss optimization to learn core language representation
Phase 2 (Steps switch_step to end):
  - Forward: HARD discrete ternary projection {-1, 0, +1}
  - Backward: MIGE-1 Sobolev sech^2 surrogate gradient
  - Result: Zero discretization shock; model settles on discrete lattice under AdamW
"""
import torch
import torch.nn as nn
from quantizers import hard_ternary, soft_ternary, soft_ternary_derivative
from mige import fused_mige1_kernel

class HomotopySobolevQuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w: torch.Tensor, T: float, epsilon: float, is_discrete: bool,
                threshold: float = 0.5, floor_grad: float = 0.05) -> torch.Tensor:
        ctx.save_for_backward(w)
        ctx.is_discrete = bool(is_discrete)
        ctx.T = float(T)
        ctx.inv_eps = float(1.0 / max(epsilon, 1e-4))
        ctx.threshold = float(threshold)
        ctx.floor_grad = float(floor_grad)

        if is_discrete:
            return hard_ternary(w, threshold)
        else:
            return soft_ternary(w, T, threshold)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (w,) = ctx.saved_tensors
        if ctx.is_discrete:
            grad_mult = fused_mige1_kernel(w, ctx.inv_eps, ctx.threshold, ctx.floor_grad)
        else:
            grad_mult = soft_ternary_derivative(w, ctx.T, ctx.threshold)
        return grad_output * grad_mult, None, None, None, None, None

def homotopy_sobolev_quantize(w: torch.Tensor, T: float, epsilon: float, is_discrete: bool,
                              threshold: float = 0.5, floor_grad: float = 0.05) -> torch.Tensor:
    return HomotopySobolevQuantizeFunction.apply(w, T, epsilon, is_discrete, threshold, floor_grad)
