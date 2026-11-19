"""
Mollified Temperature Annealing (MIGE-TA) Quantizer:
Combines:
1. Continuous Temperature Annealed Sigmoid Forward Pass:
   Q_T(w) = sigma((w-th)/T) + sigma((w+th)/T) - 1.0 (Zero surrogate mismatch)
2. Fused Analytical sech^2 Jacobian
3. MIGE-1 Non-Vanishing Sobolev Floor:
   G_T(w) = Q'_T(w) + floor_grad
   Prevents deep-basin weights from freezing when sigma' decays exponentially.
"""
import torch
import torch.nn as nn
from quantizers import soft_ternary

@torch.compile
def fused_mige_ta_backward(w: torch.Tensor, inv_T: float, threshold: float, floor_grad: float) -> torch.Tensor:
    """
    Fused analytical single-pass CUDA kernel for MIGE-TA.
    Computes exact derivative: (1/T) * [sigma'(dp) + sigma'(dn)] + floor_grad
    Using sech^2 identity: sigma'(u) = 0.25 * sech^2(u/2)
    """
    u_p = (w - threshold) * (0.5 * inv_T)
    u_n = (w + threshold) * (0.5 * inv_T)
    
    sech2_p = torch.cosh(u_p).reciprocal().square()
    sech2_n = torch.cosh(u_n).reciprocal().square()
    
    analytical_deriv = (0.25 * inv_T) * (sech2_p + sech2_n)
    floor_mask = torch.where(w.abs() <= 1.5, floor_grad, 0.0)
    return analytical_deriv + floor_mask

class MIGETAQuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w: torch.Tensor, T: float, threshold: float = 0.5,
                floor_grad: float = 0.05) -> torch.Tensor:
        ctx.save_for_backward(w)
        ctx.inv_T = float(1.0 / max(T, 1e-4))
        ctx.threshold = float(threshold)
        ctx.floor_grad = float(floor_grad)
        return soft_ternary(w, T, threshold)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (w,) = ctx.saved_tensors
        if torch.cuda.is_available():
            grad_mult = fused_mige_ta_backward(w, ctx.inv_T, ctx.threshold, ctx.floor_grad)
        else:
            # CPU fallback without torch.compile dependency
            u_p = (w - ctx.threshold) * (0.5 * ctx.inv_T)
            u_n = (w + ctx.threshold) * (0.5 * ctx.inv_T)
            sech2_p = torch.cosh(u_p).reciprocal().square()
            sech2_n = torch.cosh(u_n).reciprocal().square()
            analytical_deriv = (0.25 * ctx.inv_T) * (sech2_p + sech2_n)
            floor_mask = torch.where(w.abs() <= 1.5, ctx.floor_grad, 0.0)
            grad_mult = analytical_deriv + floor_mask

        return grad_output * grad_mult, None, None, None

def mige_ta_quantize(w: torch.Tensor, T: float, threshold: float = 0.5,
                     floor_grad: float = 0.05) -> torch.Tensor:
    return MIGETAQuantizeFunction.apply(w, T, threshold, floor_grad)
