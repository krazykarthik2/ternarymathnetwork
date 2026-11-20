"""
Quantization operators and Autograd functions for ternary neural networks:
1. Hard Ternary Quantization: Q(w) in {-1, 0, +1}
2. Straight-Through Estimator (STE): forward=Q(w), backward=clipped identity
3. Temperature Annealed Soft Quantization: Q_T(w) with exact analytical / autograd derivatives
"""
import torch
import torch.nn as nn

def hard_ternary(w: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Exact ternary quantization {-1, 0, +1}."""
    return torch.where(w > threshold, torch.ones_like(w),
           torch.where(w < -threshold, -torch.ones_like(w), torch.zeros_like(w)))

class STEQuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w: torch.Tensor, threshold: float = 0.5, clip_val: float = 1.0) -> torch.Tensor:
        ctx.save_for_backward(w)
        ctx.clip_val = clip_val
        return hard_ternary(w, threshold)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (w,) = ctx.saved_tensors
        # Clipped identity: derivative is 1 if |w| <= clip_val, else 0
        mask = (w.abs() <= ctx.clip_val).to(grad_output.dtype)
        return grad_output * mask, None, None

def ste_quantize(w: torch.Tensor, threshold: float = 0.5, clip_val: float = 1.0) -> torch.Tensor:
    return STEQuantizeFunction.apply(w, threshold, clip_val)

def soft_ternary(w: torch.Tensor, T: float, threshold: float = 0.5) -> torch.Tensor:
    """
    Temperature-parameterized smooth ternary quantizer:
    Q_T(w) = sigma((w - threshold)/T) + sigma((w + threshold)/T) - 1.0
    100% TorchScript & TorchDynamo compilable (pure tensor operations).
    """
    s_pos = torch.sigmoid((w - threshold) / T)
    s_neg = torch.sigmoid((w + threshold) / T)
    return s_pos + s_neg - 1.0

def soft_ternary_derivative(w: torch.Tensor, T: float, threshold: float = 0.5) -> torch.Tensor:
    """
    Exact analytical derivative of soft_ternary:
    Q'_T(w) = 1/T * [ sigma((w-th)/T)(1-sigma((w-th)/T)) + sigma((w+th)/T)(1-sigma((w+th)/T)) ]
    """
    s_pos = torch.sigmoid((w - threshold) / T)
    s_neg = torch.sigmoid((w + threshold) / T)
    d_pos = (1.0 / T) * s_pos * (1.0 - s_pos)
    d_neg = (1.0 / T) * s_neg * (1.0 - s_neg)
    return d_pos + d_neg

class STEAnnealQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w: torch.Tensor, T: float, threshold: float = 0.5) -> torch.Tensor:
        ctx.save_for_backward(w)
        # Forward: Smooth continuous relaxation that snaps to exact ternary as T -> 0
        s_pos = torch.sigmoid((w - threshold) / T)
        s_neg = torch.sigmoid((w + threshold) / T)
        return s_pos + s_neg - 1.0

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (w,) = ctx.saved_tensors
        # Backward: Straight-Through gradient flow with floor!
        # Prevents sigmoid derivative from dying at 1e-12 when T cools down!
        mask = (w.abs() <= 1.2).to(grad_output.dtype)
        return grad_output * mask, None, None

def soft_ternary_ste(w: torch.Tensor, T: float, threshold: float = 0.5) -> torch.Tensor:
    return STEAnnealQuantize.apply(w, T, threshold)

def temp_anneal_quantize(w: torch.Tensor, T: float, threshold: float = 0.5) -> torch.Tensor:
    return TempAnnealQuantizeFunction.apply(w, T, threshold)
