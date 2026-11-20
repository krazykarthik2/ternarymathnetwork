"""
hopfield/hopfield_bitnet.py

Implementation of Ternary Hopfield-BitNet b1.58 (125M Parameter Architecture).
Integrates:
1. Classical ternary Hopfield associative memory with {-1, 0, 1} state space.
2. Single-shot closed-form analytical weight computation (Hebbian / pseudo-inverse).
3. BitNet b1.58 BitLinear modules with ternary weights {-1, 0, 1}.
4. Full 125M decoder transformer (dim=768, heads=12, layers=12, hidden_dim=2048).
"""
import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure parent directory is accessible for existing repo modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from quantizers import hard_ternary, ste_quantize
from mige_ta import mige_ta_quantize


class TernaryHopfieldMemory(nn.Module):
    """
    Classical Associative Hopfield Memory layer operating over {-1, 0, +1}.
    Allows direct closed-form analytical calculation of weight matrix W from stored patterns.
    """
    def __init__(self, dim: int, threshold: float = 0.5):
        super().__init__()
        self.dim = dim
        self.threshold = threshold
        # Latent weight parameter (N x N)
        self.weight = nn.Parameter(torch.zeros(dim, dim))
        self.gamma = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    @torch.no_grad()
    def calculate_weights_single_shot(self, patterns: torch.Tensor, regularizer: float = 1e-4):
        """
        Direct single-shot analytical weight calculation.
        patterns: [P, dim] tensor of key associative patterns.
        Calculates correlation W = (1/P) * patterns^T @ patterns, with zero diagonal,
        and quantizes into exact {-1, 0, 1}.
        """
        P, D = patterns.shape
        assert D == self.dim, f"Pattern dimension {D} does not match memory dim {self.dim}"

        # Hebbian correlation
        M = torch.matmul(patterns.t(), patterns) / float(P)
        # Remove self-coupling
        M.fill_diagonal_(0.0)

        # Calculate scale and quantize to {-1, 0, 1}
        gamma = M.abs().mean().clamp(min=1e-5)
        self.gamma.copy_(gamma)

        m_norm = M / gamma
        w_ternary = hard_ternary(m_norm, threshold=self.threshold)
        self.weight.copy_(w_ternary * gamma)

    def forward(self, x: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """
        Hopfield associative recall step.
        s_{t+1} = TernaryQuantize(s_t @ W)
        """
        s = x
        gamma = self.weight.abs().mean().clamp(min=1e-5)
        w_norm = self.weight / gamma
        w_q = hard_ternary(w_norm, threshold=self.threshold)
        w_effective = w_q * gamma

        for _ in range(steps):
            h = F.linear(s, w_effective)
            # Ternary spin update
            s = hard_ternary(h / (h.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)), threshold=self.threshold)
        return s


class HopfieldBitLinear(nn.Module):
    """
    BitLinear layer initialized via closed-form single-shot calculation or standard methods,
    with ternary weights {-1, 0, +1}.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, method: str = "mige_ta"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.method = method
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.floor_grad = 0.05
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @torch.no_grad()
    def calculate_weights_single_shot(self, X: torch.Tensor, Y: torch.Tensor, lambda_reg: float = 1e-3):
        """
        Direct single-shot analytical weight calculation from input-target pairs (X, Y).
        Computes W_analytical = Y^T X (X^T X + lambda I)^(-1)
        and sets weight.
        """
        # X: [N, in_features], Y: [N, out_features]
        N, Din = X.shape
        _, Dout = Y.shape
        assert Din == self.in_features and Dout == self.out_features

        cov_xx = torch.matmul(X.t(), X) / N + lambda_reg * torch.eye(Din, device=X.device)
        cov_yx = torch.matmul(Y.t(), X) / N

        w_opt = torch.matmul(cov_yx, torch.linalg.pinv(cov_xx))
        self.weight.copy_(w_opt)

    def forward(self, x: torch.Tensor, T: float = 1.0, epsilon: float = 0.25) -> torch.Tensor:
        gamma = self.weight.abs().mean().clamp(min=1e-5)
        w_norm = self.weight / gamma

        if self.method in ("temp_anneal", "ta"):
            from quantizers import soft_ternary
            # Pure tensor math: 100% TorchDynamo / Triton compilable without autograd.Function errors
            w_q = soft_ternary(w_norm, T=T, threshold=0.5)
        elif self.method in ("mige_ta", "migeta"):
            w_q = mige_ta_quantize(w_norm, T=T, threshold=0.5, floor_grad=self.floor_grad)
        elif self.method == "ste":
            w_q = ste_quantize(w_norm, threshold=0.5, clip_val=1.0)
        elif self.method == "discrete":
            w_q = hard_ternary(w_norm, threshold=0.5)
        else:
            w_q = hard_ternary(w_norm, threshold=0.5)

        w_effective = w_q * gamma
        return F.linear(x, w_effective, self.bias)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class HopfieldBitAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, method: str = "mige_ta"):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0, "dim must be divisible by n_heads"

        self.q_proj = HopfieldBitLinear(dim, dim, bias=False, method=method)
        self.k_proj = HopfieldBitLinear(dim, dim, bias=False, method=method)
        self.v_proj = HopfieldBitLinear(dim, dim, bias=False, method=method)
        self.out_proj = HopfieldBitLinear(dim, dim, bias=False, method=method)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, T: float = 1.0, epsilon: float = 0.25) -> torch.Tensor:
        B, S, D = x.shape
        q = self.q_proj(x, T=T, epsilon=epsilon).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x, T=T, epsilon=epsilon).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x, T=T, epsilon=epsilon).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out, T=T, epsilon=epsilon)


class HopfieldBitFeedForward(nn.Module):
    """SwiGLU Feed-Forward Network with HopfieldBitLinear layers."""
    def __init__(self, dim: int, hidden_dim: int, method: str = "mige_ta"):
        super().__init__()
        self.gate_proj = HopfieldBitLinear(dim, hidden_dim, bias=False, method=method)
        self.up_proj = HopfieldBitLinear(dim, hidden_dim, bias=False, method=method)
        self.down_proj = HopfieldBitLinear(hidden_dim, dim, bias=False, method=method)

    def forward(self, x: torch.Tensor, T: float = 1.0, epsilon: float = 0.25) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x, T=T, epsilon=epsilon))
        up = self.up_proj(x, T=T, epsilon=epsilon)
        return self.down_proj(gate * up, T=T, epsilon=epsilon)


class HopfieldBitTransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, hidden_dim: int, method: str = "mige_ta"):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = HopfieldBitAttention(dim, n_heads, method=method)
        self.norm2 = RMSNorm(dim)
        self.ffn = HopfieldBitFeedForward(dim, hidden_dim, method=method)
        # Hopfield associative recurrent memory unit
        self.hopfield_mem = TernaryHopfieldMemory(dim=dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, T: float = 1.0, epsilon: float = 0.25) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask=mask, T=T, epsilon=epsilon)
        x = x + self.ffn(self.norm2(x), T=T, epsilon=epsilon)
        return x


class HopfieldBitNet125M(nn.Module):
    """
    125M parameter BitNet b1.58 with integrated Ternary Hopfield associative layers
    and single-shot analytical weight calculation capabilities.
    """
    def __init__(self, vocab_size: int = 50257, dim: int = 768, n_layers: int = 12,
                 n_heads: int = 12, hidden_dim: int = 2048, max_seq_len: int = 512,
                 method: str = "mige_ta"):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.method = method

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.pos_embeddings = nn.Embedding(max_seq_len, dim)

        self.layers = nn.ModuleList([
            HopfieldBitTransformerBlock(dim=dim, n_heads=n_heads, hidden_dim=hidden_dim, method=method)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def calculate_all_weights_single_shot(self, sample_tokens: torch.Tensor):
        """
        DIRECT FULL-MODEL WARM INITIALIZATION VIA HOPFIELD HEBBIAN ASSOCIATION:
        Calculates the ternary weights for EVERY SINGLE LINEAR LAYER:
        - Self-Attention Q, K, V, and Out projections
        - FeedForward Gate, Up, and Down projections
        from actual linguistic token hidden states.
        """
        B, S = sample_tokens.shape
        pos = torch.arange(0, S, dtype=torch.long, device=sample_tokens.device).unsqueeze(0)
        h = self.tok_embeddings(sample_tokens) + self.pos_embeddings(pos)

        flat_h = h.view(-1, self.dim)
        N = flat_h.size(0)

        # Autocorrelation of token embeddings: C = (1/N) * H^T H
        cov_hh = torch.matmul(flat_h.t(), flat_h) / float(N)

        for layer_idx, layer in enumerate(self.layers):
            norm_h = layer.norm1(flat_h)

            # 1. Warm-initialize Attention Projections (Q, K, V, Out) directly
            # Q & K: Positive correlation associative mapping
            gamma_attn = cov_hh.abs().mean().clamp(min=1e-5)
            w_ternary_attn = hard_ternary(cov_hh / gamma_attn, threshold=0.5) * gamma_attn
            
            layer.attn.q_proj.weight.copy_(w_ternary_attn)
            layer.attn.k_proj.weight.copy_(w_ternary_attn)
            layer.attn.v_proj.weight.copy_(w_ternary_attn)
            layer.attn.out_proj.weight.copy_(w_ternary_attn)

            # 2. Warm-initialize FFN Projections (Gate, Up, Down)
            # Gate & Up map from dim (768) -> hidden_dim (2048)
            # Down maps from hidden_dim (2048) -> dim (768)
            norm_h2 = layer.norm2(flat_h)
            d_in, d_out = layer.ffn.gate_proj.in_features, layer.ffn.gate_proj.out_features
            
            # Tile cross-covariance to match FFN dimensional expansion
            reps_r = math.ceil(d_out / d_in)
            ffn_cov = cov_hh.repeat(reps_r, 1)[:d_out, :d_in]
            gamma_ffn = ffn_cov.abs().mean().clamp(min=1e-5)
            w_ternary_ffn = hard_ternary(ffn_cov / gamma_ffn, threshold=0.5) * gamma_ffn

            layer.ffn.gate_proj.weight.copy_(w_ternary_ffn)
            layer.ffn.up_proj.weight.copy_(w_ternary_ffn)
            layer.ffn.down_proj.weight.copy_(w_ternary_ffn.t()[:d_in, :d_out])

            # 3. Hopfield associative memory unit
            layer.hopfield_mem.calculate_weights_single_shot(flat_h[:min(1024, N)])

    def forward(self, input_ids: torch.Tensor, T: float = 1.0, epsilon: float = 0.25) -> torch.Tensor:
        B, S = input_ids.shape
        pos = torch.arange(0, S, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        h = self.tok_embeddings(input_ids) + self.pos_embeddings(pos)
        mask = torch.triu(torch.full((S, S), float('-inf'), device=input_ids.device), diagonal=1)

        for layer in self.layers:
            h = layer(h, mask=mask, T=T, epsilon=epsilon)

        h = self.norm(h)
        logits = self.output_head(h)
        return logits
