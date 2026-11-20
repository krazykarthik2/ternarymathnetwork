"""
hopfield/train_hopfield_torch_maxspeed.py

Maximum Throughput PyTorch Optimization Engine for L40S:
1. torch.compile(mode="reduce-overhead") with Triton kernel fusion.
2. F.scaled_dot_product_attention (FlashAttention-2 / cuDNN fused).
3. Multiprocess PyTorch DataLoader with pin_memory=True, persistent_workers=True, num_workers=4.
4. Non-blocking asynchronous host-to-device transfers (x.to(device, non_blocking=True)).
5. Native TF32 tensor core acceleration on Ampere/Ada Lovelace.
6. torch.amp.autocast with native bfloat16.
7. Fused AdamW optimizer (torch.optim.AdamW(fused=True)).
"""
import os
import sys
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Enable maximum TF32 & cuDNN benchmarks for L40S Ada Lovelace architecture
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hopfield_bitnet import HopfieldBitNet125M, HopfieldBitAttention, HopfieldBitFeedForward, RMSNorm
from quantizers import temp_anneal_quantize


class FastFlashAttention(nn.Module):
    """Replaces manual attention matrix multiplies with PyTorch's FlashAttention-2 kernel."""
    def __init__(self, dim: int, n_heads: int, method: str = "temp_anneal"):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        from hopfield_bitnet import HopfieldBitLinear
        self.q_proj = HopfieldBitLinear(dim, dim, bias=False, method=method)
        self.k_proj = HopfieldBitLinear(dim, dim, bias=False, method=method)
        self.v_proj = HopfieldBitLinear(dim, dim, bias=False, method=method)
        self.out_proj = HopfieldBitLinear(dim, dim, bias=False, method=method)

    def forward(self, x: torch.Tensor, mask=None, T: float = 1.0, epsilon: float = 0.25) -> torch.Tensor:
        B, S, D = x.shape
        q = self.q_proj(x, T=T, epsilon=epsilon).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x, T=T, epsilon=epsilon).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x, T=T, epsilon=epsilon).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Fused FlashAttention-2 / Memory-Efficient Attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out, T=T, epsilon=epsilon)


class FastStreamingDataset(IterableDataset):
    """High-throughput multi-worker streaming dataset for PyTorch DataLoader."""
    def __init__(self, dataset_name, subset, seq_len=512):
        self.dataset_name = dataset_name
        self.subset = subset
        self.seq_len = seq_len

    def __iter__(self):
        from datasets import load_dataset
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        worker_info = torch.utils.data.get_worker_info()

        if self.subset:
            ds = load_dataset(self.dataset_name, name=self.subset, split="train", streaming=True)
        else:
            ds = load_dataset(self.dataset_name, split="train", streaming=True)

        if worker_info is not None:
            # Shard across CPU worker cores
            ds = ds.shard(num_shards=worker_info.num_workers, index=worker_info.id)

        token_buf = []
        for item in ds:
            text = item.get("text", "")
            if not text:
                continue
            token_buf.extend(tokenizer.encode(text))
            while len(token_buf) >= self.seq_len + 1:
                chunk = token_buf[:self.seq_len + 1]
                token_buf = token_buf[self.seq_len:]
                yield (torch.tensor(chunk[:-1], dtype=torch.long),
                       torch.tensor(chunk[1:], dtype=torch.long))


def main():
    parser = argparse.ArgumentParser(description="PyTorch MaxSpeed Hopfield-TA Engine")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4, help="CPU worker processes for data preloading")
    parser.add_argument("--compile", action="store_true", default=True, help="Use torch.compile")
    parser.add_argument("--output_dir", type=str, default="checkpoints_maxspeed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== PyTorch MaxSpeed TA Engine on: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) ===")

    # Initialize model
    model = HopfieldBitNet125M(dim=768, n_layers=12, n_heads=12, hidden_dim=2048, method="temp_anneal").to(device)

    # Replace standard attention with Fused FlashAttention
    for layer in model.layers:
        layer.attn = FastFlashAttention(dim=768, n_heads=12, method="temp_anneal").to(device)

    # 1. Analytical Hopfield Single-Shot Calculation
    print("\n[Step 1] Fast Single-Shot Hopfield Weight Calculation...")
    t0 = time.time()
    sample_tokens = torch.randint(0, 50257, (16, args.seq_len), device=device)
    model.calculate_all_weights_single_shot(sample_tokens)
    print(f"--> Calculated weights in {(time.time() - t0)*1000:.2f} ms")

    # Optional torch.compile for graph fusion
    if args.compile and torch.cuda.is_available() and hasattr(torch, "compile"):
        # We compile the FlashAttention submodule which holds all matmuls and SDPA
        print("--> Compiling FastFlashAttention layers with torch.compile...")
        for i, layer in enumerate(model.layers):
            try:
                layer.attn = torch.compile(layer.attn)
            except Exception as e:
                print(f"--> Layer {i} compile fallback: {e}")
        print("--> Attention compilation ready!")

    # Multi-worker asynchronous PyTorch DataLoader
    dataset = FastStreamingDataset("HuggingFaceFW/fineweb-edu", "sample-10BT", seq_len=args.seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.workers > 0 and torch.cuda.is_available()),
        prefetch_factor=4 if (args.workers > 0 and torch.cuda.is_available()) else None
    )

    # Fused AdamW optimizer (runs in native C++/CUDA kernel)
    supports_fused = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=(supports_fused and torch.cuda.is_available())
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    criterion = nn.CrossEntropyLoss()

    print(f"\n[Step 2] Executing optimized TA training for {args.steps} steps...")
    start_time = time.time()
    data_iter = iter(loader)

    for step in range(1, args.steps + 1):
        try:
            bx, by = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            bx, by = next(data_iter)

        # Async non-blocking DMA transfer to GPU
        x = bx.to(device, non_blocking=True)
        y = by.to(device, non_blocking=True)

        tau = step / float(args.steps)
        T_val = 0.01 + 0.5 * (1.0 - 0.01) * (1.0 + math.cos(math.pi * tau))

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=dtype):
            logits = model(x, T=T_val)
            loss = criterion(logits.view(-1, 50257), y.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        if step % 25 == 0 or step == 1 or step == args.steps:
            tok_speed = (step * args.batch_size * args.seq_len) / max(time.time() - start_time, 1e-5)
            ppl = math.exp(min(loss.item(), 20.0))
            print(f"Step {step:04d}/{args.steps:04d} | Loss: {loss.item():.4f} | PPL: {ppl:.2f} | T: {T_val:.3f} | Throughput: {tok_speed:.0f} tok/s")

    # Save final model
    final_ckpt = os.path.join(args.output_dir, "pure_ta_maxspeed.pt")
    torch.save({
        "step": args.steps,
        "model_state_dict": model.state_dict() if not hasattr(model, "_orig_mod") else model._orig_mod.state_dict(),
        "loss": loss.item()
    }, final_ckpt)
    print(f"\n--> Model saved to: {final_ckpt}")
    print(f"All {args.steps} steps completed in {(time.time() - start_time)/60:.2f} minutes!")


if __name__ == "__main__":
    main()
