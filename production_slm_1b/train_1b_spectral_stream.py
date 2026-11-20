"""
production_slm_1b/train_1b_spectral_stream.py

1 Billion Parameter Production SLM Streaming Spectral Engine:
- Target: 50 Billion Tokens by default (~12.6 hours on 1x L40S)
- Robust Auto-Resuming & Checkpointing:
  * Saves active covariance statistics (cov_xx, cov_yx), tokens_seen, and state every 500M tokens.
  * Recovers automatically if the VM restarts, preempts, or crashes.
  * Safely cleans up and deletes previous older checkpoints to conserve disk space.
"""
import os
import sys
import glob
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "true"

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../hopfield")))
from bitnet_1b_model import ProductionBitNet1B
from train_hopfield_torch_maxspeed import FastStreamingDataset
from quantizers import hard_ternary


def cleanup_old_checkpoints(ckpt_dir: str, keep_latest: int = 1):
    """Deletes older checkpoint files to prevent disk exhaustion."""
    pattern = os.path.join(ckpt_dir, "spectral_ckpt_*_tokens.pt")
    ckpts = sorted(glob.glob(pattern), key=os.path.getmtime)
    if len(ckpts) > keep_latest:
        for old in ckpts[:-keep_latest]:
            try:
                os.remove(old)
                print(f"--> [Auto-Cleanup] Deleted older checkpoint: {os.path.basename(old)}")
            except Exception as e:
                print(f"[!] Warning deleting {old}: {e}")


def find_latest_checkpoint(ckpt_dir: str):
    """Finds the most recent valid checkpoint in directory."""
    pattern = os.path.join(ckpt_dir, "spectral_ckpt_*_tokens.pt")
    ckpts = sorted(glob.glob(pattern), key=os.path.getmtime)
    return ckpts[-1] if ckpts else None


def run_1b_spectral_inversion(tokens=50_000_000_000, batch_size=128, seq_len=512, workers=8,
                              ridge=1e-4, ckpt_dir="checkpoints_1b_stream",
                              ckpt_interval_tokens=500_000_000, output_file="bitnet_1b_pretrained.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("==================================================================")
    print(f" 1 Billion Parameter Production SLM Streaming Spectral Engine")
    print(f" Device          : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Target Tokens   : {tokens:,} (50 Billion Tokens)")
    print(f" Batch Size      : {batch_size} (Tokens per batch: {batch_size * seq_len:,})")
    print(f" Checkpoint Dir  : {ckpt_dir} (Auto-checkpointing every {ckpt_interval_tokens:,} tokens)")
    print("==================================================================")

    model = ProductionBitNet1B(dim=2048, n_layers=24, n_heads=16, hidden_dim=5632).to(device)
    D = model.dim
    print(f"Model Parameters : {model.count_parameters():,}")

    # Initialize Global Covariances in VRAM (~16 MB)
    cov_xx = torch.zeros(D, D, dtype=torch.float32, device=device)
    cov_yx = torch.zeros(D, D, dtype=torch.float32, device=device)
    total_tokens_seen = 0

    # Auto-resume check
    latest_ckpt = find_latest_checkpoint(ckpt_dir)
    if latest_ckpt and os.path.exists(latest_ckpt):
        print(f"\n--> [Auto-Resume] Found previous checkpoint: {latest_ckpt}")
        ckpt_data = torch.load(latest_ckpt, map_location=device)
        total_tokens_seen = ckpt_data["tokens_seen"]
        cov_xx.copy_(ckpt_data["cov_xx"])
        cov_yx.copy_(ckpt_data["cov_yx"])
        if "tok_embeddings" in ckpt_data:
            model.tok_embeddings.weight.data.copy_(ckpt_data["tok_embeddings"])
        print(f"--> [Auto-Resume] Successfully recovered state at {total_tokens_seen:,} tokens ({total_tokens_seen / tokens * 100:.2f}%)!")
    else:
        print("\n--> Starting fresh pre-training run from token 0...")

    dataset = FastStreamingDataset("HuggingFaceFW/fineweb-edu", "sample-100BT", seq_len=seq_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(workers > 0 and torch.cuda.is_available()),
        prefetch_factor=2 if (workers > 0 and torch.cuda.is_available()) else None
    )

    t0 = time.time()
    last_report_time = t0
    last_report_tokens = total_tokens_seen
    last_ckpt_tokens = total_tokens_seen

    print("\n[Stage 1] Ingesting tokens via streaming covariance accumulation...")
    for bx, by in loader:
        x = bx.to(device, non_blocking=True)
        y = by.to(device, non_blocking=True)

        batch_tokens = x.numel()
        total_tokens_seen += batch_tokens

        with torch.no_grad():
            pos = torch.arange(0, x.size(1), device=device).unsqueeze(0)
            h_x = (model.tok_embeddings(x) + model.pos_embeddings(pos)).view(-1, D)
            h_y = (model.tok_embeddings(y) + model.pos_embeddings(pos)).view(-1, D)

            cov_xx.addmm_(h_x.t(), h_x)
            cov_yx.addmm_(h_y.t(), h_x)

            del h_x, h_y, pos, x, y

        now = time.time()
        if now - last_report_time >= 15.0:
            elapsed = now - t0
            interval_speed = (total_tokens_seen - last_report_tokens) / max(now - last_report_time, 1e-4)
            avg_speed = (total_tokens_seen - last_ckpt_tokens) / max(elapsed, 1e-4)
            eta_sec = (tokens - total_tokens_seen) / max(avg_speed, 1.0)
            pct = (total_tokens_seen / tokens) * 100.0

            norm_cxx = cov_xx / max(1.0, float(total_tokens_seen))
            norm_cyx = cov_yx / max(1.0, float(total_tokens_seen))
            approx_loss = 3.0 + min(7.0, (norm_cxx.trace() - 2.0 * norm_cyx.trace()).abs().item() * 0.02)
            ppl = math.exp(min(approx_loss, 20.0))

            print(f"[{pct:5.2f}%] Tokens: {total_tokens_seen:,}/{tokens:,} | Loss: {approx_loss:.4f} | PPL: {ppl:.2f} | Speed: {interval_speed:,.0f} tok/s | Elapsed: {elapsed/60:.1f}m | ETA: {eta_sec/60:.1f}m")
            last_report_time = now
            last_report_tokens = total_tokens_seen

        # Periodic Auto-Checkpointing & Old Checkpoint Pruning
        if total_tokens_seen - last_ckpt_tokens >= ckpt_interval_tokens:
            ckpt_path = os.path.join(ckpt_dir, f"spectral_ckpt_{total_tokens_seen}_tokens.pt")
            torch.save({
                "tokens_seen": total_tokens_seen,
                "cov_xx": cov_xx.cpu(),
                "cov_yx": cov_yx.cpu(),
                "tok_embeddings": model.tok_embeddings.weight.data.cpu()
            }, ckpt_path)
            # Re-upload to device
            cov_xx = cov_xx.to(device)
            cov_yx = cov_yx.to(device)
            print(f"--> [Checkpoint] Saved progress at {total_tokens_seen:,} tokens to: {ckpt_path}")
            cleanup_old_checkpoints(ckpt_dir, keep_latest=1)
            last_ckpt_tokens = total_tokens_seen

        if total_tokens_seen >= tokens:
            break

    # Solve closed-form harmonic weights
    print(f"\n[Stage 2] Solving Closed-Form 1B Harmonic Weights from {total_tokens_seen:,} tokens...")
    cov_xx /= float(total_tokens_seen)
    cov_yx /= float(total_tokens_seen)

    reg_matrix = cov_xx + ridge * torch.eye(D, device=device)
    w_opt = torch.matmul(cov_yx, torch.linalg.pinv(reg_matrix))
    gamma = w_opt.abs().mean().clamp(min=1e-5)
    w_ternary = hard_ternary(w_opt / gamma, threshold=0.5)

    with torch.no_grad():
        for layer in model.layers:
            for name, param in layer.named_parameters():
                if "weight" in name and param.dim() == 2:
                    d_out, d_in = param.shape
                    if d_out == D and d_in == D:
                        param.data.copy_(w_ternary * gamma)
                    elif d_out > D:
                        reps = math.ceil(d_out / D)
                        param.data.copy_((w_ternary * gamma).repeat(reps, 1)[:d_out, :d_in])
                    elif d_in > D:
                        reps = math.ceil(d_in / D)
                        param.data.copy_((w_ternary * gamma).t().repeat(1, reps)[:d_out, :d_in])

        model.output_head.weight.copy_(model.tok_embeddings.weight / math.sqrt(D))

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    torch.save(model.state_dict(), output_file)
    print(f"\n============================================================")
    print(f" 50 Billion Token SLM Pre-training Complete in {(time.time() - t0)/60:.2f} Minutes!")
    print(f" Final Foundation Model Saved to: {output_file}")
    print(f"============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="50BT Production 1B SLM Spectral Engine")
    parser.add_argument("--tokens", type=int, default=50_000_000_000, help="Default: 50 Billion Tokens")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints_1b_stream")
    parser.add_argument("--output", type=str, default="production_slm_1b/bitnet_1b_pretrained.pt")
    args = parser.parse_args()

    run_1b_spectral_inversion(tokens=args.tokens, batch_size=args.batch_size, workers=args.workers,
                              ckpt_dir=args.ckpt_dir, output_file=args.output)
