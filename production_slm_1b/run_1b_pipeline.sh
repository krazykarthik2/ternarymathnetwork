#!/usr/bin/env bash
# ==============================================================================
# Master Pipeline: 1 Billion Parameter Production SLM (50 Billion Tokens)
# Multi-GPU Auto-Detection: Automatically runs across 2x L40S GPUs via torchrun
# Stage 1: 50BT High-Volume Pre-training with Auto-Checkpointing & Recovery (~6.3 hrs on 2x L40S)
# Stage 2: SFT / Dialogue Alignment on Conversational ChatML with Auto-Recovery (~40 mins on 2x L40S)
# Stage 3: Bit-pack into 260MB edge binary for instant local CPU chat
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

detect_gpus() {
    # Check actual visible GPUs to PyTorch (respecting CUDA_VISIBLE_DEVICES or container isolation)
    NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo 0)
}

detect_gpus

# Default: Full 50 Billion Tokens
TOKENS="${1:-50000000000}"
SFT_STEPS="${2:-5000}"

echo "=================================================================="
echo " 1 Billion Parameter Production SLM Pipeline"
echo " PyTorch Visible CUDA GPUs  : $NUM_GPUS"
echo " Pre-training Target Tokens : $TOKENS (50 Billion Tokens)"
echo " Dialogue Tuning Steps      : $SFT_STEPS (5,000 steps)"
echo "=================================================================="

# Determine launcher command
if [ "$NUM_GPUS" -ge 2 ]; then
    echo "--> [Multi-GPU] Scaling across $NUM_GPUS GPUs using torchrun (Near Linear Speedup)..."
    LAUNCH_CMD="torchrun --standalone --nproc_per_node=$NUM_GPUS"
elif [ "$NUM_GPUS" -eq 1 ]; then
    echo "--> [Single-GPU] Running on 1x visible L40S GPU..."
    LAUNCH_CMD="python3"
else
    echo "--> [CPU fallback] Running on CPU..."
    LAUNCH_CMD="python3"
fi

# ------------------------------------------------------------------
# Stage 1: 50BT Streaming Pre-training with Auto-Resume & Auto-Cleanup
# ------------------------------------------------------------------
echo "--> [Stage 1/3] Starting 50BT Spectral Pre-training (Auto-resumes if interrupted)..."
$LAUNCH_CMD train_1b_spectral_stream.py \
    --tokens "$TOKENS" \
    --batch_size 128 \
    --workers 8 \
    --ckpt_dir "checkpoints_1b_stream" \
    --output "bitnet_1b_pretrained.pt"

# ------------------------------------------------------------------
# Stage 2: Dialogue SFT Alignment with Auto-Resume & Auto-Cleanup
# ------------------------------------------------------------------
echo ""
echo "--> [Stage 2/3] Starting ChatML Dialogue Instruction Tuning (Auto-resumes if interrupted)..."
$LAUNCH_CMD tune_1b_dialogue.py \
    --base_ckpt "bitnet_1b_pretrained.pt" \
    --steps "$SFT_STEPS" \
    --batch_size 16 \
    --ckpt_dir "checkpoints_1b_sft" \
    --output "bitnet_1b_dialogue_chat.pt"

# ------------------------------------------------------------------
# Stage 3: Bit-pack into 260MB edge binary for instant local CPU chat
# ------------------------------------------------------------------
echo ""
echo "--> [Stage 3/3] Compressing 1B FP32 Weights into 1.58-bit Ternary Edge Binary..."
python3 -c "
import sys
sys.path.insert(0, '../hopfield')
from pack_checkpoint_158bit import pack_checkpoint_158bit
pack_checkpoint_158bit('bitnet_1b_dialogue_chat.pt', 'bitnet_1b_packed_edge.pt')
"

echo "=================================================================="
echo " 1B SLM Full Training & Dialogue Alignment Completed!"
echo " 1. Pre-trained Base Model : bitnet_1b_pretrained.pt"
echo " 2. ChatML Assistant Model : bitnet_1b_dialogue_chat.pt"
echo " 3. Edge-Deployable 260MB  : bitnet_1b_packed_edge.pt"
echo "=================================================================="
