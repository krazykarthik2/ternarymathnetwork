#!/usr/bin/env bash
# ==============================================================================
# Master Pipeline: 1 Billion Parameter Production SLM (50 Billion Tokens)
# Stage 1: 50BT High-Volume Pre-training with Auto-Checkpointing & Recovery (~12.6 hrs)
# Stage 2: SFT / Dialogue Alignment on Conversational ChatML with Auto-Recovery (~1.2 hrs)
# Stage 3: Bit-pack into 260MB edge binary for instant local CPU chat
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

select_idle_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        export CUDA_VISIBLE_DEVICES=0
        return
    fi
    BEST_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -n 1 | awk -F',' '{print $1}' | tr -d ' ')
    USED_MEM=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -n 1 | awk -F',' '{print $2}' | tr -d ' ')
    if [ -n "$BEST_GPU" ]; then
        echo "--> Selected idle GPU index: $BEST_GPU (Current memory used: ${USED_MEM} MiB)"
        export CUDA_VISIBLE_DEVICES="$BEST_GPU"
    else
        export CUDA_VISIBLE_DEVICES=0
    fi
}

select_idle_gpu

# Default: Full 50 Billion Tokens
TOKENS="${1:-50000000000}"
SFT_STEPS="${2:-5000}"

echo "=================================================================="
echo " 1 Billion Parameter Production SLM Pipeline"
echo " Pre-training Target Tokens : $TOKENS (50 Billion Tokens)"
echo " Dialogue Tuning Steps      : $SFT_STEPS (5,000 steps)"
echo " CUDA_VISIBLE_DEVICES       : $CUDA_VISIBLE_DEVICES"
echo "=================================================================="

# ------------------------------------------------------------------
# Stage 1: 50BT Streaming Pre-training with Auto-Resume & Auto-Cleanup
# ------------------------------------------------------------------
echo "--> [Stage 1/3] Starting 50BT Spectral Pre-training (Auto-resumes if interrupted)..."
python3 train_1b_spectral_stream.py \
    --tokens "$TOKENS" \
    --batch_size 128 \
    --workers 8 \
    --ckpt_dir "checkpoints_1b_stream" \
    --output "bitnet_1b_pretrained.pt"

# ------------------------------------------------------------------
# Stage 2: Dialogue Instruction Tuning with Auto-Resume
# ------------------------------------------------------------------
echo "--> [Stage 2/3] Starting Dialogue Instruction Tuning..."
python3 tune_1b_dialogue.py \
    --base_ckpt "bitnet_1b_pretrained.pt" \
    --steps "$SFT_STEPS" \
    --batch_size 16 \
    --ckpt_dir "checkpoints_1b_sft" \
    --output "bitnet_1b_dialogue_chat.pt"

# ------------------------------------------------------------------
# Stage 3: Bit-pack into 260MB 1.58-bit production binary
# ------------------------------------------------------------------
echo "--> [Stage 3/3] Packing 1B model into 260MB 1.58-bit binary..."
python3 ../hopfield/pack_checkpoint_158bit.py \
    --checkpoint "bitnet_1b_dialogue_chat.pt" \
    --output "bitnet_1b_packed_edge.pt"

echo "=================================================================="
echo " 1B SLM Pipeline Finished Successfully!"
echo " Deployable Model: bitnet_1b_packed_edge.pt (~262 MB)"
echo " Test locally using: python3 chat_local_1b.py --checkpoint bitnet_1b_packed_edge.pt"
echo "=================================================================="
