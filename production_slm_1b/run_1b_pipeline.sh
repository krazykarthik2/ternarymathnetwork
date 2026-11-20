#!/usr/bin/env bash
# ==============================================================================
# Master Pipeline: 1 Billion Parameter Production SLM
# Stage 1: High-Volume Pretraining (10BT to 25BT) via Streaming Spectral Inversion (~2.5 - 6 hrs)
# Stage 2: SFT / Dialogue Alignment on Conversational ChatML (~45 mins)
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

TOKENS="${1:-10000000000}"   # 10 Billion Tokens by default
SFT_STEPS="${2:-2500}"        # Dialogue tuning steps

echo "=================================================================="
echo " 1 Billion Parameter Production SLM Pipeline"
echo " Pre-training Target Tokens : $TOKENS"
echo " Dialogue Tuning Steps      : $SFT_STEPS"
echo " CUDA_VISIBLE_DEVICES       : $CUDA_VISIBLE_DEVICES"
echo "=================================================================="

# ------------------------------------------------------------------
# Stage 1: Spectral Pretraining across the token stream
# ------------------------------------------------------------------
echo "--> [Stage 1/3] Starting High-Volume 1B Spectral Pre-training..."
python3 train_1b_spectral_stream.py \
    --tokens "$TOKENS" \
    --batch_size 128 \
    --workers 8 \
    --output "bitnet_1b_pretrained.pt"

# ------------------------------------------------------------------
# Stage 2: Dialogue Instruction Tuning (ChatML Alignment)
# ------------------------------------------------------------------
echo "--> [Stage 2/3] Starting Dialogue Instruction Tuning..."
python3 tune_1b_dialogue.py \
    --base_ckpt "bitnet_1b_pretrained.pt" \
    --steps "$SFT_STEPS" \
    --batch_size 16 \
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
