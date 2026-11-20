#!/usr/bin/env bash
# ==============================================================================
# GPU Master Script: 10 Billion Token Streaming Inversion in < 1 Hour
# Runs on 1x NVIDIA L40S with 8-worker preloading & zero-backprop streaming math.
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
BATCH_SIZE="${2:-256}"        # Large batch size enabled by forward-only O(1) memory
WORKERS="${3:-8}"

echo "=================================================================="
echo " Starting 10BT Closed-Form Spectral-Topology Training"
echo " CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo " Target Tokens        : $TOKENS"
echo " Batch Size           : $BATCH_SIZE"
echo " CPU Worker Cores     : $WORKERS"
echo "=================================================================="

python3 streaming_spectral_bitnet.py \
    --tokens "$TOKENS" \
    --batch_size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --output "topology_bitnet_10bt.pt"

echo "=================================================================="
echo " 10BT Training Complete! Running Phase B Evaluation..."
echo "=================================================================="

python3 ../hopfield/eval_checkpoint_phase_b.py \
    --checkpoint "topology_bitnet_10bt.pt" \
    --samples 250 \
    --output "phase_b_10bt_results.json"

echo "=================================================================="
echo " All Benchmarks Finished!"
echo "=================================================================="
