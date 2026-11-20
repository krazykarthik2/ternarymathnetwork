#!/usr/bin/env bash
# ==============================================================================
# GPU Master Script: Zero-Backprop Production Spectral-Laplacian BitNet
# Runs full forward-only covariance accumulation + Zelnik-Manor Graph Laplacian
# solves 10 Billion Tokens in < 1 hour, packs into 40MB, and evaluates Phase B.
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
BATCH_SIZE="${2:-256}"
WORKERS="${3:-8}"
RIDGE="${4:-0.0001}"

echo "=================================================================="
echo " Starting Zero-Backprop Spectral-Laplacian Production Engine"
echo " CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo " Target Tokens        : $TOKENS"
echo " Batch Size           : $BATCH_SIZE"
echo " Worker Cores         : $WORKERS"
echo " Ridge Regularizer    : $RIDGE"
echo "=================================================================="

# 1. Run Spectral-Laplacian Training
python3 train_production_laplacian.py \
    --tokens "$TOKENS" \
    --batch_size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --ridge "$RIDGE" \
    --output "production_laplacian_10bt.pt"

# 2. Pack Checkpoint into 40MB 1.58-bit binary
echo "--> Packing into 40MB 1.58-bit ternary binary..."
python3 ../hopfield/pack_checkpoint_158bit.py \
    --checkpoint "production_laplacian_10bt.pt" \
    --output "production_laplacian_packed.pt"

# 3. Run Phase B Benchmark
echo "--> Running Phase B downstream evaluation..."
python3 ../hopfield/eval_checkpoint_phase_b.py \
    --checkpoint "production_laplacian_10bt.pt" \
    --samples 250 \
    --output "phase_b_production_laplacian_results.json"

echo "=================================================================="
echo " Production Spectral-Laplacian Pipeline Completed Successfully!"
echo "=================================================================="
