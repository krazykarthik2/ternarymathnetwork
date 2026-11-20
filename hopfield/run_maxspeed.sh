#!/usr/bin/env bash
# ==============================================================================
# PyTorch MaxSpeed Runner (L40S Native Acceleration):
# - torch.compile (reduce-overhead)
# - FlashAttention-2 (F.scaled_dot_product_attention)
# - Multiprocessing CPU Preload Workers (num_workers=4, pin_memory=True)
# - Asynchronous non-blocking GPU transfers
# - Fused AdamW & TF32 tensor cores
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

STEPS="${1:-800}"
BATCH_SIZE="${2:-32}"
WORKERS="${3:-4}"

echo "=================================================================="
echo " Launching MaxSpeed PyTorch Engine"
echo " CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo " Target Steps         : $STEPS"
echo " Batch Size           : $BATCH_SIZE"
echo " CPU Worker Cores     : $WORKERS"
echo "=================================================================="

# 1. Run Ultra-Fast Training
python3 train_hopfield_torch_maxspeed.py \
    --steps "$STEPS" \
    --batch_size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --output_dir "checkpoints_maxspeed"

# 2. Immediate Phase B Evaluation
echo "--> Running Phase B downstream evaluation..."
python3 evaluate_hopfield_phase_b.py \
    --checkpoint "checkpoints_maxspeed/pure_ta_maxspeed.pt" \
    --samples 250 \
    --output "phase_b_maxspeed_results.json"

echo "=================================================================="
echo " Pipeline Complete in Record Speed! Results in phase_b_maxspeed_results.json"
echo "=================================================================="
