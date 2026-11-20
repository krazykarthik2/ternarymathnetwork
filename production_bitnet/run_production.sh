#!/usr/bin/env bash
# ==============================================================================
# Production BitNet b1.58 Execution Pipeline
# Official Microsoft formulation: INT8 activations x INT2 {-1,0,1} ternary weights.
# Auto-detects idle GPU.
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

STEPS="${1:-10000}"
BATCH_SIZE="${2:-32}"
WORKERS="${3:-4}"

echo "=================================================================="
echo " Launching Production BitNet b1.58 Pipeline"
echo " CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo " Target Steps         : $STEPS"
echo " Batch Size           : $BATCH_SIZE"
echo "=================================================================="

# 1. Run Production Training
python3 train_production.py \
    --steps "$STEPS" \
    --batch_size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --output_dir "checkpoints_production"

# 2. Pack Checkpoint into 40MB 1.58-bit binary
echo "--> Packing final model into 1.58-bit ternary binary..."
python3 ../hopfield/pack_checkpoint_158bit.py \
    --checkpoint "checkpoints_production/bitnet_prod_step_${STEPS}.pt" \
    --output "production_bitnet_packed.pt"

# 3. Run Phase B Benchmark
echo "--> Running Phase B downstream evaluation on final production model..."
python3 ../hopfield/eval_checkpoint_phase_b.py \
    --checkpoint "checkpoints_production/bitnet_prod_step_${STEPS}.pt" \
    --samples 250 \
    --output "phase_b_production_results.json"

echo "=================================================================="
echo " Production Pipeline Completed Successfully!"
echo "=================================================================="
