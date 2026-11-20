#!/usr/bin/env bash
# ==============================================================================
# Blazing Fast Pure TA Pipeline (3 - 5 Minutes Total on L40S)
# Strips all MIGE and swarm overhead. Pure Temperature Annealing (500 - 800 steps).
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

STEPS="${1:-800}"           # Only 800 steps needed from Hopfield init
BATCH_SIZE="${2:-32}"       # 32 batch size on L40S (~16,384 tokens/step)
EVAL_SAMPLES="${3:-250}"

echo "=================================================================="
echo " Running Blazing Fast Pure TA Pipeline"
echo " CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo " Target steps         : $STEPS (Takes ~3-5 minutes on L40S)"
echo " Batch size           : $BATCH_SIZE"
echo "=================================================================="

# 1. Pure TA Training
python3 train_hopfield_pure_ta_blazing.py \
    --steps "$STEPS" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "checkpoints_pure_ta"

# 2. Phase B Evaluation
echo "--> Running Phase B downstream evaluation on final model..."
python3 evaluate_hopfield_phase_b.py \
    --checkpoint "checkpoints_pure_ta/pure_ta_final.pt" \
    --samples "$EVAL_SAMPLES" \
    --output "phase_b_pure_ta_results.json"

echo "=================================================================="
echo " Done in record time! Results saved in phase_b_pure_ta_results.json"
echo "=================================================================="
