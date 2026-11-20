#!/usr/bin/env bash
# ==============================================================================
# Run Direct Hopfield Single-Shot Weight Calculation + Phase B Evaluation
# Automatically selects the idle GPU if another is busy.
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

CKPT="${1:-}"
SAMPLES="${2:-250}"
CALC_SAMPLES="${3:-512}"

echo "=================================================================="
echo " Launching Phase B Hopfield-BitNet Downstream Evaluation"
echo " CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo " Checkpoint           : ${CKPT:-'None (Single-Shot Analytical Calculation)'}"
echo " Eval Samples         : $SAMPLES per task"
echo "=================================================================="

if [ -n "$CKPT" ]; then
    python3 evaluate_hopfield_phase_b.py --checkpoint "$CKPT" --samples "$SAMPLES" --calc_samples "$CALC_SAMPLES"
else
    python3 evaluate_hopfield_phase_b.py --samples "$SAMPLES" --calc_samples "$CALC_SAMPLES"
fi

echo "=================================================================="
echo " Phase B evaluation finished! Results saved in phase_b_hopfield_results.json"
echo "=================================================================="
