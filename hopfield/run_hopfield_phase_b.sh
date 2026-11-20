#!/usr/bin/env bash
# ==============================================================================
# Run Direct Hopfield Single-Shot Weight Calculation + Phase B Evaluation
# Supports pure analytical zero-training evaluation OR evaluating a checkpoint.
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CKPT="${1:-}"
SAMPLES="${2:-250}"
CALC_SAMPLES="${3:-512}"

echo "=================================================================="
echo " Launching Phase B Hopfield-BitNet Downstream Evaluation"
echo " Checkpoint   : ${CKPT:-'None (Single-Shot Analytical Calculation)'}"
echo " Eval Samples : $SAMPLES per task"
echo "=================================================================="

if [ -n "$CKPT" ]; then
    python3 evaluate_hopfield_phase_b.py --checkpoint "$CKPT" --samples "$SAMPLES" --calc_samples "$CALC_SAMPLES"
else
    python3 evaluate_hopfield_phase_b.py --samples "$SAMPLES" --calc_samples "$CALC_SAMPLES"
fi

echo "=================================================================="
echo " Phase B evaluation finished! Results saved in phase_b_hopfield_results.json"
echo "=================================================================="
