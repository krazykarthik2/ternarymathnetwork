#!/usr/bin/env bash
# ==============================================================================
# GPU Launch Script: Hopfield Init + Pure Temperature Annealing (TA / MIGE-TA)
# Bypasses heuristic methods and directly pairs Hopfield with the top-performing TA.
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

METHOD="${1:-mige_ta}"       # Options: mige_ta (Mollified TA) or temp_anneal (Vanilla TA)
STEPS="${2:-2500}"          # 2000-2500 steps for fast convergence from Hopfield init
BATCH_SIZE="${3:-32}"       # 32 fits comfortably in L40S 48GB VRAM
EVAL_SAMPLES="${4:-250}"

echo "=================================================================="
echo " Starting Hopfield Init + Pure $METHOD Pipeline on GPU"
echo " Method       : $METHOD"
echo " Steps        : $STEPS"
echo " Batch Size   : $BATCH_SIZE"
echo "=================================================================="

# 1. Train with Hopfield Init + Pure TA
python3 train_hopfield_ta_fast.py \
    --method "$METHOD" \
    --steps "$STEPS" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "checkpoints_hopfield_ta"

# 2. Directly evaluate on Phase B
echo "--> Running Phase B downstream evaluation..."
python3 evaluate_hopfield_phase_b.py \
    --checkpoint "checkpoints_hopfield_ta/hopfield_ta_final.pt" \
    --samples "$EVAL_SAMPLES" \
    --output "phase_b_hopfield_ta_results.json"

echo "=================================================================="
echo " Pipeline Complete! Results saved in phase_b_hopfield_ta_results.json"
echo "=================================================================="
