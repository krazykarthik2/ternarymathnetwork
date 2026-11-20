#!/usr/bin/env bash
# ==============================================================================
# Ultra-Fast Hopfield + Swarm/TA Pipeline on L40S GPU
# Trains in 10-15 minutes instead of hours, then directly runs Phase B evaluation.
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

STEPS="${1:-1500}"
BATCH_SIZE="${2:-32}"

echo "=================================================================="
echo " Starting Fast Hopfield-BitNet Swarm/TA Pipeline on GPU"
echo " Target steps: $STEPS | Batch size: $BATCH_SIZE"
echo "=================================================================="

# 1. Fast Hybrid Training (Hopfield Init + Swarm Boundary Mutation + Short-Burst TA)
python3 train_hopfield_fast_swarm.py --steps "$STEPS" --batch_size "$BATCH_SIZE" --output_dir "checkpoints_fast_swarm"

# 2. Phase B Evaluation on the resulting checkpoint
echo "--> Running Phase B downstream evaluation on trained checkpoint..."
python3 evaluate_hopfield_phase_b.py --checkpoint "checkpoints_fast_swarm/hopfield_fast_final.pt" --samples 250 --output "phase_b_fast_swarm_results.json"

echo "=================================================================="
echo " Pipeline Complete! Results saved in phase_b_fast_swarm_results.json"
echo "=================================================================="
