#!/bin/bash
# ==============================================================================
# Phase B Downstream Zero-Shot Accuracy Benchmark Runner
# Tasks: HellaSwag, PIQA, ARC-Easy, WinoGrande
# Evaluates under strict hardware ternary {-1, 0, +1} quantization snaps.
#
# Usage:
#   ./run_tests.sh                     # Runs on all completed major models
#   ./run_tests.sh mige                # Runs Phase B on MIGE-1 (Adaptive Sobolev)
#   ./run_tests.sh temp_anneal         # Runs Phase B on Temperature Annealing
#   ./run_tests.sh ste                 # Runs Phase B on STE Baseline
# ==============================================================================
set -e

TARGET="${1:-all}"
SAMPLES=500   # 500 validation samples per task for fast, accurate zero-shot scoring

echo "=============================================================================="
echo "PHASE B: ZERO-SHOT DOWNSTREAM SUITE (HellaSwag, PIQA, ARC-Easy, WinoGrande)"
echo "Hardware Check:"
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo "=============================================================================="

# 1. Evaluate MIGE-1 (The Strictly Discrete State-of-the-Art)
if [ "$TARGET" = "all" ] || [ "$TARGET" = "mige" ] || [ "$TARGET" = "mige_1" ]; then
    CKPT="checkpoints/125M_mige/checkpoint_step_10000.pt"
    if [ -f "$CKPT" ]; then
        echo ""
        echo ">>> Running Phase B on MIGE-1 (Adaptive Sobolev) <<<"
        python evaluate_phase_b.py --checkpoint "$CKPT" --name "MIGE-1 (Adaptive Sobolev)" --num_samples $SAMPLES
    else
        echo "[-] Checkpoint not found: $CKPT (skipping)"
    fi
fi

# 2. Evaluate Temperature Annealing (Continuous Relaxation Upper Bound)
if [ "$TARGET" = "all" ] || [ "$TARGET" = "temp_anneal" ] || [ "$TARGET" = "ta" ]; then
    CKPT="checkpoints/125M_temp_anneal/checkpoint_step_10000.pt"
    if [ -f "$CKPT" ]; then
        echo ""
        echo ">>> Running Phase B on Temperature Annealing <<<"
        python evaluate_phase_b.py --checkpoint "$CKPT" --name "Temperature Annealing" --num_samples $SAMPLES
    else
        echo "[-] Checkpoint not found: $CKPT (skipping)"
    fi
fi

# 3. Evaluate STE Baseline
if [ "$TARGET" = "all" ] || [ "$TARGET" = "ste" ]; then
    CKPT="checkpoints/125M_ste/checkpoint_step_10000.pt"
    if [ -f "$CKPT" ]; then
        echo ""
        echo ">>> Running Phase B on STE Baseline <<<"
        python evaluate_phase_b.py --checkpoint "$CKPT" --name "STE Baseline" --num_samples $SAMPLES
    else
        echo "[-] Checkpoint not found: $CKPT (skipping)"
    fi
fi

echo ""
echo "=============================================================================="
echo "Phase B downstream evaluation complete!"
echo "Results written to: results/phase_b_results.json"
echo "=============================================================================="
