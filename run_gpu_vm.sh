#!/bin/bash
# ==============================================================================
# GPU VM Master Launch Script: BitNet b1.58 Phase A & Phase B Benchmark
# Supports: mige_hc, tsgo, mige_nbs, mige_1, ste, temp_anneal, eval
#
# Usage examples:
#   ./run_gpu_vm.sh mige_hc             # Starts MIGE-HC (Homotopy Continuation Sobolev)
#   ./run_gpu_vm.sh tsgo                # Starts TSGO
#   ./run_gpu_vm.sh temp_anneal         # Starts Temperature Annealing
#   ./run_gpu_vm.sh mige                # Starts standard MIGE-1
#   ./run_gpu_vm.sh eval                # Only runs held-out evaluation suite
# ==============================================================================
set -e

START_STAGE="${1:-mige_hc}"

echo "=== Step 1: Verifying GPU & Python Environment ==="
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device Count:', torch.cuda.device_count(), '| Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

DATASET_NAME="HuggingFaceFW/fineweb-edu"
SUBSET_NAME="sample-10BT"
STEPS=10000
BATCH_SIZE=16
SEQ_LEN=512

echo "=== Target: $DATASET_NAME (subset: $SUBSET_NAME) | Target Steps: $STEPS ==="
echo "=== Starting from stage: $START_STAGE ==="

# ------------------------------------------------------------------------------
# 1. Primary Target: MIGE-HC (Homotopy Continuation Sobolev)
# ------------------------------------------------------------------------------
if [ "$START_STAGE" = "mige_hc" ] || [ "$START_STAGE" = "homotopy" ] || [ "$START_STAGE" = "hc" ]; then
    echo "--> [TARGET] Starting Experiment: MIGE-HC (Homotopy Continuation Sobolev)..."
    python train_phase_a.py --scale 125M --method mige_hc --dataset "$DATASET_NAME" --subset "$SUBSET_NAME" --steps $STEPS --batch_size $BATCH_SIZE --seq_len $SEQ_LEN --hc_switch_ratio 0.60
    START_STAGE="eval"
fi

# ------------------------------------------------------------------------------
# 2. Control: TSGO
# ------------------------------------------------------------------------------
if [ "$START_STAGE" = "tsgo" ] || [ "$START_STAGE" = "swarm" ]; then
    echo "--> Starting Experiment: TSGO..."
    python train_phase_a.py --scale 125M --method tsgo --dataset "$DATASET_NAME" --subset "$SUBSET_NAME" --steps $STEPS --batch_size $BATCH_SIZE --seq_len $SEQ_LEN
    START_STAGE="eval"
fi

# ------------------------------------------------------------------------------
# 3. Control: Temperature Annealing
# ------------------------------------------------------------------------------
if [ "$START_STAGE" = "temp_anneal" ]; then
    echo "--> Starting Control: Temperature Annealing..."
    python train_phase_a.py --scale 125M --method temp_anneal --dataset "$DATASET_NAME" --subset "$SUBSET_NAME" --steps $STEPS --batch_size $BATCH_SIZE --seq_len $SEQ_LEN
    START_STAGE="eval"
fi

# ------------------------------------------------------------------------------
# 4. Control: Standard MIGE-1
# ------------------------------------------------------------------------------
if [ "$START_STAGE" = "mige" ] || [ "$START_STAGE" = "mige_1" ]; then
    echo "--> Starting Experiment: MIGE-1 (Adaptive Sobolev)..."
    python train_phase_a.py --scale 125M --method mige_1 --dataset "$DATASET_NAME" --subset "$SUBSET_NAME" --steps $STEPS --batch_size $BATCH_SIZE --seq_len $SEQ_LEN
    START_STAGE="eval"
fi

# ------------------------------------------------------------------------------
# 5. Held-Out Evaluation & Comparison
# ------------------------------------------------------------------------------
echo "=== Step 2: Running Evaluation Suite on Held-Out Tokens ==="
python evaluate_checkpoints_discrete.py
python update_comparison_charts.py

echo "=== Benchmark Complete. All Phase A Results Generated & Charts Updated ==="
