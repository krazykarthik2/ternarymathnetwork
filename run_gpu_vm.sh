#!/bin/bash
# ==============================================================================
# GPU VM Master Launch Script: BitNet b1.58 Phase A & Phase B Benchmark
# Supports: tsgo, mige_nbs, mige_1, ste, temp_anneal, eval
#
# Usage examples:
#   ./run_gpu_vm.sh tsgo                # Starts TSGO (Ternary Swarm-Guided Optimization)
#   ./run_gpu_vm.sh mige_nbs            # Starts from Normalized Bounded-Sobolev (MIGE-NBS)
#   ./run_gpu_vm.sh ste                 # Starts from STE Baseline
#   ./run_gpu_vm.sh temp_anneal         # Starts from Temperature Annealing
#   ./run_gpu_vm.sh mige                # Starts from standard MIGE-1
#   ./run_gpu_vm.sh eval                # Only runs held-out evaluation suite
# ==============================================================================
set -e

START_STAGE="${1:-tsgo}"

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
# 1. Primary Target: TSGO (Ternary Swarm-Guided Optimization)
# ------------------------------------------------------------------------------
if [ "$START_STAGE" = "tsgo" ] || [ "$START_STAGE" = "swarm" ]; then
    echo "--> [TARGET] Starting Experiment: TSGO (Ternary Swarm-Guided Optimization)..."
    python train_phase_a.py --scale 125M --method tsgo --dataset "$DATASET_NAME" --subset "$SUBSET_NAME" --steps $STEPS --batch_size $BATCH_SIZE --seq_len $SEQ_LEN
    START_STAGE="eval"
fi

# ------------------------------------------------------------------------------
# 2. Control: MIGE-NBS (Normalized Bounded-Sobolev)
# ------------------------------------------------------------------------------
if [ "$START_STAGE" = "mige_nbs" ] || [ "$START_STAGE" = "migenbs" ] || [ "$START_STAGE" = "nbs" ]; then
    echo "--> Starting Experiment: MIGE-NBS (Normalized Bounded-Sobolev)..."
    python train_phase_a.py --scale 125M --method mige_nbs --dataset "$DATASET_NAME" --subset "$SUBSET_NAME" --steps $STEPS --batch_size $BATCH_SIZE --seq_len $SEQ_LEN
    START_STAGE="eval"
fi

# ------------------------------------------------------------------------------
# 3. Control: STE Baseline
# ------------------------------------------------------------------------------
if [ "$START_STAGE" = "ste" ]; then
    echo "--> Starting Control 1: STE Baseline..."
    python train_phase_a.py --scale 125M --method ste --dataset "$DATASET_NAME" --subset "$SUBSET_NAME" --steps $STEPS --batch_size $BATCH_SIZE --seq_len $SEQ_LEN
    START_STAGE="temp_anneal"
fi

# ------------------------------------------------------------------------------
# 4. Control: Temperature Annealing
# ------------------------------------------------------------------------------
if [ "$START_STAGE" = "temp_anneal" ]; then
    echo "--> Starting Control 2: Temperature Annealing..."
    python train_phase_a.py --scale 125M --method temp_anneal --dataset "$DATASET_NAME" --subset "$SUBSET_NAME" --steps $STEPS --batch_size $BATCH_SIZE --seq_len $SEQ_LEN
    START_STAGE="mige"
fi

# ------------------------------------------------------------------------------
# 5. Control: Standard MIGE-1
# ------------------------------------------------------------------------------
if [ "$START_STAGE" = "mige" ] || [ "$START_STAGE" = "mige_1" ]; then
    echo "--> Starting Experiment: MIGE-1 (Adaptive Sobolev)..."
    python train_phase_a.py --scale 125M --method mige_1 --dataset "$DATASET_NAME" --subset "$SUBSET_NAME" --steps $STEPS --batch_size $BATCH_SIZE --seq_len $SEQ_LEN
    START_STAGE="eval"
fi

# ------------------------------------------------------------------------------
# 6. Held-Out Evaluation & Comparison
# ------------------------------------------------------------------------------
echo "=== Step 2: Running Evaluation Suite on Held-Out Tokens ==="
python evaluate_checkpoints_discrete.py
python update_comparison_charts.py

echo "=== Benchmark Complete. All Phase A Results Generated & Charts Updated ==="
