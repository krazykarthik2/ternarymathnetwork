#!/bin/bash
# ==============================================================================
# GPU VM Master Launch Script: BitNet b1.58 Phase A & Phase B Benchmark
# Flexible command-line arguments to start from any method or resume.
#
# Usage examples:
#   ./run_gpu_vm.sh                     # Runs all methods starting from STE
#   ./run_gpu_vm.sh temp_anneal         # Starts directly from Temperature Annealing
#   ./run_gpu_vm.sh mige                # Starts directly from MIGE
# ==============================================================================
set -e

# Default to starting at STE if no argument given, or accept 'temp_anneal' / 'mige'
START_STAGE="${1:-ste}"

echo "=== Step 1: Verifying GPU & Python Environment ==="
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device Count:', torch.cuda.device_count(), '| Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

DATASET_NAME="HuggingFaceFW/fineweb-edu"
SUBSET_NAME="sample-10BT"
STEPS=10000
BATCH_SIZE=32
SEQ_LEN=512

echo "=== Target: $DATASET_NAME (subset: $SUBSET_NAME) | Target Steps: $STEPS ==="
echo "=== Starting from stage: $START_STAGE ==="

# ------------------------------------------------------------------------------
# 1. Control 1: STE Baseline
# ------------------------------------------------------------------------------
if [ "$START_STAGE" = "ste" ]; then
    echo "--> [1/3] Starting Control 1: STE Baseline..."
    python train_phase_a.py --scale 125M --method ste --dataset "$DATASET_NAME" --subset "$SUBSET_NAME" --steps $STEPS --batch_size $BATCH_SIZE --seq_len $SEQ_LEN
    START_STAGE="temp_anneal"
fi

# ------------------------------------------------------------------------------
# 2. Control 2: Temperature Annealing
# ------------------------------------------------------------------------------
if [ "$START_STAGE" = "temp_anneal" ]; then
    echo "--> [2/3] Starting Control 2: Temperature Annealing..."
    python train_phase_a.py --scale 125M --method temp_anneal --dataset "$DATASET_NAME" --subset "$SUBSET_NAME" --steps $STEPS --batch_size $BATCH_SIZE --seq_len $SEQ_LEN
    START_STAGE="mige"
fi

# ------------------------------------------------------------------------------
# 3. Experiment: MIGE (Adaptive Two-Phase Sobolev Weak Derivatives)
# ------------------------------------------------------------------------------
if [ "$START_STAGE" = "mige" ]; then
    echo "--> [3/3] Starting Experiment: MIGE..."
    python train_phase_a.py --scale 125M --method mige --dataset "$DATASET_NAME" --subset "$SUBSET_NAME" --steps $STEPS --batch_size $BATCH_SIZE --seq_len $SEQ_LEN
fi

echo "=== Step 3: Phase B True Downstream Zero-Shot Reasoning Evaluation ==="
if [ -f "checkpoints/125M_ste/checkpoint_step_${STEPS}.pt" ]; then
    echo "--> Evaluating STE Checkpoint..."
    python evaluate_phase_b.py --checkpoint checkpoints/125M_ste/checkpoint_step_${STEPS}.pt --scale 125M
fi

if [ -f "checkpoints/125M_temp_anneal/checkpoint_step_${STEPS}.pt" ]; then
    echo "--> Evaluating Temp Anneal Checkpoint..."
    python evaluate_phase_b.py --checkpoint checkpoints/125M_temp_anneal/checkpoint_step_${STEPS}.pt --scale 125M
fi

if [ -f "checkpoints/125M_mige/checkpoint_step_${STEPS}.pt" ]; then
    echo "--> Evaluating MIGE Checkpoint..."
    python evaluate_phase_b.py --checkpoint checkpoints/125M_mige/checkpoint_step_${STEPS}.pt --scale 125M
fi

echo "=== Benchmark Complete. All Phase A & B Results Generated ==="
