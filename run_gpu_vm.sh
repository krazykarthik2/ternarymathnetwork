#!/bin/bash
# ==============================================================================
# GPU VM Master Launch Script: BitNet b1.58 Phase A & Phase B Benchmark
# Supports: ste, temp_anneal, mige, mige2p, and eval
#
# Usage examples:
#   ./run_gpu_vm.sh                     # Runs all methods starting from STE
#   ./run_gpu_vm.sh temp_anneal         # Starts from Temperature Annealing
#   ./run_gpu_vm.sh mige                # Starts from standard MIGE
#   ./run_gpu_vm.sh mige2p              # Starts directly from MIGE-2P Two-Phase Sobolev
#   ./run_gpu_vm.sh eval                # Only updates comparison charts & reports
# ==============================================================================
set -e

START_STAGE="${1:-ste}"

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
# 1. Control 1: STE Baseline
# ------------------------------------------------------------------------------
if [ "$START_STAGE" = "ste" ]; then
    echo "--> [1/4] Starting Control 1: STE Baseline..."
    python train_phase_a.py --scale 125M --method ste --dataset "$DATASET_NAME" --subset "$SUBSET_NAME" --steps $STEPS --batch_size $BATCH_SIZE --seq_len $SEQ_LEN
    START_STAGE="temp_anneal"
fi

# ------------------------------------------------------------------------------
# 2. Control 2: Temperature Annealing
# ------------------------------------------------------------------------------
if [ "$START_STAGE" = "temp_anneal" ]; then
    echo "--> [2/4] Starting Control 2: Temperature Annealing..."
    python train_phase_a.py --scale 125M --method temp_anneal --dataset "$DATASET_NAME" --subset "$SUBSET_NAME" --steps $STEPS --batch_size $BATCH_SIZE --seq_len $SEQ_LEN
    START_STAGE="mige"
fi

# ------------------------------------------------------------------------------
# 3. Experiment 1: Standard MIGE
# ------------------------------------------------------------------------------
if [ "$START_STAGE" = "mige" ]; then
    echo "--> [3/4] Starting Experiment: MIGE (Standard)..."
    python train_phase_a.py --scale 125M --method mige --dataset "$DATASET_NAME" --subset "$SUBSET_NAME" --steps $STEPS --batch_size $BATCH_SIZE --seq_len $SEQ_LEN
    START_STAGE="mige2p"
fi

# ------------------------------------------------------------------------------
# 4. Experiment 2: MIGE-2P (Two-Phase Topological Sobolev Schedule)
# ------------------------------------------------------------------------------
if [ "$START_STAGE" = "mige2p" ] || [ "$START_STAGE" = "mige_2p" ]; then
    echo "--> [4/4] Starting Experiment: MIGE-2P (Two-Phase Topological Sobolev Schedule)..."
    python train_phase_a.py --scale 125M --method mige_2p --dataset "$DATASET_NAME" --subset "$SUBSET_NAME" --steps $STEPS --batch_size $BATCH_SIZE --seq_len $SEQ_LEN
fi

echo "=== Step 2: Phase B & Multi-Model Comparative Evaluation ==="
python update_comparison_charts.py

echo "=== Benchmark Complete. All Phase A & B Results Generated & Charts Updated ==="
