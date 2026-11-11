#!/bin/bash
# ==============================================================================
# GPU VM Master Launch Script: BitNet b1.58 Phase A & Phase B Benchmark
# Uses the pre-packaged 10B token subset: 'sample-10BT'
# - Streaming mode: loads token batches directly into GPU VRAM/RAM
# - Does NOT download the 5.8 TB full dump. Zero storage burden.
# ==============================================================================
set -e

echo "=== Step 1: Verifying GPU & Python Environment ==="
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device Count:', torch.cuda.device_count(), '| Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Lightweight, high-quality reasoning sample: 10 Billion tokens
DATASET_NAME="HuggingFaceFW/fineweb-edu"
SUBSET_NAME="sample-10BT"
STEPS=10000
BATCH_SIZE=32
SEQ_LEN=512

echo "=== Step 2: Phase A Pre-Training (BitNet-125M) via Token Streaming ==="
echo "Target: $DATASET_NAME (subset: $SUBSET_NAME) [Zero-disk Streaming]"

# 1. Control 1: Standard BitNet b1.58 with STE
echo "--> [1/3] Starting Control 1: STE Baseline..."
python train_phase_a.py --scale 125M --method ste --dataset "$DATASET_NAME" --subset "$SUBSET_NAME" --steps $STEPS --batch_size $BATCH_SIZE --seq_len $SEQ_LEN

# 2. Control 2: Temperature Annealing
echo "--> [2/3] Starting Control 2: Temperature Annealing..."
python train_phase_a.py --scale 125M --method temp_anneal --dataset "$DATASET_NAME" --subset "$SUBSET_NAME" --steps $STEPS --batch_size $BATCH_SIZE --seq_len $SEQ_LEN

# 3. Experiment: MIGE (Adaptive Two-Phase Sobolev Weak Derivatives)
echo "--> [3/3] Starting Experiment: MIGE..."
python train_phase_a.py --scale 125M --method mige --dataset "$DATASET_NAME" --subset "$SUBSET_NAME" --steps $STEPS --batch_size $BATCH_SIZE --seq_len $SEQ_LEN

echo "=== Step 3: Phase B True Downstream Zero-Shot Reasoning Evaluation ==="
echo "--> Evaluating STE Checkpoint..."
python evaluate_phase_b.py --checkpoint checkpoints/125M_ste/checkpoint_step_${STEPS}.pt --scale 125M

echo "--> Evaluating Temp Anneal Checkpoint..."
python evaluate_phase_b.py --checkpoint checkpoints/125M_temp_anneal/checkpoint_step_${STEPS}.pt --scale 125M

echo "--> Evaluating MIGE Checkpoint..."
python evaluate_phase_b.py --checkpoint checkpoints/125M_mige/checkpoint_step_${STEPS}.pt --scale 125M

echo "=== Benchmark Complete. All Phase A & B Results Generated ==="
