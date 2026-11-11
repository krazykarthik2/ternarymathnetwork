#!/bin/bash
# ==============================================================================
# GPU VM Master Launch Script: BitNet b1.58 Phase A & Phase B Benchmark
# Assumes PyTorch, CUDA, and Transformers are already installed on the VM.
# ==============================================================================
set -e

echo "=== Step 1: Verifying Existing GPU & Environment ==="
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device Count:', torch.cuda.device_count(), '| Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "=== Step 2: Launching Phase A Core Scale Benchmark (BitNet-125M) ==="
# 1. Control 1: Standard STE Baseline
echo "--> Starting Control 1: STE Baseline..."
python train_phase_a.py --scale 125M --method ste --dataset roneneldan/TinyStories --steps 10000 --batch_size 32 --seq_len 512

# 2. Control 2: Temperature Annealing
echo "--> Starting Control 2: Temperature Annealing..."
python train_phase_a.py --scale 125M --method temp_anneal --dataset roneneldan/TinyStories --steps 10000 --batch_size 32 --seq_len 512

# 3. Experiment: MIGE (Adaptive Sobolev)
echo "--> Starting Experiment: MIGE..."
python train_phase_a.py --scale 125M --method mige --dataset roneneldan/TinyStories --steps 10000 --batch_size 32 --seq_len 512

echo "=== Step 3: Launching Phase B Zero-Shot Evaluation (Silicon Proof) ==="
python evaluate_phase_b.py --checkpoint checkpoints/125M_ste/checkpoint_step_10000.pt --scale 125M
python evaluate_phase_b.py --checkpoint checkpoints/125M_temp_anneal/checkpoint_step_10000.pt --scale 125M
python evaluate_phase_b.py --checkpoint checkpoints/125M_mige/checkpoint_step_10000.pt --scale 125M

echo "=== Benchmark Complete. All Phase A & B Results Generated ==="
