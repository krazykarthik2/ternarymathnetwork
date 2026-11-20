#!/usr/bin/env bash
# ==============================================================================
# GPU VM Multi-Run Benchmark Script: Hopfield-BitNet vs Standard BitNet baselines
# Compares:
# 1. Hopfield-BitNet-125M with Single-Shot initialization + MIGE-TA
# 2. Hopfield-BitNet-125M with Single-Shot initialization + STE
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

STEPS="${1:-2500}"
BATCH_SIZE="${2:-16}"

echo "=================================================================="
echo " Running Comparative GPU Benchmark: Hopfield-BitNet (MIGE-TA vs STE)"
echo " Steps per run: $STEPS | Batch size: $BATCH_SIZE"
echo "=================================================================="

echo "--> [1/2] Training Hopfield-BitNet-125M with MIGE-TA..."
python3 train_hopfield_125m.py \
    --method mige_ta \
    --steps "$STEPS" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "checkpoints_mige_ta"

echo "--> [2/2] Training Hopfield-BitNet-125M with STE..."
python3 train_hopfield_125m.py \
    --method ste \
    --steps "$STEPS" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "checkpoints_ste"

echo "=================================================================="
echo " Benchmark suite completed! Comparison logs written to checkpoints_*/"
echo "=================================================================="
