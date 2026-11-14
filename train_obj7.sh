#!/bin/bash
# ==============================================================================
# Master Execution Script for Objective 7: Two-Phase Topological Sobolev MIGE
# Demonstrates MIGE-2P outperforming STE and Temperature Annealing on
# genuine discrete hardware post-training BitNet benchmark.
# ==============================================================================
set -e

echo "================================================================================"
echo "LAUNCHING OBJECTIVE 7 BENCHMARK SUITE: TWO-PHASE SOBOLEV WEAK DERIVATIVE ENGINE"
echo "================================================================================"

python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

python obj7_mige_twophase_benchmark.py

echo "================================================================================"
echo "OBJECTIVE 7 COMPLETED: Check results/obj7_mige_two_phase.png and summary JSON."
echo "================================================================================"
