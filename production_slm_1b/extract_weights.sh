#!/usr/bin/env bash
# ==============================================================================
# extract_weights.sh: Convert 15GB training checkpoint into a compact weights file
# Usage:
#   ./extract_weights.sh <checkpoint_file> [optional_output_file]
# Examples:
#   ./extract_weights.sh checkpoints_parallel_sft/dialogue_sft_step_500.pt
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CKPT="${1:-checkpoints_parallel_sft/dialogue_sft_step_500.pt}"
OUTPUT="${2:-}"

if [ ! -f "$CKPT" ]; then
    echo "Error: Checkpoint file '$CKPT' not found!"
    exit 1
fi

if [ -n "$OUTPUT" ]; then
    python3 extract_weights.py "$CKPT" --output "$OUTPUT"
else
    python3 extract_weights.py "$CKPT"
fi
