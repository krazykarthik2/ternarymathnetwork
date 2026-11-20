#!/usr/bin/env bash
# ==============================================================================
# GPU VM Master Launch Script: Hopfield-BitNet b1.58 (125M Architecture)
# Tailored for NVIDIA L40S (48GB VRAM) or standard enterprise GPUs.
# ==============================================================================
set -e

# Change directory strictly to the hopfield folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo " Starting Hopfield-BitNet-125M Execution on GPU VM"
echo " Working directory: $(pwd)"
echo "=================================================================="

# Check Python and CUDA environment
echo "--> Checking CUDA device status..."
python3 -c "
import torch
print('PyTorch Version :', torch.__version__)
print('CUDA Available  :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Name     :', torch.cuda.get_device_name(0))
    print('Device Count    :', torch.cuda.device_count())
    print('VRAM Available  :', f'{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')
"

# Parse launch arguments
METHOD="${1:-mige_ta}"      # Options: mige_ta, ste, discrete
STEPS="${2:-10000}"         # Target steps
BATCH_SIZE="${3:-16}"       # Optimal batch size for L40S 48GB (can increase to 32 or 64)
SEQ_LEN="${4:-512}"
DATASET="${5:-HuggingFaceFW/fineweb-edu}"
SUBSET="${6:-sample-10BT}"

echo "=================================================================="
echo " Configuration:"
echo " - Method           : $METHOD"
echo " - Steps            : $STEPS"
echo " - Batch Size       : $BATCH_SIZE"
echo " - Sequence Length  : $SEQ_LEN"
echo " - Dataset / Subset : $DATASET ($SUBSET)"
echo "=================================================================="

# 1. Quick Unit Test & Validation of Closed-Form Single-Shot Calculation
echo "--> [Step 1/2] Running analytical verification and zero-shot benchmark..."
python3 train_and_eval.py

# 2. Launching 125M Training Pipeline
echo "--> [Step 2/2] Launching 125M Causal Pretraining on GPU..."
python3 train_hopfield_125m.py \
    --method "$METHOD" \
    --steps "$STEPS" \
    --batch_size "$BATCH_SIZE" \
    --seq_len "$SEQ_LEN" \
    --dataset "$DATASET" \
    --subset "$SUBSET" \
    --output_dir "checkpoints"

echo "=================================================================="
echo " Hopfield-BitNet-125M Training Finished Successfully!"
echo " Checkpoints stored in: $(pwd)/checkpoints"
echo "=================================================================="
