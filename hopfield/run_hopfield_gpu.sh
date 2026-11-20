#!/usr/bin/env bash
# ==============================================================================
# GPU VM Master Launch Script: Hopfield-BitNet b1.58 (125M Architecture)
# Tailored for NVIDIA L40S. Auto-selects the idle GPU if another is busy.
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

select_idle_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "[!] nvidia-smi not found. Defaulting to CUDA_VISIBLE_DEVICES=0"
        export CUDA_VISIBLE_DEVICES=0
        return
    fi
    BEST_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -n 1 | awk -F',' '{print $1}' | tr -d ' ')
    USED_MEM=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -n 1 | awk -F',' '{print $2}' | tr -d ' ')
    if [ -n "$BEST_GPU" ]; then
        echo "--> Selected idle GPU index: $BEST_GPU (Current memory used: ${USED_MEM} MiB)"
        export CUDA_VISIBLE_DEVICES="$BEST_GPU"
    else
        export CUDA_VISIBLE_DEVICES=0
    fi
}

select_idle_gpu

echo "=================================================================="
echo " Starting Hopfield-BitNet-125M Execution on GPU VM"
echo " CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
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
BATCH_SIZE="${3:-32}"       # Optimal batch size for L40S 48GB
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
