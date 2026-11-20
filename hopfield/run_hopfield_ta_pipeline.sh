#!/usr/bin/env bash
# ==============================================================================
# GPU Launch Script: Hopfield Init + Pure Temperature Annealing (TA / MIGE-TA)
# Automatically detects and selects the idle GPU with the lowest memory usage.
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

METHOD="${1:-mige_ta}"       # Options: mige_ta (Mollified TA) or temp_anneal (Vanilla TA)
STEPS="${2:-2500}"          # 2000-2500 steps for fast convergence from Hopfield init
BATCH_SIZE="${3:-32}"       # 32 fits comfortably in L40S 48GB VRAM
EVAL_SAMPLES="${4:-250}"

echo "=================================================================="
echo " Detecting available GPUs and finding the most idle device..."
echo "=================================================================="

# Function to auto-select the GPU with the lowest memory usage
select_idle_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "[!] nvidia-smi not found. Defaulting to CUDA_VISIBLE_DEVICES=0"
        export CUDA_VISIBLE_DEVICES=0
        return
    fi

    # Query index and memory.used in MiB, sort ascending by used memory
    BEST_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -n 1 | awk -F',' '{print $1}' | tr -d ' ')
    USED_MEM=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -n 1 | awk -F',' '{print $2}' | tr -d ' ')

    if [ -n "$BEST_GPU" ]; then
        echo "--> Selected idle GPU index: $BEST_GPU (Current memory used: ${USED_MEM} MiB)"
        export CUDA_VISIBLE_DEVICES="$BEST_GPU"
    else
        echo "[!] Could not parse nvidia-smi output. Defaulting to CUDA_VISIBLE_DEVICES=0"
        export CUDA_VISIBLE_DEVICES=0
    fi
}

select_idle_gpu

echo "=================================================================="
echo " Starting Hopfield Init + Pure $METHOD Pipeline"
echo " CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo " Method               : $METHOD"
echo " Steps                : $STEPS"
echo " Batch Size           : $BATCH_SIZE"
echo "=================================================================="

# 1. Train with Hopfield Init + Pure TA
python3 train_hopfield_ta_fast.py \
    --method "$METHOD" \
    --steps "$STEPS" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "checkpoints_hopfield_ta"

# 2. Directly evaluate on Phase B
echo "--> Running Phase B downstream evaluation on the resulting checkpoint..."
python3 evaluate_hopfield_phase_b.py \
    --checkpoint "checkpoints_hopfield_ta/hopfield_ta_final.pt" \
    --samples "$EVAL_SAMPLES" \
    --output "phase_b_hopfield_ta_results.json"

echo "=================================================================="
echo " Pipeline Complete! Results saved in phase_b_hopfield_ta_results.json"
echo "=================================================================="
