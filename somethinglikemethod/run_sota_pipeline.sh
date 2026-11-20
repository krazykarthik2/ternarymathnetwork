#!/usr/bin/env bash
# ==============================================================================
# Complete SOTA Runner: Spectral Warm Inversion + Two-Phase Homotopy Sobolev (MIGE-HC / NBS)
# Automatically selects idle GPU, tunes for 3,000 steps (~12 minutes), and evaluates Phase B.
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

select_idle_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
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

STEPS="${1:-3000}"
BATCH_SIZE="${2:-32}"
CKPT_SPECTRAL="${3:-topology_bitnet_10bt.pt}"

echo "=================================================================="
echo " Launching SOTA Two-Phase Spectral-Sobolev Pipeline"
echo " CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo " Initial Checkpoint   : $CKPT_SPECTRAL (1.2B Token Spectral Inversion)"
echo " Tuning Steps         : $STEPS"
echo " Batch Size           : $BATCH_SIZE"
echo "=================================================================="

# 1. Run Two-Phase Tuning
python3 hybrid_spectral_mige.py \
    --spectral_ckpt "$CKPT_SPECTRAL" \
    --steps "$STEPS" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "checkpoints_spectral_mige"

# 2. Run Phase B Evaluation on Final Checkpoint
echo "--> Running Phase B downstream evaluation on final model..."
python3 ../hopfield/eval_checkpoint_phase_b.py \
    --checkpoint "checkpoints_spectral_mige/spectral_mige_final.pt" \
    --samples 250 \
    --output "phase_b_spectral_mige_results.json"

echo "=================================================================="
echo " SOTA Evaluation Complete! Results saved in phase_b_spectral_mige_results.json"
echo "=================================================================="
