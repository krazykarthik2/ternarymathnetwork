# TernaryMathNetwork: 1.05B BitNet b1.58 SLM

A production implementation of a **1.05 Billion Parameter Small Language Model (SLM)** utilizing:
1. **Zero-Backpropagation Graph Laplacian Spectral Inversion** for high-volume 50 Billion Token (50BT) semantic pre-training ($>1.3\times 10^6\text{ tok/s}$).
2. **Warm-Start Representation Transfer (5B Tokens)** for fast causal attention routing and full English conversational fluency.
3. **Hardware-Compatible 1.58-Bit Ternary Quantization $\{-1, 0, +1\}$** compressed into a **~262 MB** single binary for instant edge and local CPU deployment.

---

## Repository Structure

```
ternarymathnetwork/
├── production_slm_1b/
│   ├── ARCHITECTURE_AND_MATH.md     # Full architectural diagrams & mathematical proofs
│   ├── bitnet_1b_model.py           # 1.05B parameter BitNet b1.58 model architecture
│   ├── train_1b_spectral_stream.py  # Stage 1: 50BT Zero-Backprop Graph Laplacian streaming engine
│   ├── train_1b_warmstart_5b.py     # Stage 2: 5BT Warm-Start Autoregressive Causal Attention engine
│   ├── finetune_dialogue_direct.py  # Stage 3: ChatML conversational instruction fine-tuning
│   ├── extract_weights.sh           # Converts 15GB recovery checkpoints into lightweight weights
│   ├── chat_local_1b.py             # Interactive local CPU chat terminal for packed 1.58-bit models
│   └── run_1b_pipeline.sh           # Unified master execution runner
├── hopfield/
│   ├── pack_checkpoint_158bit.py    # 1.58-bit ternary compression (4 weights/uint8 byte)
│   ├── unpack_and_eval_local.py     # Decompression engine for CPU inference
│   └── train_hopfield_torch_maxspeed.py # Fast streaming dataloader with instant shard skipping
└── quantizers.py                    # Hardware-accurate ternary projection functions
```

---

## Pipeline Execution Guide

### 1. Stage 1: High-Volume 50BT Spectral Pre-training
Streams FineWeb-Edu at $>1.3\text{M tok/s}$, accumulating token state covariances and solving optimal weights in closed form:
```bash
cd production_slm_1b
python3 train_1b_spectral_stream.py --tokens 50000000000 --batch_size 128
```

### 2. Stage 2: 5B Token Warm-Start Causal Training (Fluent Conversational English)
Loads the 2048-dim vocabulary manifold from Stage 1 and trains multi-head attention routing:
```bash
python3 train_1b_warmstart_5b.py \
    --spectral_ckpt "checkpoints_1b_stream/spectral_ckpt_12501009408_tokens.pt" \
    --tokens 5000000000 \
    --batch_size 8 \
    --output "bitnet_1b_fluent_5b.pt"
```

### 3. Stage 3: ChatML Dialogue Alignment (Optional SFT)
Aligns conversational instructions on multi-turn ChatML datasets:
```bash
python3 finetune_dialogue_direct.py \
    --checkpoint "bitnet_1b_fluent_5b.pt" \
    --steps 3000 \
    --batch_size 4
```

### 4. Stage 4: 1.58-Bit Edge Compression & Local Chat
Compresses the 4.2 GB FP32 model into a **262 MB** single-file binary:
```bash
python3 -c "
import sys; sys.path.insert(0, '../hopfield')
from pack_checkpoint_158bit import pack_checkpoint_158bit
pack_checkpoint_158bit('bitnet_1b_fluent_5b.pt', 'bitnet_1b_packed_edge.pt')
"

# Interactive local CPU chat
python3 chat_local_1b.py --checkpoint bitnet_1b_packed_edge.pt
```
