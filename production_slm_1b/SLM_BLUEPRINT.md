# Production 1B SLM: Architecture & Pipeline Blueprint

## 1. Executive Summary & Objective
Build an edge-deployable, production-ready **1 Billion Parameter Small Language Model (SLM)**:
- **Architecture**: 1.58-bit Ternary BitNet b1.58 (INT8 activations $\times$ INT2 weights).
- **Disk / RAM Footprint**: Compressed to **~260 MB** (fits entirely on phones, Raspberry Pi, or local laptops).
- **Phase 1 (Rapid Manifold Pre-training)**: Stream 25 to 50 Billion Tokens via Zero-Backprop Spectral-Laplacian Inversion at **> 1.1M tok/s** (~6-12 hours on 1x L40S) to absorb macro linguistic and factual priors.
- **Phase 2 (Dialogue / Instruct Tuning)**: High-resolution causal dialogue alignment on UltraChat / OpenAssistant / ShareGPT datasets to produce a responsive conversational assistant.

---

## 2. 1B Model Architecture Specification

| Parameter | 125M Workhorse | **1B Production SLM** |
| :--- | :---: | :---: |
| **Layers ($L$)** | 12 | **24** |
| **Hidden Dimension ($d_{\text{model}}$)** | 768 | **2048** |
| **Attention Heads ($h$)** | 12 ($d_k=64$) | **16 ($d_k=128$)** |
| **FFN Intermediate Dimension** | 2048 | **5632** (SwiGLU multiplier 2.75x) |
| **Vocabulary Size ($V$)** | 50,257 (GPT-2 BPE) | **50,257** |
| **Max Sequence Length** | 512 | **1024** |
| **Total Parameter Count** | 169 Million | **1,048,576,000 (~1.05 Billion)** |
| **Uncompressed FP32 Size** | 647 MB | **~4.19 GB** |
| **Packed 1.58-Bit Size** | **38.8 MB** | **~262 MB** |

---

## 3. Two-Stage Production Lifecycle

### Stage 1: Fast Manifold Pre-training (High-Volume Streaming)
- **Dataset**: `HuggingFaceFW/fineweb-edu` (sample-10BT or full stream).
- **Token Target**: 25 to 50 Billion Tokens.
- **Method**: Zero-Backpropagation Streaming Covariance Accumulation with Zelnik-Manor Graph Laplacian regularization.
- **Speed**: ~1,100,000 tokens/sec.
- **Time**: ~6.3 hours for 25BT, ~12.6 hours for 50BT.

### Stage 2: SFT / Dialogue Alignment (Instruction Tuning)
- **Dataset**: `HuggingFaceH4/ultrachat_200k` or `teknium/OpenHermes-2.5`.
- **Target**: 5,000 steps of low-rank or full BitNet STE tuning with ChatML format:
  ```text
  <|im_start|>system
  You are an efficient, helpful 1B SLM assistant.<|im_end|>
  <|im_start|>user
  {prompt}<|im_end|>
  <|im_start|>assistant
  {response}<|im_end|>
  ```
- **Time**: ~45 minutes on 1x L40S GPU.
