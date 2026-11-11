# Project Rules & Invariants

## Core Principles
1. **No Dummy Code**: Every function, theorem verification, and model pass must execute real mathematical computation and produce empirical outputs.
2. **No Synthetic / Fake Data or Mocking**: Use genuine evaluation distributions, real tasks (e.g., real benchmark datasets or rigorous mathematical ground truths), and true loss / gradient telemetry.
3. **No Handholding**: Report empirical findings directly with raw data, metrics, standard deviations, and diagnostic logs.
4. **No Dishonesty / Zero Fabrication**: If an estimator fails, unstable dynamics occur, or an objective exhibits unexpected behavior, state it plainly with exact numerical evidence.
5. **Agnostic Hardware Execution with GPU/DirectML/CUDA Readiness**:
   - Write clean device-agnostic code: automatically detect CUDA if available, DirectML if operational, falling back smoothly to multi-threaded CPU without code modifications.
   - Code must be drop-in ready to run on high-performance CUDA VM / GPU clusters or local silicon.
