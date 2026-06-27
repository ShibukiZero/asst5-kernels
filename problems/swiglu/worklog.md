# SwiGLU Work Log

## Problem

`out = Swish(xW + b) ⊙ (xV + c)`, `Swish(g) = g·sigmoid(β·g)`, β=1.

- `x`: `[batch, seq, in_features]` = `[256, 64, 2048]` (fp32). Flatten rows: **M = batch·seq = 16384**.
- `W`, `V`: `[in_features, hidden] = [2048, 4096]` (fp32). **K = 2048, H = 4096**.
- `out`: `[256, 64, 4096]` (fp32).
- Correctness tolerance: `allclose(rtol=1e-2, atol=1e-2)` — **loose ⇒ lower precision (TF32/BF16) is allowed**.

### Roofline (the target)

Two GEMMs `[M,K]×[K,H]` (x@W, x@V) + cheap elementwise (bias, swish, multiply).

| Quantity | Value |
|----------|-------|
| FLOPs | 2·(2·M·K·H) ≈ **550 GFLOP** |
| Bytes moved | read ~200 MB + write ~268 MB ≈ 470 MB |
| Arithmetic intensity | ~1170 FLOP/byte → **compute-bound** |

Compute roofline by precision (H100 SXM): FP32 (CUDA core ~67 TF/s) ≈ **8.2 ms**; TF32 TC (~495 TF/s) ≈ **1.1 ms**; BF16 TC (~990 TF/s) ≈ **0.56 ms**. (Memory roofline ≈ 470 MB / 3.35 TB/s ≈ 140 µs — far below, confirms compute-bound.)

Hardware: NVIDIA H100 80GB HBM3 (Nebius VM), CUDA 13, torch 2.12.1+cu130, ncu 2025.3.1.

---

## Optimization Entries

Each entry: how I thought about it, what I changed, the result, and the profiling. Failures recorded too.

### Entry 0 — Baseline (PyTorch reference)

Date: 2026-06-25

Code version: `submission.py` = `reference.ref_kernel` — `gate=x@W+b; swish=gate*sigmoid(β·gate); value=x@V+c; out=swish*value`.

Command: `./run.sh swiglu benchmark | profile`

Correctness: **pass**

Performance:
- **Runtime: 12.164 ms** (mean of 3).

Profiler stats (custom_kernel kernels; generate_input's randn excluded):

| Kernel | What | Duration | Compute% | DRAM% |
|--------|------|----------|----------|-------|
| `sm80_xmma_gemm_**f32f32_f32f32_f32**_…_**ffma**_…cublas` | GEMM x@W | 6.71 ms | **86.6%** | 10.9% |
| `sm80_xmma_gemm_f32f32…ffma…cublas` | GEMM x@V | 6.71 ms | 86.6% | 10.9% |
| elementwise ×5 (bias, sigmoid, swish, multiplies) | epilogue | ~1.3 ms total | 5–61% | 68–92% |

Observation — what limits performance:
1. **The two GEMMs are pure FP32 on CUDA cores — Tensor Cores are idle.** Kernel name `f32f32_f32f32_f32 … ffma` = FP32 FMA (no TF32/BF16); Compute 86.6% / DRAM 10.9% ⇒ **compute-bound on the FP32 path**. The two GEMMs are ~90% of the runtime.
2. **Epilogue is unfused and memory-bound** (DRAM 68–92%): bias/sigmoid/swish/multiply run as ~5 separate elementwise kernels, materializing `gate`/`value` (256 MB each) ⇒ extra DRAM traffic + launches (~1.3 ms).

Hypothesis:
- The tolerance (1e-2) allows lower precision, but the baseline runs FP32 on CUDA cores. Moving the GEMMs to **TF32 / BF16 Tensor Cores** should give ~7–14× on the GEMM (the dominant cost). Then **fuse** the two GEMMs (`x @ [W|V]`) and the epilogue to cut the ~1.3 ms elementwise traffic.

Next step: enable TF32 (and try BF16) for the matmuls; measure correctness + runtime.

---

### Entry 1 — Tensor Cores via precision (TF32 / BF16)

Date: 2026-06-25

Thinking: the 1e-2 tolerance allows lower precision, and the baseline GEMMs ran FP32 on CUDA cores. Switch the matmuls to Tensor Cores: TF32 via `set_float32_matmul_precision("high")`, BF16 via casting matmul inputs (fp32 epilogue). Env-switchable (`SG_PREC`).

Code version: `submission.py` (default `tf32`).

Results:

| Precision | Correctness | Runtime | vs baseline |
|-----------|-------------|---------|-------------|
| fp32 (control) | pass | 12.17 ms | 1× |
| **tf32** | **pass** | **2.94 ms** | **4.1×** |
| bf16 | **FAIL** | — | exceeds tolerance |

Profiler stats (TF32 path):
- GEMMs: `sm90_xmma_gemm_f32f32_**tf32f32**_f32 … warpgroupsize … cublas` — **TF32 Tensor Cores (Hopper wgmma)**, **977 µs each** (baseline FP32 was 6.71 ms → **6.9×**), Compute 73% / DRAM 28%. The two GEMMs (~1.95 ms) are now the dominant cost.
- Epilogue: elementwise kernels ~227 µs each, DRAM ~68% (memory-bound, **still unfused**).

Observation:
- **TF32 Tensor Cores: 4.1× and correct.** TF32 keeps ~10 mantissa bits (rel err ~1e-3) → comfortably within rtol=1e-2.
- **BF16 fails correctness.** 8 mantissa bits (~4e-3/elem) accumulated over K=2048 pushes the result past 1e-2. (Could be rescued with bf16×3 / error correction, but not worth it — TF32 is the sweet spot.)
- New balance: with TF32 the two GEMMs drop to ~1.5 ms, so the **unfused elementwise epilogue (~1.3 ms) is now ~44% of runtime** — the next target. The two GEMMs are also still separate launches over the same `x`.

Next step: fuse — (a) the epilogue (bias + swish + multiply) into one pass instead of ~5 elementwise kernels materializing `gate`/`value`; (b) the two GEMMs into one `x @ [W|V]`. Consider a Triton fused matmul+epilogue (Triton's sweet spot).

---

### Entry 2 — PyTorch-level fusion (TF32 + SiLU + torch.compile)

Date: 2026-06-25

Thinking: after TF32, the two GEMMs are ~1.5 ms and the unfused elementwise epilogue (~1.3 ms) dominates the rest. Fuse it at the PyTorch level: (a) `F.silu` (β=1 ⇒ swish == silu) collapses sigmoid+multiply into one kernel; (b) `torch.compile` lets Inductor auto-fuse the whole epilogue; (c) fuse the two GEMMs into `x @ [W|V]`.

Results (all TF32):

| Variant | Correctness | Runtime | vs Entry 1 (2.94 ms) |
|---------|-------------|---------|----------------------|
| `F.silu` (fused swish, separate GEMMs) | pass | 2.53 ms | 1.16× |
| **`torch.compile`** (Inductor fuses epilogue) | pass | **2.036 ms** | **1.45×** |
| concat GEMM `x @ [W\|V]` | **FAIL** | — | dropped |

Profiler stats (torch.compile):
- GEMMs: the same two TF32 cuBLAS kernels (977 / 983 µs) — Inductor keeps cuBLAS for the matmuls.
- Epilogue: collapsed into **one** Inductor-generated Triton kernel `triton_poi_fused__unsafe_view_add_mul_silu_0`, **256 µs**, DRAM 91.6% (memory-bound). The ~5 separate elementwise kernels are gone.
- ⇒ **2.036 ms ≈ 1.95 ms (two GEMMs, ~96%) + 0.26 ms (fused epilogue).** The epilogue is now essentially solved; the two TF32 GEMMs are the remaining ~96%.

Observation:
- `F.silu` alone: 2.94→2.53 ms (one fused kernel instead of sigmoid + multiply).
- **`torch.compile` is the best PyTorch-level result, 2.036 ms (~6× over baseline)** — Inductor fuses the bias/silu/multiply epilogue (auto-generated Triton), removing the `gate`/`value` materialization.
- **concat GEMM failed correctness — but it's not a bug.** Debugging showed exactly **1 element / 67 M** exceeds tol (max abs diff 0.146 on outputs up to ~32 000). Cause: `check_implementation` compares against the reference's **two separate** TF32 GEMMs; concatenating into one GEMM makes cuBLAS pick a **different algorithm**, diverging the TF32 rounding on one near-zero output (diff 0.146 > atol 0.01). Mathematically correct, but it leaves the reference's exact numerical path. Also not faster than `compile` ⇒ dropped. **Lesson: a "correct" reformulation can trip a strict tolerance test if it changes the numerical path relative to the reference; keep the same GEMM structure as the reference.**

Next step (Entry 3): hand-written **Triton fused matmul + epilogue**. Profiling refines the goal — the epilogue is already a single fused kernel, so the only remaining lever is the **two GEMMs (~96%)**: a fused kernel that computes `gate=x@W` and `value=x@V` in one pass (load each `x` tile once, feed both accumulators) and applies the epilogue inline (no `gate`/`value` write-back). High bar — must beat cuBLAS TF32 (already at 73% compute).
