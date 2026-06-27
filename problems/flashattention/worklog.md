# FlashAttention Work Log

## Problem

Optimize exact attention for the H100 leaderboard.

Key workload:
- Inputs: `Q`, `K`, `V`, each `[batch_size, num_heads, seq_len, head_dim]`
- Output: `[batch_size, num_heads, seq_len, head_dim]`
- Benchmark cases:
  - `batch_size=1; num_heads=64; seq_len=1024; head_dim=128`
  - `batch_size=2; num_heads=64; seq_len=4096; head_dim=128`
  - `batch_size=4; num_heads=64; seq_len=8192; head_dim=128`

## Optimization Entries

Append one entry per meaningful experiment.

### Roofline (the target)

Per (b,h): two matmuls, each `2·S²·D` FLOP, + a softmax over an S×S matrix.
Large case (B=4, N=64, S=8192, D=128): BN=256 (b,h) pairs.

| Quantity | Value |
|----------|-------|
| Total FLOPs (2 GEMMs) | 2·(2·B·N·S²·D) ≈ **8.8 TFLOP** |
| Compute floor (FP16 TC ~990 TF/s) | **~9 ms** |
| S×S scores matrix per `[B,N,S,S]` (fp16) | **34 GB** (×2 with probs ⇒ 68 GB resident) |
| Flash-attention HBM (Q+K+V+O only) | **~2.15 GB** ⇒ ~0.6 ms at 3.35 TB/s |

⇒ exact attention is ~9 ms of compute. The baseline pays ~13× that because it **materializes and re-streams the 34 GB S×S matrix** (the O(N²) memory wall). Flash-attention's job: never write S×S, approach the ~9 ms floor.

Hardware: H100 80GB HBM3 (Nebius VM), CUDA 13, torch 2.12. FP16 inputs, tol `rtol=atol=1e-2`.

---

### Entry 0 — Baseline (PyTorch reference, naive 3-op attention)

Date: 2026-06-27

Code version: `submission.py` = `reference.ref_kernel` — `scores = q@kᵀ/√d; p = softmax(scores); out = p@v`. Materializes the full `[B,N,S,S]` scores + probs.

Command: `./run.sh flashattention benchmark`

Correctness: **pass** (custom == reference).

Performance (harness benchmark):

| case (B,N,S,D) | runtime |
|----------------|---------|
| small (1,64,1024,128) | 0.344 ms |
| medium (2,64,4096,128) | 12.685 ms |
| **large (4,64,8192,128)** | **115.04 ms** |

Profiling — stage breakdown on the large case (CUDA events; matches harness full() = 114.87 ms):

| stage | time | what limits it |
|-------|------|----------------|
| `QK^T` GEMM (+scale) | 13.5 ms | **memory-bound**: writes 34 GB scores @ ~2.5 TB/s (only 326 TF/s of 990 peak) |
| softmax | **66.9 ms** | **memory-bound**: reads+writes ~69 GB @ only **1.03 TB/s** (wide-row softmax, multiple passes) |
| `PV` GEMM | 14.0 ms | **memory-bound**: reads 34 GB probs @ ~2.45 TB/s |
| scale/elementwise + overhead | ~20 ms | extra 34 GB passes |
| peak memory | **71 GB** | 34 GB scores + 34 GB probs resident (barely fits 80 GB; larger ⇒ OOM) |

Observation — what limits performance:
- **The two GEMMs would be ~9 ms at peak; the baseline is 115 ms (~13×).** Almost none of it is real compute — both GEMMs run at ~2.5 TB/s (memory-bound, not compute-bound) because they write/read the 34 GB S×S matrix, and **softmax alone is 67 ms (58%)** streaming 69 GB at a poor 1.03 TB/s.
- This is the textbook **O(N²) memory wall**: ~106 of the 115 ms is moving the S×S scores/probs through HBM, not computing.
- Note (process): my first profiling pass was wrong — I called `generate_input(B,N,S,D)` positionally, but its signature is `(batch, heads, head_dim, seq_len, seed)` (head_dim before seq_len), so I accidentally profiled S=128/D=8192 and got an impossible 0.76 ms. The harness (keyword args) was right. Fixed by calling with keywords; numbers above are correct.

Hypothesis:
- **FlashAttention**: tile Q/K/V, compute attention block-by-block in SRAM with **online softmax**, never materializing S×S. HBM traffic drops from ~34 GB×(several passes) to ~2.15 GB (Q/K/V/O) ⇒ becomes compute-bound, should approach the ~9 ms floor. PyTorch's `F.scaled_dot_product_attention` (cuDNN/flash) is the library reference (README: ~28 ms large, ~4.5×).

Next step (Entry 1): drop-in `F.scaled_dot_product_attention(q,k,v)` — the library FlashAttention; measure the ~4.5× and set the bar a hand-written kernel must beat.

---

### Entry 1 — `F.scaled_dot_product_attention` (library FlashAttention)

Date: 2026-06-27

Thinking: the win is killing the S×S materialization (Entry 0). PyTorch's SDPA already wraps fused FlashAttention backends; its defaults match the reference exactly (`scale = 1/√d`, `is_causal=False`). One-liner.

Code version: `versions/v1_sdpa.py` (= `submission.py`): `return F.scaled_dot_product_attention(q, k, v)`.

Correctness: **pass** (all 3 cases).

Performance (harness benchmark):

| case (B,N,S,D) | baseline | SDPA | speedup |
|----------------|----------|------|---------|
| small (1,64,1024,128) | 0.344 ms | 0.063 ms | 5.5× |
| medium (2,64,4096,128) | 12.685 ms | 1.632 ms | 7.8× |
| **large (4,64,8192,128)** | **115.04 ms** | **12.76 ms** | **9.0×** |

Backend comparison (large, `torch.nn.attention.sdpa_kernel`, my-loop timing):

| backend | time | note |
|---------|------|------|
| **cuDNN** | **14.1 ms** | default dispatcher picks this ⇒ harness 12.76 ms |
| Flash (FA-2) | 24.6 ms | the classic flash-attn; matches README's "28 ms" |
| mem-efficient | 47.9 ms | xformers-style |

Observation:
- **SDPA (cuDNN) = 12.76 ms, 9× over baseline.** Achieves 8.8 TFLOP / 12.76 ms ≈ **690 TF/s (~70 % of FP16 peak)** — i.e. it's now **compute-bound and near the ~9 ms floor**, exactly as the roofline predicted once the S×S traffic is gone. Peak memory drops from 71 GB to a few GB (no S×S).
- The default backend is **cuDNN**, which is ~2× faster than the bundled FA-2 "flash" backend (14 vs 24.6 ms) — on Hopper cuDNN is FA-3-class (wgmma + TMA + warp-specialization). The README's 28 ms was the FA-2 backend; cuDNN moved the bar much lower.

Hypothesis / next: a hand-written Triton FlashAttention (online softmax + tiling) is the learning goal. **Bar to beat: cuDNN's ~12.76 ms** (≈70 % MFU) — a high bar (same lesson as swiglu: the library is near-peak). Realistic aim: learn the algorithm and get within ~1.5–2× of cuDNN; matching/beating it would need FA-3-level Hopper engineering (CUTLASS/CuTe). Correctness regime is friendly (FP16, tol 1e-2, FP32 online-softmax accumulators are *more* accurate than the reference), unlike swiglu's TF32 self-reference trap.

Next step (Entry 2): hand-written Triton FlashAttention; measure vs the 12.76 ms cuDNN bar.
