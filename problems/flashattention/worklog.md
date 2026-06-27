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

---

### Entry 2 — Hand-written Triton FlashAttention (FA-2 style)

Date: 2026-06-27

Thinking: implement real flash attention — tile Q (BLOCK_M rows) × K/V (BLOCK_N cols), keep the running softmax stats `m` (row max) and `l` (exp-sum) per query row, never materialize S×S. Standard FA-2 tricks: fold `sm_scale·log2(e)` into Q and use `exp2` (HW instruction); rescale the accumulator by `α = exp2(m_old − m_new)` each K/V block. No boundary mask (all benchmark seq_lens are multiples of 128).

Code version: `versions/v2_triton.py` (= `submission.py`).

Correctness: **pass** all 3 cases (small max abs diff 0.001 — FP32 online-softmax accumulators, well within 1e-2).

Tile-size sweep (large case, correctness checked on small):

| (BLOCK_M, BLOCK_N, warps, stages) | time |
|-----------------------------------|------|
| **(128, 128, 8, 3)** | **18.28 ms** ✅ best |
| (64, 64, 4, 3) | 19.61 |
| (128, 64, 8, 4) | 20.45 |
| (128, 64, 8, 3) | 20.58 |
| (64, 128, 4, 3) | 21.84 |
| (128, 128, 4, 3) | 26.13 |
| (128, 32, 4, 4) | 25.57 |
| (64, 64, 8, 3) | 37.49 |
| (128, 128, 8, 4) | OOM (shmem 294 KB > 228 KB) |

Best config harness numbers:

| case | Triton | vs baseline | vs cuDNN (E1) |
|------|--------|-------------|---------------|
| small | 0.094 ms | 3.7× | (cuDNN 0.063) |
| medium | 2.247 ms | 5.6× | (cuDNN 1.632) |
| **large** | **18.86 ms** (best 17.07) | **6.1×** | 1.48× slower (cuDNN 12.76) |

Profiling (ncu, best config, large; ncu duration replay-inflated to 21.4 ms vs real 18.9):

| metric | value |
|--------|-------|
| **DRAM Throughput** | **3.0 %** |
| Compute (SM) Throughput | 52.2 % |
| Memory Throughput | 37.4 % |
| Achieved Occupancy | 12.5 % (theoretical 12.5 %, **register-limited**) |
| top warp stall | waiting at CTA barrier (sibling warps) |

Observation:
- **The flash-attention win is real and confirmed by ncu: DRAM is only 3 %** (vs the baseline which was memory-bound at ~2.5 TB/s streaming the 34 GB S×S). The O(N²) memory wall is gone; the kernel is now compute-bound. Peak memory ~few GB (no S×S).
- **18.86 ms ≈ 467 TF/s ≈ 47 % MFU.** It beats the baseline 6.1× and **beats PyTorch's own FA-2 "flash" backend (24.6 ms)**, but loses to **cuDNN (12.76 ms, ~70 % MFU)** by 1.48×.
- **Why the gap to cuDNN:** SM throughput is only 52 % and occupancy 12.5 %, **limited by registers** (the `[128,128]` FP32 accumulator + Q tile). cuDNN on Hopper is FA-3-class — **warp-specialized** (separate producer/consumer warpgroups), **TMA** async loads, and software-pipelined MMA/softmax — which keep the tensor cores fed at ~70 %. Our single-program Triton kernel can't express that scheduling, so it stalls at barriers and under-occupies.

Hypothesis / next: the gap is compute-scheduling, not memory. Levers that *might* close some of it (diminishing returns): smaller `BLOCK_M` to cut register pressure / raise occupancy; `tl.dot` with FP8 (accuracy risk); Triton's newer warp-specialization / TMA pipelining (`tl.async`); or a `triton.autotune` over a wider grid. Matching cuDNN would essentially mean re-implementing FA-3 (CUTLASS/CuTe) — a large undertaking. **Conclusion: kept Triton (18.86 ms) as the hand-written best; it's the real learning artifact (correct flash attention, memory wall eliminated), within ~1.5× of FA-3-class cuDNN.**

Lessons: (1) Flash attention's value is **IO**, and ncu proves it — DRAM 3 % vs a memory-bound baseline; the whole 115→19 ms win is from not touching S×S. (2) Hand-Triton flash is genuinely good (beats FA-2 backend) — FA is Triton's sweet spot — but the last ~1.5× to FA-3/cuDNN needs Hopper warp-specialization/TMA that Triton doesn't fully expose. (3) Online softmax with FP32 accumulators is *more* accurate than the FP16 reference → correctness is easy here (unlike swiglu's TF32 trap). (4) Best tile was the largest that fits shmem (128×128); pushing num_stages to 4 OOMs shared memory — the classic flash-attention occupancy/shmem tension.
