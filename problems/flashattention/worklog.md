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
- Note (process): the first profiling pass was wrong — `generate_input(B,N,S,D)` was called positionally, but its signature is `(batch, heads, head_dim, seq_len, seed)` (head_dim before seq_len), so it accidentally profiled S=128/D=8192 and got an impossible 0.76 ms. The harness (keyword args) was right. Fixed by calling with keywords; numbers above are correct.

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

Backend comparison (large, `torch.nn.attention.sdpa_kernel`, manual-loop timing):

| backend | time | note |
|---------|------|------|
| **cuDNN** | **14.1 ms** | default dispatcher picks this ⇒ harness 12.76 ms |
| Flash (FA-2) | 24.6 ms | the classic flash-attn; matches README's "28 ms" |
| mem-efficient | 47.9 ms | xformers-style |

Observation:
- **SDPA (cuDNN) = 12.76 ms, 9× over baseline.** Achieves 8.8 TFLOP / 12.76 ms ≈ **690 TF/s (~70 % of FP16 peak)** — i.e. it's now **compute-bound and near the ~9 ms floor**, exactly as the roofline predicted once the S×S traffic is gone. Peak memory drops from 71 GB to a few GB (no S×S).
- The default backend is **cuDNN**, which is ~2× faster than the bundled FA-2 "flash" backend (14 vs 24.6 ms) — on Hopper cuDNN is FA-3-class (wgmma + TMA + warp-specialization). The README's 28 ms was the FA-2 backend; cuDNN moved the bar much lower.

Profiling (ncu): the cuDNN kernel is `cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_…` — **Compute(SM) 76.9 %, DRAM 4.4 %, L2 50 %**. Confirms it's an sm90 flash-attention wgmma kernel (FA-3 lineage), compute-bound, S×S traffic gone. Note: this **76.9 % SM matches the hand CUTLASS FA-3's 76.2 %** (Entry 4) — hardware-level parity, not just wall-clock.

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
- **Why the gap to cuDNN:** SM throughput is only 52 % and occupancy 12.5 %, **limited by registers** (the `[128,128]` FP32 accumulator + Q tile). cuDNN on Hopper is FA-3-class — **warp-specialized** (separate producer/consumer warpgroups), **TMA** async loads, and software-pipelined MMA/softmax — which keep the tensor cores fed at ~70 %. The single-program Triton kernel can't express that scheduling, so it stalls at barriers and under-occupies.

Hypothesis / next: the gap is compute-scheduling, not memory. Levers that *might* close some of it (diminishing returns): smaller `BLOCK_M` to cut register pressure / raise occupancy; `tl.dot` with FP8 (accuracy risk); Triton's newer warp-specialization / TMA pipelining (`tl.async`); or a `triton.autotune` over a wider grid. Matching cuDNN would essentially mean re-implementing FA-3 (CUTLASS/CuTe) — a large undertaking. **Conclusion: kept Triton (18.86 ms) as the hand-written best; it's the real learning artifact (correct flash attention, memory wall eliminated), within ~1.5× of FA-3-class cuDNN.**

Lessons: (1) Flash attention's value is **IO**, and ncu proves it — DRAM 3 % vs a memory-bound baseline; the whole 115→19 ms win is from not touching S×S. (2) Hand-Triton flash is genuinely good (beats FA-2 backend) — FA is Triton's sweet spot — but the last ~1.5× to FA-3/cuDNN needs Hopper warp-specialization/TMA that Triton doesn't fully expose. (3) Online softmax with FP32 accumulators is *more* accurate than the FP16 reference → correctness is easy here (unlike swiglu's TF32 trap). (4) Best tile was the largest that fits shmem (128×128); pushing num_stages to 4 OOMs shared memory — the classic flash-attention occupancy/shmem tension.

---

### Entry 3 — Trying to close the gap to cuDNN (warp-spec + occupancy tuning) — NO GAIN

Date: 2026-06-27

Thinking: Entry 2's ncu said the kernel is compute-bound at only 52 % SM, occupancy 12.5 % (register-limited), stalling at CTA barriers — i.e. MMA (tensor cores) and softmax (CUDA cores) don't overlap. The textbook fix is FA-3-style **warp specialization** (producer/consumer warpgroups ping-pong MMA ↔ softmax). Triton 3.7.1 exposes this as a one-line loop hint `tl.range(..., warp_specialize=True)`, plus occupancy knobs (`num_warps`, `maxnreg`). Try them.

Code version: experiments in `versions/v2_triton.py` parameter space (not a new submission; nothing beat Entry 2).

Results (large case, best tile BLOCK_M=BLOCK_N=128 unless noted; cuDNN bar = 12.76 ms, Entry-2 Triton = 18.0 ms):

| change | time | verdict |
|--------|------|---------|
| `warp_specialize=True` (8 warps, 3 stages) | 18.1 ms | **no change** vs 18.0 |
| `warp_specialize=True` + 4 warps | compile FAIL | `NVGPUWarpSpecialization` MLIR pass fails |
| `warp_specialize=True` + num_stages=4 | OOM shmem | 294 KB > 228 KB |
| num_warps=16 | 36.8 ms | much worse (warp contention) |
| num_warps=16, stages=2 | 37.9 ms | worse |
| maxnreg=128 | 203.7 ms | catastrophic register spill |
| maxnreg=160 | 22.8 ms | worse (spill) |
| maxnreg=192 | 19.0 ms | worse |
| maxnreg=224 | 18.3 ms | ≈ no cap, no gain |
| BLOCK_M=128, BLOCK_N=256 | OOM shmem | — |

Observation:
- **Nothing beat 18 ms.** `warp_specialize=True` compiled (for 8 warps) but gave **zero speedup** — Triton's automatic WS pass does not reproduce FA-3's hand-crafted MMA↔softmax overlap for this kernel (and it can't even compile at 4 warps).
- **Every occupancy-raising knob made it slower**: more warps → contention; `maxnreg` caps → register spills (the `[128,128]` FP32 accumulator genuinely needs the registers). This *confirms* occupancy is **not** the lever — consistent with Entry 2's BLOCK_M=64 sweep also being slower. The real bottleneck is the lack of compute overlap, which these knobs can't fix.

Conclusion: **~18 ms is the ceiling for this (standard) Triton flash-attention formulation; kept Entry 2 (18.86 ms) as the hand-written best.** The remaining 1.48× to cuDNN is FA-3-level Hopper scheduling (warp-specialized producer/consumer + TMA + software-pipelined MMA/softmax). Triton's automatic `warp_specialize` doesn't deliver it; closing the gap would require either a deep TMA + manual-pipeline Triton rewrite (uncertain — the one untried lever, but DRAM is only 3 % so loads aren't the bottleneck) or hand-written **CUTLASS/CuTe FA-3** (very large effort). Not pursued — diminishing returns, same lesson as swiglu (the vendor library is near-peak; hand-written gets close but the last ~1.5× needs vendor-level engineering).

Lessons: (1) `warp_specialize=True` is a cheap one-liner to *try*, but auto-WS ≠ hand-crafted FA-3 WS — don't expect the FA-3 speedup for free. (2) When ncu says "occupancy-limited," verify by *raising* occupancy — here every attempt was slower, proving the diagnosis (occupancy) was a symptom, not the cause (compute overlap). (3) Record the dead ends: WS, num_warps, maxnreg all explored and rejected with numbers.

---

### Entry 4 — CUTLASS FA-3 (Hopper FMHA) — matches cuDNN ✅

Date: 2026-06-27

Thinking: Entry 3 proved the gap to cuDNN is hand-crafted warp-specialization (MMA↔softmax overlap), which Triton's auto-WS can't deliver. Rather than hand-write FA-3 from scratch (months of CuTe work), instantiate **CUTLASS's own Hopper FMHA collective** (bundled example `88_hopper_fmha`) — it *is* FA-3 (warp-specialized + TMA + wgmma cooperative) — via `load_inline`, adapted to the problem.

Code version: `versions/v3_cutlass_fa3.py` (= `submission.py`).

How it was wired (the CUTLASS learning):
- `Operation = cutlass::device::Universal< FmhaBuilder<half_t, float, float, TileShape, StrideQ, StrideK, StrideV, DefaultFusion, KernelTmaWarpSpecializedCooperative>::Kernel >`.
- **TileShape = `Shape<_128,_128,_128>`** (BlockQ, BlockKV, head_dim=128), the example's D=128 cooperative config.
- **`DefaultFusion`** = non-causal, no residual mask (the seq_lens are multiples of 128); `CausalFusion`/`ResidualFusion` are the other options.
- Problem shape `(B,H,S,S,D)`; strides map the contiguous `[B,H,S,D]` to the kernel's `(S, D, (B,H))` layout: `stride = (D, _1, (H·S·D, S·D))` — D-major, no copy.
- **Softmax scale is auto-derived** from D in `to_underlying_arguments` (`1/√d`, `log2(e)/√d`) — matches the reference, nothing to pass.
- Needed an LSE scratch buffer `[B·H·S]` (ignored). Launch on `at::cuda::getCurrentCUDAStream()` so the harness's event timing is valid. `sys.stdout` guard + pre-build for the spawned-worker JIT (same as swiglu).

Correctness: **pass** all 3 cases (large max abs diff 0.0006).

Performance (harness benchmark):

| case | baseline | Triton (E2) | **FA-3 (E4)** | cuDNN (E1) |
|------|----------|-------------|---------------|------------|
| small | 0.344 | 0.094 | 0.071 | 0.063 |
| medium | 12.685 | 2.247 | 1.642 | 1.632 |
| **large** | **115.04** | **18.86** | **12.79** | 12.76 |

**FA-3 = 12.79 ms ≈ cuDNN's 12.76 ms — library parity** (9.0× over baseline, 1.48× over the Triton kernel). ~573 TF/s (~58 % MFU). Head-to-head in one process, FA-3 14.32 vs cuDNN 14.29 ms.

Profiling (ncu, large) vs the Triton kernel:

| metric | Triton (E2) | **FA-3 (E4)** |
|--------|-------------|---------------|
| Compute (SM) Throughput | 52.2 % | **76.2 %** |
| DRAM Throughput | 3.0 % | 4.3 % |
| Achieved Occupancy | 12.5 % | 14.0 % |

Observation:
- **Same low occupancy (~13 %) and same low DRAM (~4 %), but FA-3 hits 76 % SM vs Triton's 52 %.** That +24 pp is exactly the **warp-specialization** payoff: producer/consumer warpgroups keep the tensor cores fed (MMA overlaps softmax) without needing high occupancy. This *confirms Entry 3's diagnosis* — the bottleneck was compute overlap, not occupancy — and shows the fix that Triton's auto-WS couldn't provide but hand-crafted CUTLASS FA-3 does.
- cuDNN on Hopper is the same FA-3 lineage, so parity is expected; it wasn't beaten, it was matched.

Conclusion: **Entry 4 (CUTLASS FA-3, 12.79 ms) is the new best — library-parity hand-instantiated FlashAttention-3.** The honest framing: FA-3 was *not* hand-written; CUTLASS's FMHA collective (FmhaBuilder + cooperative WS/TMA dispatch) was wired to the problem. That *is* the realistic "serious CUTLASS-FA3" — and it closes the entire gap (115 → 12.8 ms, matching the vendor library).

Lessons: (1) The pragmatic way to "write FA-3" is to instantiate CUTLASS's FMHA collective, not hand-roll CuTe — `FmhaBuilder` + the right fusion/dispatch/tile is ~80 lines via load_inline. (2) ncu nails the mechanism: warp-specialization buys SM utilization (52→76 %) at the *same* occupancy — occupancy and utilization are different things. (3) Unlike swiglu (library was unbeatable), here matching the library by hand is achievable because the vendor kernel *is* open CUTLASS — reuse beats reinvention. (4) Reused everything from the swiglu CUTLASS work: load_inline build deps, sm90a flags, the spawned-worker stdout guard, stream handling.

---

### Entry 5 — Trying to BEAT cuDNN: FMHA config sweep (schedule / tile / scheduler / accQK) — no improvement

Date: 2026-06-27

Thinking: Entry 4 matched cuDNN (12.79 vs 12.76). cuDNN's heuristic might not pick the optimal config for this exact shape (non-causal, S=8192, D=128, fp16), so sweep the CUTLASS FMHA knobs: schedule (cooperative vs **pingpong**), TileShape (128×128 vs **128×256**), TileScheduler (individual vs **persistent**), and **accQK=fp16**.

Results:

| config | result |
|--------|--------|
| cooperative 128×128 (Entry 4) | **12.79 ms** (harness, stable) |
| pingpong 128×128 | quick-loop 14.18 ms (looked best!) → **harness 14.64 ms mean, std 0.93, worst 17.1** — worse + high-variance |
| coop 128×256 (fp32 acc) | FAIL init (`C7511` wgmma serialized — insufficient registers) |
| coop / pingpong 128×256 + accQK=fp16 | FAIL init (same constraint for D=128 on this shape) |
| cooperative 128×128 + persistent scheduler | 16.67 ms (worse) |

Profiling (ncu, large) — pingpong vs the Entry-4 cooperative kernel:

| metric | cooperative (E4) | pingpong (E5) |
|--------|------------------|---------------|
| Compute (SM) | 76.2 % | 76.2 % |
| DRAM | 4.3 % | 4.3 % |
| Achieved Occupancy | 14.0 % | 14.0 % |
| ncu duration (replay) | 14.99 ms | 14.83 ms |

⇒ **the two are identical in steady-state ncu SoL** (same 76 % SM). Yet pingpong's end-to-end harness time is worse and high-variance (14.64 ms mean, std 0.93) vs cooperative's stable 12.79 ms. So the regression is **not per-kernel compute efficiency** — it's scheduling/launch behavior across the 256·(S/128) tiles (pingpong's two alternating math warpgroups are more sensitive to L2 state / wave quantization under `clear_l2_cache`). A single-kernel ncu profile *cannot* see this; only the full benchmark does.

Conclusion: **nothing beat cooperative 128×128; kept Entry 4 (12.79 ms).** Pingpong looked marginally faster in a quick `perf_counter` loop (14.18 vs coop 14.36) and has identical ncu SoL, but the **authoritative harness** (with `clear_l2_cache` + 100 runs) exposed it as *worse* and high-variance (14.64 ms mean). The 128×256 tiles can't initialize for D=128 on this problem (register/shmem limit). cuDNN parity (12.79 ms, ~58 % MFU) stands as the ceiling — the vendor is matched, not beaten.

Lessons: (1) **Trust the harness measurement, not quick loops** — without `clear_l2_cache` + enough iters, pingpong looked best but was actually worse + noisy. (2) cuDNN's default config (cooperative-ish) is already optimal for this shape; the easy knobs don't beat it. (3) Beating cuDNN would need something cuDNN doesn't do for fp16 (e.g. FP8 — but that fails the fp16 1e-2 tolerance) — diminishing returns; matched is the right place to stop.

---

### Entry 6 — cached / slim CUTLASS FA-3 — WASH (no win over the FP16 floor)

Date: 2026-06-28

Hypothesis (review): the gap to SDPA (12.79 CUTLASS vs 12.76 SDPA) is ~30 µs of per-call
wrapper overhead (allocate O/LSE/workspace, can_implement, initialize) that the CUDA-event
timing might catch as a GPU-idle gap. Cache it (per shape+ptrs) so the hot path is just
`op.run`. Also fix the `device_id=0` hardcode.

Code version: `versions/v4_fa3_cached.py` (static `std::map` cache of O/LSE/workspace +
initialized `Operation`; hybrid: small/medium → SDPA, large → cached FA-3).

Same-session baseline (large case 4×64×8192×128, the only benchmarked shape):

| variant | mean | best | std |
|---------|-----:|-----:|----:|
| SDPA (v1) | 12.78–12.79 ms | 12.744 ms | 0.025 |
| CUTLASS FA-3 (v3) | 12.78 ms | 12.759 ms | 0.015 |
| **cached FA-3 (v4)** | 12.78–12.82 ms | 12.762 ms | — |

Result: **WASH.** SDPA, uncached CUTLASS, and cached CUTLASS are **statistically identical
(~12.78 ms)**. The "12.79 vs 12.76" was within run-to-run noise, NOT wrapper overhead — so
the alloc/init does not meaningfully enter the CUDA-event timed region, and caching it buys
nothing. Correctness passes (math unchanged). There is no FP16 wrapper headroom to harvest.

### Entry 7 — FP8 (e4m3) CUTLASS FA-3 — numerically VIABLE, but BLOCKED by a CUTLASS build wall

Date: 2026-06-28

Goal: change the roofline with Hopper FP8 tensor cores (~2× FP16). The prior Entry-4 note
guessed "FP8 fails the 1e-2 tolerance" — **this turned out to be wrong**, and worth the check.

Correctness (de-risked FIRST in PyTorch, before any kernel build): the output is tiny
(**mean|O| = 0.0145, max 0.19**), so atol=1e-2 is ~0.7× the mean output — huge absolute
headroom. A faithful FP8 simulation (e4m3 cast of Q/K/V, **per-row P scaling** before the
P→fp8 cast as a real FP8 FMHA does, fp8 O output, V_SCALE dequant):

| scheme | maxdiff | violations |
|--------|--------:|-----------:|
| naive (P cast to fp8 with NO scaling) | 0.082 | **52.7%** (P≈1/8192 underflows e4m3) |
| Q/K fp8, P/V fp32 | 0.0064 | 0% |
| full fp8 + per-row P-scale, fp8 O out, V_SCALE 1…64 | **0.0090** | **0%** |

So **FP8 is numerically viable** (maxdiff ~0.009 < atol 0.01, 0 violations) — *provided* the
kernel does the internal per-row P scaling (which a real FA-3 FP8 impl does). V_SCALE turned
out unnecessary (1–64 identical); the dominant error is the QK/PV fp8 math, not the O cast.
Caveat: this relies on the benchmark's **random-normal inputs (no outliers)** — FP8 has no
headroom for heavy-tailed real-LLM activations. (Honest scope note.)

Build: **FAILS.** Swapping `Element = cutlass::float_e4m3_t` into the `FmhaBuilder` (tried
TileShape 128×256×128 and 128×128×128) routes the QK mainloop to
`MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInput`, and `StageCountAutoCarveout` hands it a
plain `StageCount<5>` that "has no member bytes" → *"Could not find a mainloop
specialization."* The bundled `cutlass_library` source's `FmhaBuilder` has no working direct
FP8 path; example 88's FP8 goes through its full `FwdRunner` + `#define FP8` machinery
(different stage-count / mainloop wiring). Reproducing that is a ~200-line port (or a CUTLASS
header patch) — beyond the 2-entry budget for a borderline-correctness (0.009 vs 0.01),
speed-unverified payoff.

Decision: **neither entry beats the FP16 floor.** Submission stays at the FP16 vendor floor
**(~12.76 ms; `v3_cutlass_fa3.py`, cuDNN-parity)**. Honest standing: FP16 attention here is
a hard vendor wall (SDPA = CUTLASS = cached, all 12.78). FP8 is the only real lever and is
*numerically* in-budget (refuting the earlier guess), but the CUTLASS FP8 FMHA does not
build via the direct builder in this environment — the win is gated by a toolchain wall, not
by math or by tolerance.

Lessons: (1) Re-test inherited "it won't work" claims — FP8 *does* pass 1e-2 here (the prior
guess was wrong); de-risk correctness in a 20-line sim before building. (2) The small-P
underflow (52.7% → 0%) shows FP8 attention REQUIRES per-row P scaling — naive P→fp8 is
catastrophic. (3) A real win can be blocked by toolchain (builder/version) rather than
algorithm; know when that wall is past the budget and stop honestly.
