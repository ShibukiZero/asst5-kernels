# 1D Occupancy Decoder Work Log

## Problem

Optimize the 1D occupancy decoder forward pass for the H100 leaderboard.

Key workload:
- Queries: `[batch_size, num_queries, q_in_dim]`
- Latents: `[batch_size, num_latents, width]`
- Output: `[batch_size, num_queries, 1]`
- Main benchmark case: `batch_size=1; num_queries=250000; num_latents=1024; width=768; num_heads=12; q_in_dim=3`

## Summary of Results

Env: Nebius H100 80GB, torch 2.12.1+cu130, triton 3.7.1. CUDA-event benchmark, L2
cleared each run, tol rtol=atol=1e-2. (README's reference machine reports the PyTorch
path at 7.4 ms; ours is a different stack — all speedups below are vs OUR baseline.)

| Entry | Approach | Runtime | vs baseline | Verdict |
|------:|----------|--------:|:-----------:|---------|
| 0 | PyTorch reference (rebuilds nn.Module each call) | 18.789 ms | 1.00x | baseline |
| 1 | Functional forward (no Module rebuild) | ~5.35 ms | 3.5x | ✅ kept the framework win |
| 2 | torch.compile (default Inductor) | 4.070 ms | 4.6x | ✅ glue fusion |
| 3 | compile mode sweep: max-autotune / cudagraphs | 4.667 / 4.079 ms | — | ✗ max-autotune LOSES (swaps cuBLAS GEMM); cudagraphs WASH |
| 4 | hand Triton flash for SDPA (tile sweep) | 0.90x cudnn (attn) | — | ✗ loses to cudnn (24.7% occ) |
| 5 | CUTLASS FA-3 for SDPA (warp-spec FMHA) | 0.85x cudnn (attn) | — | ✗ loses to cudnn (kv too short → 13.6% occ) |
| 6 | q-fold: collapse out_layer+c_q into one GEMM | 3.590 ms | 5.2x | ✅ deletes 1 big GEMM (3→2) |
| **7** | **q-fold + fused LayerNorm→out_proj tail (Triton)** | **3.418 ms** | **5.5x** | ✅ **BEST — submission** |

**Bottom line:** two phases. (a) Structural: removing the per-call nn.Module construction
(Entry 1, 3.5x) + Inductor glue fusion (Entry 2, →4.07 ms). (b) **Graph-level edits that
beat Entry 2 without touching the vendor kernels** (Entries 6-7, 4.07→3.42 ms): the SDPA
and GEMM *backends* are a vendor wall (hand Triton flash 0.90x, FA-3 0.85x, max-autotune
loses), so instead **change the graph** — fold the two consecutive affines `out_layer·c_q`
into one cached GEMM (−0.49 ms) and fuse `LayerNorm→out_proj` into one per-row Triton
kernel that never materializes the 250k×768 normalized tensor (−0.17 ms). Both are exact
(5 seeds, maxdiff 7e-4), legitimate (correct for any input, no cached reference values).
SDPA (2.16 ms, ~63%) stays the untouched vendor floor. **Practical optimum = Entry 7,
3.418 ms (5.5x).**

`submission.py` = Entry 7 (`versions/v7_qfold_tailfused.py`). Every entry has profiling
(ncu for 0/1/2/4/5/6/7, torch.profiler for 3). Code for each entry in `versions/`.

## Optimization Entries

Append one entry per meaningful experiment.

### Entry 0 - Baseline

Date: 2026-06-28

Environment: Nebius H100 80GB HBM3, driver 580.159.04, torch 2.12.1+cu130,
triton 3.7.1, Nsight Compute 2025.3.1. venv `~/asst5-venv`. Run with
`PYTHONPATH=$PWD` (problem dir) so `reference`/`submission` import; `ncu` and
`ncu_report` need CUDA bin + `nsight-compute/.../extras/python` on PATH/PYTHONPATH.

Code version:
- File(s): `submission.py` = `reference.ref_kernel` (stock PyTorch), `versions/v0_baseline.py`
- Short description: builds `OneDOccupancyDecoder` nn.Module each call, loads the
  14 weights via `.copy_()`, `model.eval()`, `model(queries, latents)`.

Command:
```bash
PYTHONPATH=$PWD python ../eval.py test      test_cases/test.txt   # correctness
PYTHONPATH=$PWD python ../eval.py benchmark test_cases/test.txt   # CUDA-event timing
PATH=/usr/local/cuda/bin:$PATH PYTHONPATH=/opt/nvidia/nsight-compute/2025.3.1/extras/python:$PWD \
  python ../eval.py profile test_cases/test.txt                   # ncu per-kernel
```

Correctness:
- Status: pass (`check: pass`)
- Notes: tol rtol=atol=1e-2 (loose); softmax in fp32 internally (SDPA).

Performance:
- Runtime: **mean 18.789 ms, std 0.643, best 17.955 ms** (100 runs, CUDA events,
  L2 cleared each run). NOTE: README's reference machine reports 7.399 ms for the
  same PyTorch path — see "the 18.8 vs 7.4 gap" below.
- README's two Triton baselines (9.158 ms CA-Triton, 7.277 ms Triton-MLP+PyTorch-CA)
  are CA's own experiments; their code is NOT shipped (templates/ only has skeletons),
  so they can't be re-run here. Only the PyTorch path is reproducible.

ncu per-kernel breakdown (single forward; durations are isolated-kernel times):

| # | kernel | dur | Compute% | DRAM% | maps to |
|---|--------|----:|---------:|------:|---------|
| 0,1 | distribution_elementwise | 15us | 47 | 0 | **wasted** random weight init inside the nn.Module ctor (immediately overwritten by copy_) |
| 2 | "Kernel2" | 402us | 52 | 27 | query_in.in_layer GEMM (3→768) on 250k rows |
| 3 | vectorized_elementwise | 253us | 69 | 88 | SiLU on 250k×768 (mem-bound) |
| 4 | nvjet 192x208 ...bias_TNT | 494us | 79 | 46 | query_in.out_layer GEMM (768→768) |
| 5 | nvjet 192x208 ...bias_TNT | 494us | 79 | 46 | c_q GEMM (768→768) |
| 6,7 | nvjet 96x64 ...bias_TNN | 8us×2 | 19 | 10 | c_k, c_v GEMMs on 1024 latents (tiny) |
| 8 | **cudnn flash sdpa wgmma f16** | **2.17 ms** | 50 | 10 | **cross-attention SDPA — single biggest kernel (35%)** |
| 9 | nvjet 192x208 ...bias_TNT | 489us | 79 | 46 | c_proj GEMM (768→768) |
| 10 | unrolled_elementwise | 620us | 59 | 54 | head reshape/contiguous of attn output (250k×12×64→250k×768) |
| 11 | **vectorized_layer_norm** | **777us** | 78 | 58 | LayerNorm(768) over 250k rows (fp32 internal) |
| 12 | vectorized_elementwise | 367us | 8 | 92 | fp16↔fp32 cast around LayerNorm (mem-bound) |
| 13 | elementwise | 4us | 7 | 0 | trivial |
| 14 | nvjet 384x8 ...badd_TNT | 130us | 5 | 89 | out_proj GEMM (768→1), N=1, mem-bound |

Sum of forward kernels (excl. 0,1) ≈ **6.21 ms** of GPU work.

Observation — **the 18.8 vs 7.4 vs 6.2 ms gap**:
- Pure GPU kernel work = ~6.2 ms (matches README's 7.4 ms ballpark).
- The harness times `custom_kernel(data)` with CUDA events. `ref_kernel` rebuilds
  the nn.Module every call: `OneDOccupancyDecoder(...).to(...)` (random-inits all
  Linear weights — kernels 0,1, then throws them away), 14 host-launched `.copy_()`,
  `.eval()`. This CPU-side construction + serialized small launches starve the GPU,
  so the event-bracketed timeline stretches with bubbles: 6.2 ms of work → 18.8 ms wall.
- So ~12 ms of the baseline is pure framework/launch overhead, not math.

Where the real math goes: **SDPA 35%, the three 768×768 GEMMs 24%, LayerNorm 12.5%,
head-reshape 10%, SiLU+casts 10%.** SDPA is already cudnn FlashAttention (wgmma fp16,
fp32 softmax) — README confirms CA's hand Triton CA port LOST (0.81×), so don't fight it.

Hypothesis (next entries):
1. **Entry 1 — functional forward, no Module rebuild.** Compute the forward directly
   from the weights dict with `F.linear`/`F.scaled_dot_product_attention`. Removes the
   ctor random-init (kernels 0,1) and the CPU construction bubbles → should collapse
   toward the ~6 ms GPU floor. Biggest, cheapest win.
2. **Entry 2 — torch.compile the functional forward.** Fuse the elementwise glue
   (SiLU, the LayerNorm fp32 casts, bias adds, the head reshape) — kernels 3,10,11,12
   are ~2.0 ms of mostly mem-bound traffic that should fuse into the neighbouring ops.
3. Keep GEMMs on cuBLAS and attention on cudnn (libraries already strong here).

Next step: Entry 1 functional forward.

### Entry 1 - Functional forward (no nn.Module rebuild)

Date: 2026-06-28

Code version:
- File(s): `submission.py`, `versions/v1_functional.py`
- Short description: compute the graph straight from the weights dict with
  `F.linear` / `F.silu` / `F.scaled_dot_product_attention` / `F.layer_norm`. No
  Module construction, no `.copy_()`, no `.eval()`. `NUM_HEADS=12`, `LN_EPS=1e-6`
  hardcoded (problem constants — `custom_kernel` only receives queries/latents/weights).

Correctness:
- Status: pass (1e-2). Bit-for-bit same op sequence as reference.

Performance:
- Runtime: **mean ~5.35 ms** (three runs: 5.32 / 5.40 / 5.41 ms, std <0.01).
  **3.5× over baseline (18.789 -> 5.35).** Below the ncu isolated-sum (6.2 ms)
  because the harness benchmark runs kernels back-to-back with warm L2 (latents
  reused) and overlapped launches, while ncu serializes + clears L2 per kernel.
- ncu per-kernel: **identical forward kernel list to Entry 0** (same 13 forward
  kernels; K0,1 are `generate_input`'s two randn, not model init — Entry 0's note
  corrected). GEMMs still cuBLAS (nvjet), attention still cudnn flash (2.14 ms).

Observation:
- The entire 18.8->5.35 ms win came from removing CPU-side per-call Module
  construction + 14 serialized `.copy_()` launches that were starving the GPU.
  The math is unchanged. We are now at the real GPU floor for this op sequence.
- Remaining GPU time is dominated by **SDPA 2.14 ms (40% now)**, then the three
  768x768 GEMMs (~1.48 ms), LayerNorm 0.78 ms + its fp32 casts 0.37 ms (1.15 ms
  of mem-bound glue), head-reshape contiguous 0.62 ms, SiLU 0.25 ms.

Hypothesis:
- **Entry 2 — torch.compile** the functional forward: fuse SiLU, the LayerNorm
  fp32 up/down casts, biases, and ideally fold the head-reshape `.contiguous()`
  into the SDPA epilogue/prologue. Targets kernels 3,10,11,12 (~2.0 ms mem-bound).
- SDPA stays on cudnn (README: CA's hand Triton CA port lost 0.81x).

Next step: Entry 2 torch.compile.

### Entry 2 - torch.compile (TorchInductor, default mode, fullgraph)

Date: 2026-06-28

Code version:
- File(s): `submission.py`, `versions/v2_compile.py`
- Short description: core compute pulled into `_forward(tensors...)` and wrapped in
  `torch.compile(..., fullgraph=True)`. Tensors passed explicitly (not the dict) to
  avoid guard churn. First call compiles (absorbed by the harness correctness call);
  static shapes -> no recompiles.

Correctness:
- Status: pass (1e-2).

Performance:
- Runtime: **mean 4.070 ms, std 0.062, best 3.904 ms** (100 runs).
  **1.31x over Entry 1 (5.35 -> 4.07); 4.6x over baseline.**

ncu per-kernel vs Entry 1 (what Inductor fused):

| op | Entry 1 | Entry 2 | delta |
|----|--------:|--------:|------:|
| in_layer (3->768) | 402us (generic) | ~167us (triton prologue + small nvjet) | -235us |
| SiLU | 252us | 251us (`triton_poi_fused_silu_view`) | ~same (between two GEMMs, can't fuse into epilogue) |
| out_layer / c_q / c_proj GEMMs | 3x ~494us | 3x ~494us (cuBLAS nvjet, unchanged) | 0 |
| c_k, c_v | 8us x2 | 8us x2 | 0 |
| SDPA | 2.14 ms | 2.17 ms (cudnn flash, unchanged) | 0 |
| **head reshape contiguous** | **620us** | **gone** | **-620us** (transpose folded into the SDPA->c_proj layout; no explicit copy) |
| **LayerNorm + fp32 casts** | **777 + 366 = 1143us** | **251us** (`triton_per_fused__to_copy_native_layer_norm_view`) | **-892us** (fused into one kernel) |
| out_proj | 130us | 131us | 0 |

- Isolated-sum forward: 6.2 -> ~5.07 ms; benchmark 5.35 -> 4.07 ms.

Observation:
- torch.compile delivered exactly the predicted glue fusion (~1.75 ms saved across
  reshape + layernorm + in_layer) for ~zero effort, GEMMs/attention untouched.
- **SDPA is now the dominant cost: 2.17 ms = ~43% of the isolated sum.** It runs at
  only 50% compute / 10% DRAM throughput -> neither bound saturated, so it's
  latency/occupancy-limited on this lopsided shape (250k queries x 1024 KV, hd 64).
- Next-biggest: the three 768x768 GEMMs (~1.48 ms, 79% compute - fairly tight).

Hypothesis:
- **Entry 3 — max-autotune (+ CUDA graphs).** CUDA graphs collapse the ~13 per-call
  kernel launches into one replay (kills inter-kernel launch latency, which is part
  of why benchmark 4.07 < isolated-sum 5.07 already - more to get). max-autotune may
  also find better GEMM/epilogue templates. Risk: cudagraphs + the harness's static
  reused `data` should be compatible; verify correctness.
- Later (Entry 4+): attack SDPA itself only if a custom flash for this shape can beat
  cudnn's 50%-compute run - README says CA's hand Triton CA port lost (0.81x), so
  this is the hard, uncertain frontier; do the cheap CUDA-graph win first.

Next step: Entry 3 max-autotune + cudagraphs.

### Entry 3 - compile-mode sweep: max-autotune (LOSS) + reduce-overhead/cudagraphs (WASH)

Date: 2026-06-28

Code version:
- File(s): `versions/v3_compile_modes.py` (the max-autotune variant kept for record;
  submission stays on Entry 2's default mode).
- Two one-line variants of Entry 2, only the `torch.compile(mode=...)` changed.

Results (100 runs each):

| mode | runtime | vs Entry 2 (4.070) |
|------|--------:|--------------------|
| default (Entry 2) | 4.070 ms | — |
| `max-autotune` | **4.667 ms** | **LOSS (+15%)** |
| `reduce-overhead` (cudagraphs) | 4.079 ms | wash |

Correctness: both pass (1e-2).

Profiling (torch.profiler, CUDA self-time, ms/iter over 10 iters).

Note on ncu for max-autotune (tried, abandoned): the idea was "warm the inductor
autotune cache outside ncu, then ncu only replays the final fixed kernels." It does
NOT work with this eval.py: `run_profiling` launches ncu around a fresh one-shot
subprocess (`python -c ...`) that calls `custom_kernel` exactly once with no warmup,
and that subprocess did not reuse the warm inductor FX/autotune cache cross-process
-> it re-autotuned *under* ncu, so ncu instrumented all the GEMM trial-kernel
launches and the .ncu-rep ballooned (>65 MB, still climbing) before timing out.
(Also learned: a backgrounded ssh `ncu` is not killed by stopping the local task;
the remote ncu reparents to init and must be `pkill`ed on the VM. And `pkill -f
"eval.py"` in a one-liner matches its own remote shell — verify with `pgrep -xc ncu`.)
torch.profiler kernel names give the needed evidence directly, so we use those:

max-autotune variant (4.318 ms self-sum):
| kernel | ms/iter | % |
|--------|--------:|--:|
| cudnn flash sdpa | 1.74 | 40.3 |
| **`triton_tem_fused_addmm_silu_t_view`** | **1.74** | **40.2** |
| triton layernorm+casts | 0.26 | 5.9 |
| triton silu | 0.25 | 5.9 |
| nvjet (one small GEMM survived) | 0.16 | 3.7 |
| triton_tem addmm+layernorm | 0.14 | 3.1 |

reduce-overhead / cudagraphs variant (4.427 ms self-sum):
| kernel | ms/iter | % |
|--------|--------:|--:|
| cudnn flash sdpa | 2.14 | 48.4 |
| nvjet 192x208 (3 big GEMMs, aggregated) | 1.60 | 36.1 |
| triton silu | 0.25 | 5.8 |
| triton layernorm+casts | 0.25 | 5.7 |
| nvjet out_proj | 0.14 | 3.1 |
| nvjet c_k/c_v | 0.01 | 0.3 |

Observation:
- **max-autotune LOST — direct kernel evidence.** The autotuner replaced a big cuBLAS
  GEMM with a fused Triton template `triton_tem_fused_addmm_silu_t_view` that costs
  **1.74 ms** — as large as the whole SDPA, ~3.5x the ~0.49 ms the cuBLAS nvjet took.
  Only one small nvjet GEMM survived. For these large-M (250k-row) 768x768 fp16 GEMMs,
  cuBLAS crushes Inductor's Triton templates. (Same lesson as rk4/swiglu: vendor
  libraries win on bread-and-butter GEMM shapes; don't let autotune override them.)
- **cudagraphs = WASH — confirmed.** reduce-overhead's kernel set is identical to
  Entry 2 default: SDPA + three nvjet GEMMs (the 192x208 line aggregates all three
  at 1.60 ms) + the same fused silu/layernorm triton kernels. No fusion change, so the
  only possible gain was launch latency — negligible against the big kernels.
- **cudagraphs was a WASH.** The forward is dominated by big kernels (SDPA 2.17 ms,
  three ~0.5 ms GEMMs); per-launch latency is negligible against those. The
  4.07 < isolated-sum 5.07 gap came from warm-L2 reuse + kernel overlap, not launch
  latency that cudagraphs could remove.

Decision: keep submission on **default torch.compile (Entry 2, 4.070 ms)** — best so far.

Hypothesis (the real remaining frontier):
- **SDPA = 2.17 ms = ~43% of the work** and runs at only 50% compute / 10% DRAM ->
  not bound by either, i.e. latency/occupancy-limited on this lopsided shape
  (250k queries x 1024 KV, 12 heads, hd 64). Roofline: attention FLOPs ~1.57 TFLOP
  -> ~1.6 ms fp16 floor on H100, so cudnn's 2.17 ms is ~74% of floor. A custom flash
  tuned for huge-Q / tiny-KV *might* claw back ~0.4-0.5 ms.
- Entry 4 = attack SDPA: try CUTLASS FA-3 (example 88 FmhaBuilder, the recipe that
  matched cuDNN on the flashattention problem) and/or a hand Triton flash for this
  shape. HIGH RISK / uncertain: README reports CA's hand Triton CA port lost (0.81x).
  Everything cheap has been harvested; this is the only remaining lever.

Next step: Entry 4 - SDPA via CUTLASS FA-3 (or hand Triton flash for huge-Q/tiny-KV).

### Entry 4 - hand Triton flash attention for the SDPA (tile-strategy sweep) - LOSES to cudnn

Date: 2026-06-28

Goal: replace the cudnn SDPA (the dominant kernel) with a hand Triton FA-2 tuned
for this lopsided shape (q_len 250k, kv_len 1024, 12 heads, hd 64, no mask). Isolated
attention micro-benchmark (just Q,K,V -> attn out), correctness vs `F.sdpa` (maxerr).

Code version: `versions/v4_triton_flash.py` (best config), sweep scripts
`tmp/decoder_flash_sweep*.py`. Grid = (ceil(LQ/BM), B*H); each program does BM queries
for one (b,h), loops over kv in BN chunks with online softmax, fp32 accumulate.

Sweep results (isolated attention; cudnn SDPA = ~1.79-1.81 ms on this shape):

Gen 1 (plain FA-2, scale-after-dot, full N-masking) - 48 configs BMx{64,128,256}:
- best **2.466 ms = 0.72x cudnn** at BM=128 BN=64 nw=8 ns=3. All configs lose.
- BN=256 with few warps blows up (register spill: 256x256x4w = 43 ms).

Gen 2 (+exp2 softmax, +drop N-mask since LKV%BN==0) - same tile grid:
- best **2.017 ms = 0.90x cudnn** at BM=128 BN=64 nw=8 ns=4. exp2+no-mask lifted
  0.72x -> 0.90x. Still 10% slower than cudnn. Correctness maxerr ~2e-4 (tol 5e-2).
- Pattern: BM=128, BN=64, nw=8 is the sweet spot; BN>=256 spills shared memory
  (out of resource at ns=4), nw=8 hurts small BM=64 tiles.

`warp_specialize=True` (the Hopper feature cudnn/FA-3 win with): **hard LLVM crash**
"unsupported load type for producer commit" - it requires TMA / block-pointer
(`make_block_ptr`) loads, not the raw pointer-arithmetic loads here. Process-level
abort (uncatchable), so it must be excluded from the sweep. Enabling it = a full
rewrite to tensor-descriptor loads.

ncu of the best Triton flash (BM=128 BN=64 nw=8 ns=4), grid (1954,12)x(256):

| metric | value |
|--------|------:|
| sm__throughput (SM%) | 57.9% |
| compute-mem throughput | 51.1% |
| **achieved occupancy** | **24.7%** |
| IPC | 0.58 inst/cycle |

Observation:
- **Hand Triton flash LOSES: 0.90x cudnn at best**, exactly as the README warned
  (CA's hand Triton cross-attn port was 0.81x). The ncu tells why: **24.7% occupancy,
  0.58 IPC** - BM=128 + 8 warps eats registers/shared mem, capping occupancy, so the
  kernel can't hide latency. cudnn's wgmma + warp-specialized + TMA pipeline keeps the
  tensor cores fed at far higher effective utilization. Plain pointer-load Triton FA-2
  simply can't express that scheme (warp_specialize crashes without TMA loads).
- Same architectural ceiling found on the flashattention problem: Triton FA-2 ~52% SM
  vs CUTLASS FA-3 / cudnn ~76%. The last ~10-25% is Hopper warp-spec/TMA, not tiling.

Decision: **do NOT adopt.** Keep cudnn SDPA. Submission stays Entry 2 (4.070 ms).
The only paths that could *match* (not clearly beat) cudnn are CUTLASS FA-3 (example 88,
which merely TIED cudnn on the flashattention problem) or a TMA/warp-specialized Triton
rewrite (block pointers). High effort for an expected tie -> low value. Entry 2 stands
as the practical optimum (4.6x over baseline).

### Entry 5 - CUTLASS FA-3 (real Hopper warp-spec FMHA) for the SDPA - LOSES to cudnn

Date: 2026-06-28

Goal: the "do it properly" attempt - replace SDPA with CUTLASS FA-3 (example 88
FmhaBuilder, the warp-specialized + TMA + wgmma kernel that TIED cudnn on the
flashattention problem). Adapted from `flashattention/versions/v3_cutlass_fa3.py`:
head_dim 128 -> 64 (TileShape K-mode = _64; example 88 `run_fwd_64` shapes), and
**separate Q vs K/V strides** since q_len(250k) != kv_len(1024). Non-causal ->
DefaultFusion. Built via load_inline (needs `/usr/local/cuda/bin` on PATH).

Code version: `versions/v5_cutlass_fa3.py`. Isolated attention vs `F.sdpa`, maxerr.

Variant sweep (isolated attention; cudnn SDPA = ~1.77-1.80 ms on this shape):

| schedule + TileShape (M,N,D) | time | vs cudnn | correctness |
|------------------------------|-----:|---------:|-------------|
| WarpSpec **Cooperative** 128x64x64 | 2.359 ms | 0.76x | maxerr 2.4e-4 |
| WarpSpec **Pingpong** 128x64x64 | 2.158 ms | 0.83x | 2.4e-4 |
| WarpSpec **Cooperative** 128x128x64 (best FA-3) | 2.084 ms | 0.85x | 2.4e-4 |
| (hand Triton best, Entry 4) | 2.017 ms | 0.90x | 2e-4 |
| **cudnn SDPA** | **~1.78 ms** | **1.00x** | — |

ncu of the best FA-3 (coop 128x128x64), grid (1954,1,12)x(384):

| metric | FA-3 coop | (Triton, Entry 4) |
|--------|----------:|------------------:|
| sm__throughput (SM%) | 52.3% | 57.9% |
| compute-mem throughput | 28.4% | 51.1% |
| **achieved occupancy** | **13.6%** | 24.7% |
| IPC | 0.53 | 0.58 |

Observation - **even real FA-3 LOSES to cudnn here (0.85x at best), and loses to my
own hand Triton too.** This is the opposite of the flashattention problem, where FA-3
tied cudnn. The cause is the lopsided shape: **kv_len=1024 is tiny** (only 8-16 N-tiles
per q-tile). FA-3's whole advantage is a deep warp-specialized producer/consumer TMA
pipeline that amortizes over a LONG kv loop; with kv this short the pipeline can't fill,
the producer warpgroup starves, and **occupancy collapses to 13.6%** (worse than the
plain Triton's 24.7%, far below what's needed to hide latency). The more sophisticated
the kernel, the more its pipeline overhead dominates when there isn't enough kv work to
pipeline. cudnn evidently dispatches a kernel matched to short-kv and wins.

Lesson: warp-spec/TMA flash is the right tool for balanced or long-kv attention, NOT for
the huge-Q / tiny-KV decode-style shape. Vendor cudnn's shape-aware heuristic beats a
single hand-picked FA-3 tile here.

Decision: **do NOT adopt.** cudnn SDPA is the floor for this attention; nothing tried
(Triton 0.90x, FA-3 0.85x) beats it. **Submission stays Entry 2 (torch.compile, 4.070 ms,
4.6x over baseline) - the confirmed practical optimum for this problem.** The SDPA
frontier is closed: it is a vendor-library wall, like the L2 wall in rk4.

---

### Entry 6 — q-fold: collapse out_layer + c_q into one linear — BEATS Entry 2 ✅

Date: 2026-06-28

Context: a review proposed attacking the **compute graph** instead of the SDPA/GEMM
backends (Entries 3-5 proved those are a vendor wall: max-autotune 4.667, Triton flash
0.90x cudnn, FA-3 0.85x). The query path runs `out_layer` then `c_q` as two consecutive
affine maps with **no nonlinearity between them** (SiLU is before out_layer) → foldable.

Math: `qh = (h·qo_w^T + qo_b)·cq_w^T + cq_b = h·(cq_w·qo_w)^T + (qo_b·cq_w^T + cq_b)`
⇒ `qfold_w = cq_w @ qo_w`, `qfold_b = F.linear(qo_b, cq_w, cq_b)`. The folded weight is
computed **in fp32** (then cast to fp16) to minimise folding error, and **cached** (it
depends only on weights, not on queries/latents — legitimate, not input memorization).
Deletes one 250k×768×768 cuBLAS GEMM: the three big 768→768 GEMMs (out_layer, c_q,
c_proj) become two (qfold, c_proj). Builds on Entry 2's torch.compile.

Code version: `versions/v6_qfold.py`.

Correctness: **5/5 seeds pass** (5531,1,42,1234,9999), **maxdiff ≈ 0.0007** — far under
the 1e-2 tol. The fp32-computed fold makes the fp16-rounding risk a non-issue.

Performance: **mean 3.590 ms** (best 3.453) vs same-session Entry 2 **4.095 ms** →
**−0.505 ms / 12.3%**, ~5.2× over baseline.

Profiling (ncu) — the predicted "3 big GEMMs → 2" signal, confirmed:

| big nvjet 192x208 GEMM | Entry 2 | Entry 6 |
|------------------------|--------:|--------:|
| out_layer (250k×768×768) | ~497 µs | — folded |
| c_q (250k×768×768) | ~495 µs | — folded |
| **qfold (replaces both)** | — | **497 µs** |
| c_proj (250k×768×768) | ~491 µs | 489 µs |
| in_layer (small) | ~159 µs | 155 µs |

3 × ~490 µs → 2 × ~490 µs: one full GEMM gone. SDPA (2.16 ms), LayerNorm (250 µs),
out_proj (130 µs) unchanged. The −505 ms wall drop ≈ the removed 495 µs GEMM.

Decision tree: Entry 6 **passed** ⇒ Entry 7 stacks the tail fusion on top.

### Entry 7 — Entry 6 + fused LayerNorm→out_proj tail (Triton) — BEST ✅

Date: 2026-06-28

Thinking: the output is `[B, 250k, 1]`, so materializing the normalized `[B, 250k, 768]`
tensor and GEMM-ing it to a scalar is wasted traffic. One Triton kernel per row: mean/var
(fp32) → normalize → **cast to fp16** (matches `LayerNorm.type_as`) → dot with out_proj
weight (fp32 accumulate) + bias → one fp16 scalar. Replaces Entry 6's tail (Inductor
LayerNorm ~250 µs + out_proj GEMM ~130 µs, both streaming 250k×768) with one read of y.
The compiled prefix is truncated after c_proj; the tail is the custom kernel.

Code version: `versions/v7_qfold_tailfused.py`. num_warps swept {1,2,4,8}: 1/2 tie
~3.41 ms, 4 = 3.49, 8 = 3.65 (768-element rows want few warps); kept **num_warps=2**.

Correctness: **5/5 seeds pass, maxdiff ≈ 0.0007** (identical to Entry 6 — the fused tail
adds no error; the fp16-cast-before-dot faithfully mimics the reference).

Performance: **mean 3.418 ms** (best 3.228; a later confirm run 3.431) vs Entry 6
3.590 ms (−~165 µs) and **vs Entry 2 4.095 ms → −0.66 ms / 16.5%**, ~5.5× over baseline.

Profiling (ncu): the two tail kernels (LayerNorm 250 µs + out_proj nvjet_384x8 130 µs =
380 µs) collapse into **one `_ln_out_kernel` at 181 µs** — grid (250000)×(64 thr), and
its own SoL: **SM 88.3%, DRAM 64.1%, occupancy 91.6%**. So it's mildly SM-bound (the
per-row mean/var/dot), not pure bandwidth — but still ~200 µs cheaper than the pair it
replaces. q-fold's 3→2 big-GEMM structure is intact.

Decision: **submit Entry 7 (`v7_qfold_tailfused.py`, 3.418 ms)** — fastest passing.
Both wins are legitimate graph transforms (linear folding + tail fusion): correct for any
input (5 seeds, maxdiff 7e-4), no vendor-kernel replacement, no cached reference values.
The SDPA (2.16 ms, 63% of the runtime) remains the untouched vendor floor; the graph-level
wins took the decoder from 4.07 → 3.42 ms without fighting it.

Lessons: (1) When the vendor kernels are a wall, **change the graph, not the backend** —
folding two consecutive affines deletes a whole GEMM for a cached weight-only precompute.
(2) Compute the fold in **fp32** to keep the (only) correctness risk negligible. (3) For a
scalar output, **don't materialize the wide normalized tensor** — fuse LN+projection into
one per-row kernel. (4) Tiny-row reductions (768) want **few warps** (1-2), not the
elementwise default. (5) None of this touches SDPA — the biggest kernel can stay vendor
while graph-level edits still buy 16%.
