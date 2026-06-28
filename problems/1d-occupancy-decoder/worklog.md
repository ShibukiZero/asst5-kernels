# 1D Occupancy Decoder Work Log

## Problem

Optimize the 1D occupancy decoder forward pass for the H100 leaderboard.

Key workload:
- Queries: `[batch_size, num_queries, q_in_dim]`
- Latents: `[batch_size, num_latents, width]`
- Output: `[batch_size, num_queries, 1]`
- Main benchmark case: `batch_size=1; num_queries=250000; num_latents=1024; width=768; num_heads=12; q_in_dim=3`

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
