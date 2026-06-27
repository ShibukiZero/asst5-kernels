# RK4 Work Log

## Problem

Optimize the 3D heat equation solver with 8th-order finite-difference Laplacian and RK4 time integration for the H100 leaderboard.

Key workload:
- Input field: `(Nz, Ny, Nx)`, float32
- Output field: `(Nz, Ny, Nx)` after RK4 updates
- Main benchmark case: `grid_size=600; n_steps=10`

## Optimization Entries

Append one entry per meaningful experiment.

### Roofline (the target)

Solve `u_t = α∇²u`: per RK4 step = **4 Laplacian sweeps** (one per stage k1..k4) + elementwise stage combines. Laplacian = 8th-order, radius-4, 25-point stencil. Grid 600³ (interior 592³), fp32, 10 steps.

| Quantity | Value |
|----------|-------|
| Field size | 600³·4B = **0.86 GB** |
| Min HBM per step (4 sweeps, neighbor-reuse: read field + write k each) | ~4·(2·0.86) ≈ **7 GB** |
| 10 steps | ~70 GB → **~21 ms** at 3.35 TB/s |
| Arithmetic | ~25 FMA/point/sweep → tiny ⇒ **memory-bound** |

⇒ a fully fused stencil+RK4 kernel should be ~20–40 ms. Naive CUDA (README) = 148 ms; naive Triton = 317 ms; **PyTorch = 1458 ms**. Lots of room.

Hardware: H100 80GB HBM3 (Nebius VM), CUDA 13, torch 2.12. **Tolerance: rtol=atol=1e-6 (very tight, near fp32 eps).**

---

### Entry 0 — Baseline (PyTorch reference, sliced stencil + RK4)

Date: 2026-06-27

Code version: `submission.py` = `reference.ref_kernel` — each Laplacian term `c_i·(left+right)` and each RK4 add/mul is a separate PyTorch slice op; materializes full-size temporaries; `DeterministicContext` for reproducibility.

Command: `./run.sh rk4 benchmark`

Correctness: **pass** (custom == reference).

Performance: **1408.5 ms** (grid 600, 10 steps; mean of 3, std 0.13).

Profiling (torch.profiler, n_steps=2; steps are identical):

| stat | value |
|------|-------|
| CUDA kernel launches | **~219 per step** (438 in 2 steps; ~2190 for the full run) |
| CUDA time | 286 ms / 2 steps → ~143 ms/step (×10 = 1430 ≈ baseline) |
| kernel types | all `(vectorized_)elementwise_kernel` (~500–790 µs each) + 17 DtoD memcpy/2 steps |
| ncu (one elementwise) | **DRAM 92.4 %**, L2 82.8 %, Compute 4.9 % — pure DRAM-bandwidth-bound |

Observation — what limits performance:
- **The baseline does ~219 separate, memory-bound kernel passes per step.** Every stencil term and every RK4 combine is its own kernel that streams the full ~0.86 GB field through HBM (each ~0.5–0.8 ms = one read+write of the field). No data reuse across the 25-point stencil — each `c_i·(u_shift_a + u_shift_b)` reads the field twice and writes a 0.86 GB temporary.
- It's **~219 × ~0.65 ms of pure HBM streaming** per step — entirely launch + bandwidth bound, ~70× over the ~21 ms roofline.

Hypothesis:
- **Fuse.** Compute the whole 25-point Laplacian per point in ONE kernel (load each field tile + halo once into shared memory / registers, reuse for all 25 taps), and fuse the RK4 stages to avoid materializing k1..k4 and the `u_stage` resets. Collapses ~219 kernels → a handful, ~one field-read per sweep. PyTorch-level: `torch.compile` (Inductor should fuse the slice chain) for a quick first win; then a hand CUDA/Triton fused stencil.
- **Correctness watch (1e-6):** fp32 rel-eps ~1e-7, tol 1e-6 leaves ~10× headroom. A custom kernel must keep the reference's op structure closely; nvcc's default `-fmad=true` (contract mul+add → single-rounding FMA) differs from PyTorch's separate mul/add and may break 1e-6 — test early, likely need `-fmad=false` or explicit `__fadd_rn/__fmul_rn`.

Next step (Entry 1): `torch.compile` the reference; measure the fusion win and confirm 1e-6 still holds.

---

### Entry 1 — torch.compile (Inductor fuses the stencil + RK4)

Date: 2026-06-27

Thinking: the baseline's ~219 tiny kernels/step are pure launch+bandwidth waste. `torch.compile` should fuse the slice/add/mul chain into a few Triton kernels. The reference's in-place `copy_`/slice-assign/swap + `DeterministicContext` graph-break, so a **functional per-step** version with math identical to the reference was written and compiled.

Code version: `versions/v1_compile.py` (= `submission.py`).

Correctness: **pass** at 1e-6 — Inductor's fused fp32 codegen matches the eager reference within tolerance (the `-fmad`/reordering concern did NOT bite for Inductor here).

Performance: **212.8 ms** (vs baseline 1408.5 ms = **6.6×**). Beats naive Triton (317 ms), approaches naive CUDA (148 ms).

Profiling (torch.profiler, n_steps=2):

| | baseline (E0) | torch.compile (E1) |
|---|---|---|
| kernels/step | ~219 | **~5 substantial** (+ ~9 tiny scalar) |
| CUDA time/step | ~143 ms | ~21.8 ms |
| dominant kernel | — | `triton_poi_fused_add_copy_mul_slice_1` 6.79 ms ×2/step |
| ncu (dominant kernel) | — | DRAM 14 %, L2 41 %, SM 49 %, **occupancy 23 %** — occupancy/latency-limited, nothing saturated |

Observation:
- **Inductor fused ~219 → ~5 kernels/step → 6.6×.** Big, cheap win.
- **ncu shows the dominant Inductor kernel is occupancy-limited (23 %), not bandwidth-bound** (DRAM 14 %, L2 41 %, SM 49 %) — it under-utilizes everything. This is *why* torch.compile (213 ms) later loses to the hand Triton/CUDA (~85 ms), which reach 80 % occupancy / 90 % L2.
- **But it's still far from the ~21 ms roofline (218 ms here).** The dominant fused kernel is **6.79 ms ≈ 22 GB of HBM traffic ≈ 25 field-reads** — i.e. the fused stencil computes all 25 taps in one kernel but reads each shifted slice **from global memory separately (no neighbor reuse)**. Inductor generates a pointwise kernel; it does not stage a tile+halo in shared memory. Plus it still materializes the per-stage `k`/`u_stage` intermediates.

Hypothesis / next (Entry 2): hand-written **CUDA/Triton stencil with shared-memory halo reuse** — load each field tile + 4-cell halo **once** into shared memory, compute all 25 taps from SRAM (≈ 1 field-read instead of ~25), and fuse the RK4 stages to avoid materializing k1..k4. Should cut the dominant kernel ~10–25× toward the roofline. Correctness: must hold 1e-6 — replicate the reference's op order and test `-fmad=false` vs default early (Inductor passed, but a hand CUDA kernel with default FMA might not).

---

### Entry 2 — Hand-written Triton fused stencil + RK4

Date: 2026-06-27

Thinking: beat Inductor (which already generates Triton) by (a) cutting to **4 kernels/step** — 3 stage + 1 final, each fusing the 25-tap Laplacian + the stage combine + boundary-copy into one pass — and (b) getting neighbor reuse. (Plan was to go straight to CUDA; doing Triton first as a stepping stone + tile-strategy study.) RK4 stages have cross-tile halo dependencies, so they *must* be separate kernel launches (4/step); within each, the lap reuse comes from the **L2 cache** on the shifted `tl.load`s (Triton's block model can't cleanly express a 3D tile+halo in shared memory — that's the CUDA job for Entry 3).

Code version: `versions/v2_triton.py` (= `submission.py`). 25 taps as masked `tl.load`s at `base ± d`, `± d·Nx`, `± d·Ny·Nx`; module constants as `tl.constexpr`; one z-plane per program, 2D (y,x) tile.

Correctness: **pass at 1e-6** (max abs diff 4.77e-7 on 64³/3-steps) — Triton's fp32 codegen is within tol, same as Inductor.

Tile-strategy sweep (grid 600, 10 steps):

| (BLOCK_X, BLOCK_Y, warps) | time |
|---------------------------|------|
| **(128, 4, 4)** | **87.9 ms** ✅ best |
| (64, 4, 4) | 89.1 |
| (64, 8, 8) | 89.9 |
| (32, 8, 4) | 101.2 |
| (32, 32, 8) | 111.8 |
| (16, 16, 4) | 135.8 |

Best = **wide contiguous-x tile (BX=128), thin y (BY=4), 4 warps** — wide x maximizes coalescing and L2 reuse of the x-direction taps (±1..4 share cache lines); thin y keeps the working set small.

Performance (harness): **88.0 ms** — vs baseline 1408 (**16×**), vs torch.compile 212.8 (**2.4×**), vs naive CUDA 148 (**1.7×**), vs naive Triton 317 (3.6×).

Profiling (torch.profiler + ncu, large):

| | value |
|---|---|
| kernels/step | **4** (3 `_stage` + 1 `_final`) + 1 one-time DtoD clone |
| time/step | ~9.1 ms (`_stage` 2.09 ms ×3 + `_final` 2.54 ms) |
| ncu `_stage` | DRAM 46.9 %, Memory 64.0 %, Compute(SM) 42.7 %, 2.33 ms |

Observation:
- **88 ms, beats torch.compile by 2.4× and naive CUDA by 1.7×** — the win over Inductor is the 4-kernels/step fusion (no separate intermediate temporaries) + L2-cached reuse of the shifted loads.
- **But still ~2.3× over the per-stage roofline** (~1 ms: read field+u, write k+us ≈ 3.4 GB), and `_stage` only reaches **47 % DRAM** (not bandwidth-saturated). The remaining inefficiency: the **z-direction taps (z±1..4) are re-read ~9× across z-programs** (no reuse along z), plus 25-load latency / masking overhead. The x/y reuse is from L2; the z reuse is missing.

Hypothesis / next (Entry 3, CUDA): **2.5D blocking** — a 2D (x,y) tile in shared memory, march along z keeping the 9 z-planes (z−4..z+4) in registers so **each plane is loaded from HBM once** (kills the z-redundancy), compute the in-plane taps from shared memory. Should push `_stage` from 2.33 ms toward ~1 ms → roughly halve total toward the ~21 ms roofline. Plus `-fmad` control for 1e-6.

---

### Entry 3 — Naive CUDA (correctness gate for the CUDA path) + fmad finding

Date: 2026-06-27

Thinking: before the complex 2.5D kernel, write the simplest CUDA (one thread per point, 25-tap Laplacian read from global/cache, 4 kernels/step like the Triton) to (a) **nail the 1e-6 / FMA question** and (b) get a CUDA baseline.

Code version: `versions/v3_cuda_naive.py`. Scalars `ihx/S/dt` computed in **fp32 in C++** to match the reference's fp32 ops; coefficients as `float = double_literal` (rounds like PyTorch's scalar promotion).

**Correctness gate — the key result:** passes 1e-6 with **both** `--fmad=true` and `--fmad=false` (max abs diff **4.77e-7 = ~1 fp32 ulp, identical for both**). So the FMA-contraction worry was **unfounded** for this problem — a straightforward CUDA fp32 implementation matches the reference within tolerance regardless of fmad. (Good to know; no need to fight the compiler.)

Performance: **85.5 ms** (local timing) — **ties the Triton Entry 2 (88 ms)** and beats the README's naive CUDA (148 ms), because the 4 stages' combines + boundary-copy are fused into the kernels (their "naive" likely doesn't).

Profiling (ncu, `stage_k`, large):

| metric | value |
|--------|-------|
| **L2 Cache Throughput** | **90.4 %** (the limiter) |
| Memory Throughput | 90.4 % |
| DRAM Throughput | 50.1 % |
| Compute (SM) | 66.3 % |
| Achieved Occupancy | 80.2 % |
| Duration | 2.13 ms |

Observation: naive CUDA ≈ Triton (~85–88 ms), and ncu shows it is **L2-bandwidth-bound (90.4 % L2 throughput)**, *not* DRAM-bound (50 %) or compute-bound (66 %), at a healthy 80 % occupancy. **The 25 neighbor reads are served by L2 at ~90 % saturation — L2 is doing the stencil reuse.** (The earlier "~47 % DRAM" guess was wrong; the real picture is L2-limited.) This both explains why naive is fast *and* foreshadows Entry 4: since L2 already provides near-saturated reuse, manually moving reuse to shared memory can't help (and adds halo + barrier overhead). **No win over Entry 2 yet** (kept Triton/naive as best). To go faster one must cut L2 traffic (e.g. vectorized/coalesced loads), not add manual tiling.

Next (Entry 4): **2.5D blocking** — shared-memory (x,y) tile + register queue marching in z, so each plane is read once. Target: cut `_stage` toward ~1 ms / total toward the ~21 ms roofline.

---

### Entry 4 — CUDA 2.5D blocking (shared-mem tile + register z-march) — SLOWER, kept naive

Date: 2026-06-27

Thinking: the textbook fast 3D stencil — a 2D (x,y) tile in shared memory, march along z with a 9-deep register queue (z−4..z+4) so each plane's center is loaded once (kills z-redundancy). Boundary handled by initializing all output buffers to `u0` once (Dirichlet boundary is constant) → marching kernels write interior-only, no per-step clones.

Code version: `versions/v4_cuda_25d.py` (z-chunked variant). Block (BX=32,BY=8); `__shared__ float sm[BY+8][BX+8]`; per-thread `float q[9]`.

Correctness: **pass at 1e-6** (max abs diff 4.77e-7).

Performance — **SLOWER than naive**:

| variant | time |
|---------|------|
| naive CUDA (E3) | 85 ms |
| Triton (E2) | 88 ms |
| 2.5D, march all z (1 block/(x,y)-tile) | **150 ms** |
| 2.5D + z-chunking (ZC=32) | **134 ms** |

Profiling (ncu, the march kernel): **Achieved Occupancy 12.5 %** (theoretical 50 %), **DRAM 0.8 %, Compute 4 %** — the GPU is ~96 % idle; top stall is the `__syncthreads` barrier.

Observation — why the textbook technique LOST:
1. **Halo-overlap redundancy.** The shared-load reads a `(BX+8)×(BY+8)` tile+halo per z-step; for 32×8 that's `(40×16)/(32×8) = 2.5×` the field per stage — the in-plane redundancy *negates* the z-reuse. Bigger tiles cut it (64×16 → 1.7×) but cost shared mem / registers (occupancy ↓). Inescapable for radius-4.
2. **Sync + serialization.** 2 `__syncthreads`/z-step, and (non-chunked) only ~1425 long-running blocks → low occupancy, GPU starved. z-chunking helped (150→134) by adding blocks, but not enough.
3. **The big-L2 effect.** H100 has a **50 MB L2**; the naive kernel's 25 neighbor reads are *already* largely served from L2 (the reuse 2.5D does by hand happens automatically), **without** the halo-overlap waste or the barriers. So naive (216M threads, massive latency hiding, L2 reuse) beats hand 2.5D.

Conclusion: **kept naive CUDA / Triton (~85–88 ms) as best.** This is a genuine modern-GPU lesson — the classic shared-memory 2.5D stencil blocking (a win on older small-cache GPUs) can *lose* to a naive massively-parallel kernel on a large-L2 GPU, because L2 already provides the reuse and the explicit tiling only adds halo redundancy + barrier overhead. (Echoes Entry 2's finding that occupancy/parallelism, not manual reuse, was the lever.)

Lessons: (1) Don't assume the textbook optimization wins — measure. 2.5D blocking was the "obvious" answer and it was 1.6–1.8× *slower*. (2) On big-L2 GPUs, a naive massively-parallel stencil is hard to beat; manual shared-mem tiling fights the cache rather than helping it. (3) ncu's occupancy + DRAM% immediately showed the 2.5D was starved (12.5 % occ, <1 % DRAM), not reuse-limited.

---

### Entry 5 — CUDA x-register-coarsening (relieve the L2 bottleneck) — SLOWER, kept naive

Date: 2026-06-27

Thinking: Entry 3's ncu showed naive is **L2-bandwidth-bound (90 % L2, 50 % DRAM)** — i.e. the 25 loads/point × 216M points flood L2. DRAM at 50 % implies ~2× theoretical headroom *if* L2 traffic can be cut. Lever: each thread computes **CO consecutive x** and loads the contiguous x-run (CO+8 values) **once into registers**, reusing it for all CO outputs' x-derivative (x-direction loads 9/pt → ~3/pt). No shared memory, no barriers — should keep occupancy.

Code version: `versions/v5_cuda_coarsen.py`. Sweep over (BX, BY, CO).

Correctness: **pass at 1e-6** (4.77e-7, all configs).

Performance — **SLOWER, again**:

| (BX,BY,CO) | time |
|------------|------|
| naive (E3) | **85 ms** |
| (64,8,4) | 145 ms |
| (32,8,4) | 172 ms |
| (64,4,4) | 177 ms |
| (64,4,8) | 340 ms |
| (32,4,8) | 377 ms |

Profiling (ncu, coarsened (64,8,4) march): occupancy **65.8 %** (down from naive's 80 %), and **DRAM 6.6 %, L2 58 %, SM 40 %, Memory 64 % — nothing saturated.** The kernel went **latency-bound**: 4× work/thread + the `r[CO+8]` register array cut the thread count and raised register pressure → too few warps in flight to hide memory latency → every unit idle → slower, even though L2 *traffic* dropped (L2 90 %→58 %, DRAM 50 %→7 %).

Conclusion: **kept naive CUDA (85 ms) as the practical optimum.** This is the third traffic-reduction technique to lose (concat already n/a; 2.5D shared-mem; x-coarsening registers). The pattern is conclusive: **this stencil is memory-*latency*-bound at the kernel level, and it needs the naive kernel's ~216M-thread parallelism to hide that latency. Every technique that cuts threads to reduce L2/DRAM traffic starves the pipeline and loses more than it saves.** The L2-90 % "wall" is therefore *not* relievable on this access pattern — the DRAM-50 % headroom is real in theory but unreachable, because reaching it requires fewer memory ops per thread, which means fewer threads, which means latency-bound.

Answer to "what's the bottleneck": **L2 bandwidth (90 %)** for the naive — but it's a *balanced* L2-bound-at-high-occupancy kernel. The only way to cut L2 traffic (block/register reuse) sacrifices the parallelism that hides latency, so naive is the sweet spot. **~85 ms (16.6× over baseline) stands as the best.**

Lessons: (1) "Theoretical headroom" (DRAM 50 %) ≠ reachable — the techniques to claim it have side effects (parallelism loss) that dominate. (2) For a latency-bound memory kernel, **occupancy/parallelism is the currency**; trading it for locality is a net loss on a big-L2 GPU. (3) Three independent optimizations (2.5D, coarsening, and earlier the Triton tile sweep) all converged on the same conclusion — naive massive parallelism wins. Knowing when to stop optimizing is itself the result.
