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

Thinking: the baseline's ~219 tiny kernels/step are pure launch+bandwidth waste. `torch.compile` should fuse the slice/add/mul chain into a few Triton kernels. The reference's in-place `copy_`/slice-assign/swap + `DeterministicContext` graph-break, so I rewrote a **functional per-step** version with math identical to the reference and compiled that.

Code version: `versions/v1_compile.py` (= `submission.py`).

Correctness: **pass** at 1e-6 — Inductor's fused fp32 codegen matches the eager reference within tolerance (the `-fmad`/reordering concern did NOT bite for Inductor here).

Performance: **212.8 ms** (vs baseline 1408.5 ms = **6.6×**). Beats naive Triton (317 ms), approaches naive CUDA (148 ms).

Profiling (torch.profiler, n_steps=2):

| | baseline (E0) | torch.compile (E1) |
|---|---|---|
| kernels/step | ~219 | **~5 substantial** (+ ~9 tiny scalar) |
| CUDA time/step | ~143 ms | ~21.8 ms |
| dominant kernel | — | `triton_poi_fused_add_copy_mul_slice_1` 6.79 ms ×2/step |

Observation:
- **Inductor fused ~219 → ~5 kernels/step → 6.6×.** Big, cheap win.
- **But it's still far from the ~21 ms roofline (we're at 218 ms).** The dominant fused kernel is **6.79 ms ≈ 22 GB of HBM traffic ≈ 25 field-reads** — i.e. the fused stencil computes all 25 taps in one kernel but reads each shifted slice **from global memory separately (no neighbor reuse)**. Inductor generates a pointwise kernel; it does not stage a tile+halo in shared memory. Plus it still materializes the per-stage `k`/`u_stage` intermediates.

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

Performance: **85.5 ms** (my timing) — **ties the Triton Entry 2 (88 ms)** and beats the README's naive CUDA (148 ms), because we fuse the 4 stages' combines + boundary-copy into the kernels (their "naive" likely doesn't).

Observation: naive CUDA ≈ Triton because both read the 25 taps from global with L2 caching and neither reuses the z-direction — same ~2.3× over roofline, same ~47% DRAM. **No win over Entry 2 yet** (kept Triton as best). This is the baseline the 2.5D kernel must beat.

Next (Entry 4): **2.5D blocking** — shared-memory (x,y) tile + register queue marching in z, so each plane is read once. Target: cut `_stage` toward ~1 ms / total toward the ~21 ms roofline.
