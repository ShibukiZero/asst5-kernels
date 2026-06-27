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
