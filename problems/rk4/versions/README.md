# RK4 — version history (code for each worklog Entry)

`submission.*` is gitignored; these preserve the code for each step in `../worklog.md`.
Benchmark: grid 600³, 10 steps. Baseline PyTorch = 1408 ms. Tolerance rtol=atol=1e-6.

| File | Entry | Runtime | Notes |
|------|-------|---------|-------|
| (baseline) `reference.ref_kernel` | 0 | 1408.5 ms | sliced stencil + RK4; ~219 memory-bound kernels/step |
| **`v1_compile.py`** | 1 | **212.8 ms** | ✅ torch.compile (functional per-step); ~5 fused kernels/step, 6.6×. Passes 1e-6. Stencil still reads 25 taps from HBM (no halo reuse) |
