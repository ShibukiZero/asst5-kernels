# RK4 — version history (code for each worklog Entry)

`submission.*` is gitignored; these preserve the code for each step in `../worklog.md`.
Benchmark: grid 600³, 10 steps. Baseline PyTorch = 1408 ms. Tolerance rtol=atol=1e-6.

| File | Entry | Runtime | Notes |
|------|-------|---------|-------|
| (baseline) `reference.ref_kernel` | 0 | 1408.5 ms | sliced stencil + RK4; ~219 memory-bound kernels/step |
| `v1_compile.py` | 1 | 212.8 ms | ✅ torch.compile (functional per-step); ~5 fused kernels/step, 6.6×. Passes 1e-6. Stencil still reads 25 taps from HBM (no halo reuse) |
| `v2_triton.py` | 2 | 88.0 ms | hand Triton, 4 fused kernels/step, L2 reuse, tile BX=128/BY=4. 16×, beats torch.compile 2.4×. Passes 1e-6 |
| `v3_cuda_naive.py` | 3 | 85.5 ms | naive CUDA (1 thread/point, L2-cached); correctness gate (fmad doesn't matter). Beats README naive CUDA (148) via fused combines |
| `v4_cuda_25d.py` | 4 | 134 ms | ⚖️ CUDA 2.5D blocking (shmem tile + register z-march, z-chunked). Correct, but SLOWER: halo-overlap redundancy + barriers; H100's 50MB L2 already gives naive its reuse. Textbook technique loses to naive-on-big-L2 |
| `v5_cuda_coarsen.py` | 5 | 145 ms | ⚖️ CUDA x-register-coarsening to cut L2 traffic. Correct, but SLOWER: 4× work/thread + register pressure → occupancy/parallelism loss → latency-bound (nothing saturated). naive parallelism wins again |
| `v6_cuda_xshfl.py` | 6 | **83.087 ms** | ✅ **BEST** — warp-shuffle x-reuse + TPB=160 + 32-bit idx. +2.8%; L2 went UP to 97% (efficiency win, not L2 relief — x was already L1-cheap) |
| `v7_plane_tile.py` | 7 | 89.583 ms | ⚖️ per-z-plane shared x/y tile (no z-march). LOSES despite cutting L2 97→62% AND holding occupancy 84% — barrier collapses MLP → latency-bound. Refutes "occupancy was the problem" |
