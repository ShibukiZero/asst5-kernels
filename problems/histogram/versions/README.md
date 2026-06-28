# Histogram — version history (code for each worklog Entry)

`submission.*` is gitignored (course convention), so the working kernel is not
tracked. These files preserve the code for each step described in `../worklog.md`.

Note: the `vN` code tags are stable filenames; the **Entry** column matches the
worklog after Entry 2 (best-effort Triton) was inserted, so Entry N (N≥3) maps to
code `v(N-1)`.

| File | Entry | Runtime | Notes |
|------|-------|---------|-------|
| (baseline) `reference.ref_kernel` | 0 | 51.31 ms | PyTorch per-channel `bincount` loop |
| `v1_transpose.py` | 1 | 45.36 ms | ❌ transpose: over-fetch relocated |
| `v2a_triton_atomics.py` | 2 (A) | 11.04 ms | Triton global atomics — reproduces CUDA's L2-atomic wall, ~1.7× slower |
| `v2b_triton_onehot.py` | 2 (B) | 629.18 ms | ❌ Triton one-hot privatize — 256× compute blow-up, L1/TEX-bound, 12% occ |
| `v2c_triton_tl_histogram.py` | 2 (C) | **4.107 ms** | ✅ best Triton — `tl.histogram`, strided reads absorbed by L2 (L2-bound) |
| `v2_fused_global_atomics.cu` | 3 | 6.77 ms | read once, global atomics |
| `v3_shared_privatized.cu` | 4 | 0.816 ms | shared-mem sub-histograms |
| `v4_bankconflict_padding.cu` | 5 | 0.813 ms | ❌ padding: no effect |
| `v5a_row_unroll.cu` | 6 | 0.359 ms | unroll hides load latency |
| `v6_vectorized_int.cu` | 7 | 0.400 ms | ❌ vectorize: occupancy crash |
| `v7_grid_tuned.cu` | 8 | 0.343 ms | tuned launch (CH=32, by=32, gy=32) |
| `v8_ch64_vec2.cu` | 9 | 0.297 ms | CH=64 + uint16 vec load (occupancy-preserving) + lane-major conflict-free shared; L1TEX 91.6%→42% |
| **`v9_ch128_vec4.cu`** | 10 | **0.270 ms** | ✅ BEST overall — CH=128 + uint32 (uchar4) vec load + lane-major shared; DRAM→54%, won at 50% occupancy via ILP |

To run a CUDA version: copy it to `../submission.cu`, then
`python ../wrap_cuda_submission.py local` and `./run.sh histogram <mode>`.
To run a Triton/PyTorch version (`v1`, `v2a/b/c`): copy it to `../submission.py`
directly (no wrap step) and `./run.sh histogram <mode>`.
