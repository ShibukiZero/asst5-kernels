# Histogram — version history (code for each worklog Entry)

`submission.*` is gitignored (course convention), so the working kernel is not
tracked. These files preserve the code for each step described in `../worklog.md`.

| File | Entry | Runtime | Notes |
|------|-------|---------|-------|
| (baseline) `reference.ref_kernel` | 0 | 51.31 ms | PyTorch per-channel `bincount` loop |
| `v1_transpose.py` | 1 | 45.36 ms | ❌ transpose: over-fetch relocated |
| `v2_fused_global_atomics.cu` | 2 | 6.77 ms | read once, global atomics |
| `v3_shared_privatized.cu` | 3 | 0.816 ms | shared-mem sub-histograms |
| `v4_bankconflict_padding.cu` | 4 | 0.813 ms | ❌ padding: no effect |
| **`v5a_row_unroll.cu`** | 5 | **0.359 ms** | ✅ BEST — unroll hides load latency |
| `v6_vectorized_int.cu` | 6 | 0.400 ms | ❌ vectorize: occupancy crash |

To run a version: copy it to `../submission.cu` (or `../submission.py` for v1),
then `python ../wrap_cuda_submission.py local` and `./run.sh histogram <mode>`.
