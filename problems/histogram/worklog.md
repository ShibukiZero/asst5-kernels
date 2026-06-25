# Histogram Work Log

## Problem

Optimize multi-channel histogram computation for the H100 leaderboard.

- Input `array`: `[length, num_channels]` = `[1048576, 512]`, dtype **uint8**, values in `[0, num_bins-1]`.
- Output `histogram`: `[num_channels, num_bins]` = `[512, 256]`, int32. `histogram[c][b]` = count of value `b` in channel `c`.
- 512 independent histograms, each over ~1M elements into 256 bins.

### Roofline (the target)

| Quantity | Value |
|----------|-------|
| Input to read | 512 MiB (2^29 bytes) |
| Output to write | 512 KiB (negligible) |
| Arithmetic | ~1 increment / element → essentially zero |
| **Memory-bound lower bound** (H100 SXM, HBM3 ≈ 3.35 TB/s) | **512 MiB / 3.35 TB/s ≈ 160 µs** |

A perfect kernel reads the input **once** and counts. North star ≈ **160 µs**.

Hardware: NVIDIA H100 80GB HBM3 (Nebius VM), CUDA 13, torch 2.12.1+cu130, ncu 2025.3.1.

---

## Optimization Entries

Each entry: how I thought about it, what I changed, the result, and the profiling. Failures recorded too.

### Entry 0 — Baseline (PyTorch reference)

Date: 2026-06-24

Code version:
- File: `submission.py` = `reference.ref_kernel`
- Description: Python `for c in range(512)` loop; per channel: take column `array[:, c]` (a **strided** view), `torch.bincount(...)`, write row.

Command:
```bash
./run.sh histogram benchmark
```

Correctness:
- Status: **pass**

Performance:
- **Runtime: 51.314 ms** (mean of 3; std 0.080 ms, best 51.238 ms)
- vs ideal ~160 µs → **≈ 320× slower than the read-bound limit**.

Profiler stats (ncu, representative per-channel kernels — full 512-channel profile is impractical/wasteful, but every channel is near-identical):

| Kernel (per channel) | DRAM read | Duration | DRAM thru | L2 hit |
|----------------------|-----------|----------|-----------|--------|
| `direct_copy` (materialize column `array[:,c]`) | **67.1 MB** | 38.2 µs | 54.6% | 9.0% |
| `reduce` (bincount computes max) | 1.1 MB | 9.0 µs | 3.7% | 65% |
| `kernelHistogram1D` (atomic bincount) | **67.1 MB** | 36.9 µs | 56.5% | 11% |
| + small fills/casts | ~4–35 KB | ~2–3 µs | — | — |
| **≈ per channel** | **~134 MB** | **~84–89 µs** | — | — |

(One-time `randint` in `generate_input` is outside the timed region.)

Observation — what limits performance:
1. **Strided column over-fetch (root cause).** Processing 1 column = 1 MB of useful data costs **67 MB of DRAM reads**. `array[:, c]` has stride `num_channels`=512 B, so each 32/128 B memory sector delivers only 1 useful byte → **~64× over-fetch**. L2 hit rate is ~9–11% (no locality across the strided reads).
2. **Two strided passes per channel** (copy + histogram) ≈ 134 MB/channel × 512 ≈ **~68 GB total DRAM traffic** vs **0.5 GB** ideal → ~130× wasted bandwidth.
3. **Launch overhead.** 512 iterations × ~4–6 kernels ≈ ~2–3k kernel launches.
4. Even the big kernels only hit ~55% DRAM throughput — strided access can't saturate HBM.

Hypothesis:
- The kernel is memory-bound and we are wasting >100× bandwidth purely on the **access pattern** (strided column reads) and reading the array many times instead of once. A single fused kernel that reads the array **once, coalesced** (consecutive threads → consecutive channels within a row) should collapse the runtime toward the ~160 µs roofline.

Next step:
- Replace the Python loop with **one fused kernel** that streams the array once with coalesced reads, then attack atomic contention.

---

