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

### Entry 1 — Physical transpose (PyTorch), to test the access-pattern hypothesis

Date: 2026-06-24

Thinking: if strided column reads are the problem, physically transposing `[length, channels] → [channels, length]` makes each channel contiguous; then per-channel bincount should read contiguous data. Predicted ~1.5 GB traffic (~3× ideal), much faster than baseline.

Code version:
- `submission.py`: `arr_t = array.t().contiguous()` (materialize transpose), then per-channel `bincount` on `arr_t[c]`.

Command: `./run.sh histogram benchmark` / `profile`

Correctness: **pass**

Performance:
- **Runtime: 45.36 ms** (mean of 20) vs baseline 51.31 ms → only **1.13×** faster. (Prediction missed — see below.)

Profiler stats (representative kernels):

| Kernel | DRAM read | L2 hit | DRAM thru | Note |
|--------|-----------|--------|-----------|------|
| `t().contiguous()` (1 big copy) | **34.35 GB** | 3.2% | 59.9% | the over-fetch **moved here** |
| `kernelHistogram1D` (per channel) | **1.06 MB** | 72% | — | now contiguous ✓ (was 67 MB) |
| `reduce` max + fills (per channel) | ~1.1 MB | 65% | — | still present, ×512 |

Time decomposition: transpose ≈ 34.35 GB / (3.35 TB/s × 60%) ≈ **17 ms**; remaining ≈ **28 ms** is the 512-iteration loop overhead (launches + max-reduce + fills), ~55 µs/channel even though bincount is now cheap.

Observation — prediction vs reality:
- ✅ Correct: transposing makes the per-channel bincount read contiguous (67 MB → 1.06 MB, L2 hit 3%→72%).
- ❌ Wrong: I assumed `.contiguous()` is a coalesced transpose (~0.5 GB). It is **not** — PyTorch does a naive element-wise copy whose **read side is still strided** → **34.35 GB read (~68× over-fetch)**. The over-fetch didn't vanish, it **relocated** into the copy.
- ❌ Wrong: predicted ~1.5 GB total / ~3× ideal. Actual ≈ 35 GB, ~1.13× faster.

Conclusion / lesson:
1. A physical transpose via generic `.contiguous()` just moves the same 64–68× strided over-fetch into the copy kernel — no net win unless you hand-write a shared-memory coalesced transpose (and even then you pay an extra full copy).
2. The 512-iteration Python loop is itself ~28 ms of fixed overhead, independent of access pattern.
3. → Both point at the **fused single kernel**: read the original array once, row-major/coalesced, and `atomicAdd` into per-channel bins — never materialize a transpose (saves 34 GB) and no per-channel loop (saves 28 ms).

---

### Entry 2 — Fused CUDA kernel, coalesced read, global atomics

Date: 2026-06-24

Thinking: read the original array once in row-major order so a warp's 32 threads read 32 consecutive channels (coalesced), and scatter counts with `atomicAdd` to a global `[channels, bins]` histogram — no transpose, no per-channel loop. Read-once is guaranteed by a bijection: each `(row, channel)` is handled by exactly one thread in one loop iteration.

Code version:
- `submission.cu` (compiled via `load_inline`). One `__global__ hist_kernel`:
  - `c = blockIdx.x*blockDim.x + threadIdx.x` → channel (consecutive threads → consecutive channels → coalesced).
  - grid-stride over rows: `for (r = blockIdx.y; r < length; r += gridDim.y)`.
  - `atomicAdd(&hist[c*num_bins + v], 1)` into the global histogram.
  - Launch: `block=256`, `grid=(ceil(512/256)=2, 2048)` → 4096 blocks, ~1.05M threads.

Command: `./run.sh histogram test|benchmark|profile`

Correctness: **pass**

Performance:
- **Runtime: 6.770 ms** (mean of 3) → **7.6× over baseline** (51.31 → 6.77 ms), 6.7× over the transpose version.

Profiler stats (`hist_kernel`, 6.85 ms):

| Metric | Value | Meaning |
|--------|-------|---------|
| **DRAM_Read** | **512.65 MB** | array read **exactly once** (was 68 GB) — 135× traffic gone |
| DRAM_Throughput | **2.35%** | DRAM now idle → **no longer memory-bound** |
| **L1→L2 Traffic** | **16.00 GB** | the global atomics |
| L2_Cache_Throughput | **88%** | **L2 is the new bottleneck** |
| L2_Cache_Hit_Rate | 97.7% | the 512 KB histogram lives in L2 |
| Compute_Throughput | 12.9% | compute idle too |

Observation:
- Both Entry-0 predictions confirmed: (1) reading once collapses DRAM traffic to the 512 MiB minimum and drops DRAM throughput to ~2%; (2) the new wall is **global-atomic traffic** — 537M `atomicAdd`s generate 16 GB of L1↔L2 traffic and saturate L2 (88%). The 6.85 ms is spent waiting on L2 atomics, not reading data (DRAM at 2.35% ⇒ the same 512 MB could stream in ~160 µs if not gated by atomics).
- Still ~42× off the ~160 µs roofline; the entire gap is atomics.

Hypothesis / next step:
- Privatize: each block accumulates a sub-histogram in **shared memory** (on-chip, ~100× faster than L2 atomics, no L2 traffic), then flushes once per bin to global. Should remove most of the 16 GB L2 traffic and return to DRAM-bound. Constraint: full 512 KB histogram > 228 KB shared/SM ⇒ tile over channels.

---

### Entry 3 — Shared-memory privatized sub-histograms (channel-tiled)

Date: 2026-06-24

Thinking: move the 537M atomics off L2 (the v2 wall) onto on-chip shared memory. Each block owns CH=32 channels and keeps a private `[CH x BINS]` sub-histogram in shared mem (32 KB, fits the per-block limit); threads accumulate with shared-mem atomics, then flush once to global. Coalesced global load preserved (warp = 32 consecutive channels of a row).

Code version (`submission.cu`):
- `__shared__ int s[CH*BINS]` with `CH=32, BINS=256` (32 KB/block).
- block `(CH, 16)` = 512 threads (x=local channel, y=row-lane); grid `(num_channels/CH, 128)` = `(16,128)` = 2048 blocks.
- zero shared → accumulate `atomicAdd(&s[lc*num_bins+v],1)` (grid-stride rows) → `__syncthreads()` → flush `atomicAdd(&hist[gc*num_bins+bin], s[...])`.

Command: `./run.sh histogram test|benchmark|profile`

Correctness: **pass**

Performance:
- **Runtime: 0.816 ms** (mean of 5) → **63× over baseline** (51.31 → 0.816 ms), **8.3× over v2** (6.77 ms). Now ~5× off the ~160 µs roofline.

Profiler stats (`hist_kernel`, 856 µs):

| Metric | Value | Meaning |
|--------|-------|---------|
| DRAM_Read | 514 MB | still read **once** ✓ |
| **L1→L2 Traffic** | **64 MB** | was 16 GB — global atomics gone (250× less) ✓ |
| DRAM_Throughput | 18.9% | up from 2.35%, still not saturated |
| L2 / L1 / Compute thru | 34% / 43% / 35% | **nothing saturated** |
| **Achieved occupancy** | **95.3%** | occupancy is NOT the limiter |
| **Shared-atomic bank conflicts** | **42.6 M** | ← suspected bottleneck — **WRONG**, disproven in Entry 4 |

Observation:
- Shared-memory privatization worked: L2 traffic collapsed 16 GB → 64 MB; L2 no longer the wall.
- But no resource is saturated (all 19–43%) **despite 95% occupancy** ⇒ warps are resident but **stalled**, not starved. The cause is **shared-memory bank conflicts** (42.6 M): `s[lc*256 + v]` has bank = `(lc*256+v) mod 32` = `v mod 32` — independent of `lc`, so a warp's 32 threads (varied values) collide heavily, serializing the shared atomics.

Hypothesis / next step (v4):
- Pad the per-channel stride 256 → 257: `s[lc*257 + v]` ⇒ bank = `(lc+v) mod 32` (257 mod 32 = 1). A warp's lc = 0..31 then spreads across all 32 banks regardless of value → conflict-free. Shared cost 32 KB → 32.1 KB (negligible). Re-measure bank conflicts and whether we become DRAM-bound.
- If same-address atomic contention (row-lanes hitting the same `(lc,v)`) shows up next, consider replicated sub-histograms.

> ⚠️ **Correction (after Entry 4):** the "bank conflicts are the bottleneck" attribution above was **wrong** — inferred from a nonzero metric without measuring warp stall reasons. Padding (Entry 4) disproved it; the real bottleneck is global-load latency. Kept here to show the (mistaken) reasoning at the time.

---

### Entry 4 — Bank-conflict padding (256 → 257) — NO EFFECT (failed)

Date: 2026-06-24

Thinking: pad shared stride to 257 to make bank = `(lc+v) mod 32`, expecting conflict-free shared atomics.

Code version (`submission.cu`): `SPADC = BINS+1 = 257`; `s[lc*spad + v]`; shared 32 KB → 32.1 KB. Everything else identical to Entry 3.

Correctness: **pass**

Performance:
- **Runtime: 0.813 ms** (mean of 5) — **unchanged** from Entry 3 (0.816 ms).

Profiler stats:

| Metric | Entry 3 | Entry 4 (padded) |
|--------|---------|------------------|
| Shared-atomic bank conflicts | 42.6 M | **42.99 M (unchanged)** |
| Achieved occupancy | 95.3% | 95.7% |
| DRAM / L1 / SM throughput | 19/43/35% | 19/44/34% |
| Duration | 856 µs | 852 µs |

Why it failed (the lesson):
- My padding reasoning assumed all threads in a warp write the **same** value v (then bank = `(lc+v)%32` is a perfect permutation). But the data is **uniform-random**: each lane's `v_lc` is independent → bank = `(lc+v_lc)%32` is still random → **same conflict rate**. **Padding only removes bank conflicts for correlated/identical writes; for random values it does nothing.**
- More importantly, padding moving the runtime by ~0 shows **bank conflicts were never the real bottleneck** (42.6 M conflicts is only ~8% of the 537 M shared atomics; the Entry-3 hypothesis was wrong).

Re-diagnosis attempt #1 (ALSO WRONG): I then guessed "raw shared-atomic throughput" — again from indirect signals, without measuring. Disproven below.

Re-diagnosis #2 — measured warp stall reasons (the *direct* signal):

| Stall reason (per issued inst) | Value |
|--------------------------------|-------|
| **long_scoreboard** (waiting on global-memory load) | **38.0** ← dominant |
| barrier (`__syncthreads`) | 0.58 |
| short_scoreboard (shared memory) | 0.13 |
| **mio_throttle** (shared-atomic / MIO pipe) | **0.01** |
| lg_throttle | 0 |
| issue_active (warps actually issuing) | 34.6% |

**True bottleneck: global-load latency (latency-bound).** `mio_throttle ≈ 0` proves shared atomics are NOT the limiter; `short_scoreboard ≈ 0` rules out shared memory / bank conflicts; DRAM at 19% rules out bandwidth. Warps stall on `long_scoreboard` because each thread does **1-byte load → dependent atomicAdd → …**: memory-level parallelism is too low to hide the ~hundreds-of-cycles global load latency, even at 95% occupancy.

**Method lesson (this cost two wrong calls — Entry 3 bank-conflicts, and re-diagnosis #1):** when no resource is saturated, **pull the stall-reason breakdown before naming a bottleneck.** Do not infer causation from a metric merely being nonzero/large (42.6 M bank conflicts looked damning but was ~8% noise).

Corrected implication: the ~160 µs read roofline may actually be **reachable** — DRAM sits at 19% because we're latency-bound, not because counting is intrinsically expensive. Hiding the load latency should let DRAM throughput climb.

Next step (v5): raise memory-level parallelism to hide the load — **vectorized loads** (each thread reads `uchar4`/`int` = 4 channels) and/or **unroll the row loop** (several independent loads in flight before the dependent atomics). Validate by checking `long_scoreboard` drops and DRAM throughput rises. (This diagnosis stays unconfirmed until v5 moves the needle — applying the lesson, not trusting it on faith.)

---

### Entry 5 — Row-loop unroll (×8) on the v3 base; padding reverted

Date: 2026-06-24

Thinking: Entry 4 diagnosed latency-bound on the global load. v4's padding was a no-op, so reverted it (back to the clean v3 layout, `s[lc*num_bins+v]`) and instead raised memory-level parallelism: unroll the grid-stride row loop by UNROLL=8 so each thread issues 8 *independent* loads (into registers) before the 8 dependent atomics → 8 loads in flight to hide the ~hundreds-of-cycles load latency.

Code version (`submission.cu`): v3 + `#define UNROLL 8`; unrolled body loads `v[8]` then does 8 `atomicAdd`. Same mapping/grid as v3. Read-once bijection unchanged (loads just regrouped).

Correctness: **pass**

Performance:
- **Runtime: 0.359 ms** (mean of 3) → **143× over baseline**, **2.3× over v3** (0.816 ms). Now ~2.2× off the ~160 µs roofline.

Validation (re-measured stall reasons — diagnosis confirmed, not assumed):

| Metric | v4 | v5a | |
|--------|----|----|--|
| long_scoreboard (global-load wait) | 38.0 | **6.01** | ↓6× — latency now largely hidden ✓ |
| DRAM throughput | 19% | **39%** | ↑2× — feeding faster ✓ |
| issue_active | 34.6% | **64.7%** | warps issue ~2× more ✓ |
| mio_throttle (shared atomics) | 0.01 | 1.94 | shared atomics now emerging |
| short_scoreboard | 0.13 | 0.95 | shared memory emerging |

Observation:
- The latency-bound diagnosis is **confirmed by experiment**: adding MLP dropped `long_scoreboard` 38→6 and doubled DRAM throughput. This is the validation that Entry 3/4's guesses lacked.
- New state is more balanced: `long_scoreboard` (6.0) still the top stall but much smaller; shared-atomic (`mio_throttle` 1.94) and shared-memory (`short_scoreboard` 0.95) now visible. DRAM at 39% ⇒ ~2.5× headroom to bandwidth saturation.

Next step (v5b): vectorized loads — each thread reads `uchar4`/`int` (4 channels) per load → 4× fewer load instructions and more bytes/request, pushing DRAM higher. Also consider larger UNROLL. Watch whether shared atomics (`mio_throttle`) become the next wall.

---

