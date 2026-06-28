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

Each entry: how it was thought about, what changed, the result, and the profiling. Failures recorded too.

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
- The kernel is memory-bound and >100× bandwidth is wasted purely on the **access pattern** (strided column reads) and reading the array many times instead of once. A single fused kernel that reads the array **once, coalesced** (consecutive threads → consecutive channels within a row) should collapse the runtime toward the ~160 µs roofline.

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
- ❌ Wrong: `.contiguous()` was assumed to be a coalesced transpose (~0.5 GB). It is **not** — PyTorch does a naive element-wise copy whose **read side is still strided** → **34.35 GB read (~68× over-fetch)**. The over-fetch didn't vanish, it **relocated** into the copy.
- ❌ Wrong: predicted ~1.5 GB total / ~3× ideal. Actual ≈ 35 GB, ~1.13× faster.

Conclusion / lesson:
1. A physical transpose via generic `.contiguous()` just moves the same 64–68× strided over-fetch into the copy kernel — no net win unless you hand-write a shared-memory coalesced transpose (and even then you pay an extra full copy).
2. The 512-iteration Python loop is itself ~28 ms of fixed overhead, independent of access pattern.
3. → Both point at the **fused single kernel**: read the original array once, row-major/coalesced, and `atomicAdd` into per-channel bins — never materialize a transpose (saves 34 GB) and no per-channel loop (saves 28 ms).

---

### Entry 2 — Best-effort Triton (before dropping to CUDA): three approaches

Date: 2026-06-27

Thinking: Entry 1 left a clear decision point — strided over-fetch is the
root cause, a physical transpose just relocates it, so the answer is a **fused
single-pass kernel that reads once, coalesced**. Before hand-writing CUDA, what
is the *best a competent Triton author* can do here? Triton is the obvious reach
for a fused kernel. The catch: Triton hides shared memory (SRAM is auto-managed)
and offers **no cheap privatized scatter**, which turns out to be exactly the
optimization that makes histogram fast. So this entry tries all three Triton
expressions and measures each, to find Triton's real ceiling on this problem.

The Triton trilemma — you cannot get {coalesced read + cheap counting + no
atomics} at once:

- **A. Global atomics** (`tl.atomic_add` per element). Coalesced read (a program
  owns a strip of consecutive channels), but the scatter is one global atomic per
  element → the same wall as CUDA's global-atomic version.
- **B. One-hot privatization** (the "Triton-idiomatic" no-atomics answer, the
  moral equivalent of CUDA shared-mem sub-histograms). Build a private
  `[NUM_BINS, BLOCK_C]` histogram in registers by comparing the loaded tile
  against all bins (3D broadcast) and `tl.sum`-reducing over rows; flush once.
  No per-element atomics — but pays a **NUM_BINS× compute blow-up** (256
  comparisons per element) and there is no cheap scatter into a register
  accumulator.
- **C. Built-in `tl.histogram`**. Efficient counting, but it reduces a *flattened
  1D block*, so it can't keep channels separate → each program must own **one
  channel** and read that channel's **column**, which is strided by C=512 → the
  ~64× over-fetch of Entry 0/1 returns.

Code versions: `versions/v2a_triton_atomics.py`, `v2b_triton_onehot.py`,
`v2c_triton_tl_histogram.py`. Each defines `custom_kernel` directly (no
`wrap_cuda_submission.py` — that is only for `.cu`).

Hardware/stack: same H100 (Nebius), torch 2.12.1+cu130, **triton 3.7.1**, ncu 2025.3.1.

Correctness: **all three pass.**

Performance (benchmark mean):

| Variant | Runtime | vs CUDA fused (Entry 3, 6.77 ms) | vs ideal 160 µs |
|---------|---------|----------------------------------|-----------------|
| A. global atomics | 11.035 ms (best 10.810) | 1.6× **slower** | ~69× |
| B. one-hot privatize | **629.181 ms** | ~93× slower | ~3900× |
| **C. `tl.histogram` (strided)** | **4.107 ms** (best 3.848) | **1.6× faster** | ~26× |

Profiler stats (ncu) — the bottleneck for each:

| Metric | A atomics | B one-hot | C tl.histogram |
|--------|-----------|-----------|----------------|
| Duration | 11.59 ms | (783 ms under ncu) | 4.12 ms |
| **DRAM_Read** | 515 MB (once ✓) | — | **1.75 GB** (over-fetch back, but ~3.4× not 64×) |
| L2→L1 / L1→L2 | 16.5 / 16.0 GB | — | **16.0 GB** L2→L1 |
| **Bottleneck pipe** | **L2 87.96%** | **L1/TEX 97.13%** | **L2 91.62%** |
| DRAM_Throughput | 1.40% | 0.03% | 13.66% |
| L2_Cache_Hit_Rate | 98.69% | — | 88.55% |
| Compute (SM) | 9.48% | 14.70% | 57.45% |
| Achieved Occupancy | — | **12.43%** | — |

Observation — what each result teaches:

- **A (11.0 ms) reproduces CUDA Entry 3's wall exactly** — 16 GB of L1↔L2 atomic
  traffic, L2 saturated at 88%, DRAM idle (1.4%). Same picture as v2 (16 GB, L2
  88%, DRAM 2.35%), but **~1.7× slower** (11.59 vs 6.85 ms): Triton's atomic
  codegen + masked 2D pointer arithmetic is less efficient than the hand-rolled
  CUDA. L2 is already saturated, so tuning BLOCK_C/BLOCK_R cannot help.
- **B (629 ms) is the catastrophic one** — the no-atomics privatization that wins
  in CUDA is a disaster in Triton. It is **L1/TEX-pipe bound (97%) with 12%
  occupancy**: the 256× one-hot materialization (`[NUM_BINS, BLOCK_R, BLOCK_C]`
  intermediates) thrashes the L1/TEX pipe, and the `[256, BLOCK_C]` register
  accumulator crushes occupancy. SM/compute is only 15% — the work is moving
  data through the SRAM pipe, not computing. One-hot is only viable for *small*
  bin counts; at 256 it is ~57× slower than even the naive atomic version.
- **C (4.1 ms) is the surprise winner** — and beats CUDA's naive fused-atomic v2.
  The strided column read *does* bring over-fetch back, but only ~3.4× (1.75 GB),
  not 64×: all 512 "one-channel" programs run concurrently and **share each
  row's 512 bytes in L2** (L2 hit 88.55%), so the over-fetch is absorbed by L2
  (16 GB L2→L1) instead of hammering DRAM. Combined with `tl.histogram`'s
  atomic-free counting, it lands at 4.1 ms — **L2-bound** at 91.6%.

Conclusion — **best-effort Triton ≈ 4.1 ms (Variant C)**, respectable (it beats
the naive CUDA fused-atomic kernel) but it **plateaus ~12× behind hand-written
CUDA's best (0.343 ms) and ~26× off the 160 µs roofline.** The reason is
structural, not a tuning miss: the step that takes CUDA from 6.77 ms → 0.816 ms
(Entry 3→4: shared-memory privatized sub-histograms, coalesced) requires
user-managed shared memory and cheap shared-atomics — and **Triton's abstraction
has no equivalent.** Its three escape hatches each fail differently: A can't
privatize (L2-atomic wall), B privatizes but the one-hot is compute-catastrophic,
C counts efficiently but pays strided reads. Triton excels at fused
elementwise/matmul shapes; histogram's privatized-scatter pattern is precisely
where giving up shared-memory control costs you the key optimization.

Next step: to break past 4 ms shared memory must be hand-managed — i.e. drop to
CUDA. Entry 3 is the same idea as Variant A (fused, coalesced, global atomics)
but in CUDA, which then unlocks the shared-mem privatization Triton can't express.

---

### Entry 3 — Fused CUDA kernel, coalesced read, global atomics

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

### Entry 4 — Shared-memory privatized sub-histograms (channel-tiled)

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
| **Shared-atomic bank conflicts** | **42.6 M** | ← suspected bottleneck — **WRONG**, disproven in Entry 5 |

Observation:
- Shared-memory privatization worked: L2 traffic collapsed 16 GB → 64 MB; L2 no longer the wall.
- But no resource is saturated (all 19–43%) **despite 95% occupancy** ⇒ warps are resident but **stalled**, not starved. The cause is **shared-memory bank conflicts** (42.6 M): `s[lc*256 + v]` has bank = `(lc*256+v) mod 32` = `v mod 32` — independent of `lc`, so a warp's 32 threads (varied values) collide heavily, serializing the shared atomics.

Hypothesis / next step (v4):
- Pad the per-channel stride 256 → 257: `s[lc*257 + v]` ⇒ bank = `(lc+v) mod 32` (257 mod 32 = 1). A warp's lc = 0..31 then spreads across all 32 banks regardless of value → conflict-free. Shared cost 32 KB → 32.1 KB (negligible). Re-measure bank conflicts and whether the kernel becomes DRAM-bound.
- If same-address atomic contention (row-lanes hitting the same `(lc,v)`) shows up next, consider replicated sub-histograms.

> ⚠️ **Correction (after Entry 5):** the "bank conflicts are the bottleneck" attribution above was **wrong** — inferred from a nonzero metric without measuring warp stall reasons. Padding (Entry 5) disproved it; the real bottleneck is global-load latency. Kept here to show the (mistaken) reasoning at the time.

---

### Entry 5 — Bank-conflict padding (256 → 257) — NO EFFECT (failed)

Date: 2026-06-24

Thinking: pad shared stride to 257 to make bank = `(lc+v) mod 32`, expecting conflict-free shared atomics.

Code version (`submission.cu`): `SPADC = BINS+1 = 257`; `s[lc*spad + v]`; shared 32 KB → 32.1 KB. Everything else identical to Entry 4.

Correctness: **pass**

Performance:
- **Runtime: 0.813 ms** (mean of 5) — **unchanged** from Entry 4 (0.816 ms).

Profiler stats:

| Metric | Entry 4 | Entry 5 (padded) |
|--------|---------|------------------|
| Shared-atomic bank conflicts | 42.6 M | **42.99 M (unchanged)** |
| Achieved occupancy | 95.3% | 95.7% |
| DRAM / L1 / SM throughput | 19/43/35% | 19/44/34% |
| Duration | 856 µs | 852 µs |

Why it failed (the lesson):
- The padding reasoning assumed all threads in a warp write the **same** value v (then bank = `(lc+v)%32` is a perfect permutation). But the data is **uniform-random**: each lane's `v_lc` is independent → bank = `(lc+v_lc)%32` is still random → **same conflict rate**. **Padding only removes bank conflicts for correlated/identical writes; for random values it does nothing.**
- More importantly, padding moving the runtime by ~0 shows **bank conflicts were never the real bottleneck** (42.6 M conflicts is only ~8% of the 537 M shared atomics; the Entry-4 hypothesis was wrong).

Re-diagnosis attempt #1 (ALSO WRONG): "raw shared-atomic throughput" was then guessed — again from indirect signals, without measuring. Disproven below.

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

**Method lesson (this cost two wrong calls — Entry 4 bank-conflicts, and re-diagnosis #1):** when no resource is saturated, **pull the stall-reason breakdown before naming a bottleneck.** Do not infer causation from a metric merely being nonzero/large (42.6 M bank conflicts looked damning but was ~8% noise).

Corrected implication: the ~160 µs read roofline may actually be **reachable** — DRAM sits at 19% because the kernel is latency-bound, not because counting is intrinsically expensive. Hiding the load latency should let DRAM throughput climb.

Next step (v5): raise memory-level parallelism to hide the load — **vectorized loads** (each thread reads `uchar4`/`int` = 4 channels) and/or **unroll the row loop** (several independent loads in flight before the dependent atomics). Validate by checking `long_scoreboard` drops and DRAM throughput rises. (This diagnosis stays unconfirmed until v5 moves the needle — applying the lesson, not trusting it on faith.)

---

### Entry 6 — Row-loop unroll (×8) on the v3 base; padding reverted

Date: 2026-06-24

Thinking: Entry 5 diagnosed latency-bound on the global load. v4's padding was a no-op, so reverted it (back to the clean v3 layout, `s[lc*num_bins+v]`) and instead raised memory-level parallelism: unroll the grid-stride row loop by UNROLL=8 so each thread issues 8 *independent* loads (into registers) before the 8 dependent atomics → 8 loads in flight to hide the ~hundreds-of-cycles load latency.

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
- The latency-bound diagnosis is **confirmed by experiment**: adding MLP dropped `long_scoreboard` 38→6 and doubled DRAM throughput. This is the validation that Entry 4/5's guesses lacked.
- New state is more balanced: `long_scoreboard` (6.0) still the top stall but much smaller; shared-atomic (`mio_throttle` 1.94) and shared-memory (`short_scoreboard` 0.95) now visible. DRAM at 39% ⇒ ~2.5× headroom to bandwidth saturation.

Next step (v5b): vectorized loads — each thread reads `uchar4`/`int` (4 channels) per load → 4× fewer load instructions and more bytes/request, pushing DRAM higher. Also consider larger UNROLL. Watch whether shared atomics (`mio_throttle`) become the next wall.

---

### Entry 7 — Vectorized int loads (4 ch/thread) + unroll — SLOWER (failed)

Date: 2026-06-24

Thinking: each thread reads one 32-bit word = 4 consecutive channels (4× fewer load instructions, more bytes/request), kept the ×8 unroll. Expected DRAM to climb past 39%.

Code version (`submission.cu`): `CHX=CH/4=8` threads in x, each reads `int` via `reinterpret_cast<const uint32_t*>(data)`, unpacks 4 bytes → 4 atomics. block `(8,16)`=128 threads, grid `(16,256)`.

Correctness: **pass**

Performance:
- **Runtime: 0.400 ms** — **slower** than v5a (0.359 ms) by ~11%.

Profiler stats vs v5a:

| Metric | v5a | v6 (vectorized) | |
|--------|-----|-----------------|--|
| **Achieved occupancy** | 92.7% | **35.5%** | ← crashed |
| DRAM throughput | 39% | 37% | similar |
| issue_active | 64.7% | 43.8% | down |
| long_scoreboard | 6.01 | 4.72 | vectorize+unroll hid the load *better* |
| short_scoreboard | 0.95 | 2.65 | shared mem up |

Observation:
- Vectorization did its job on the load (long_scoreboard 6→4.7), **but it forced a block-shape change** (4 channels/thread ⇒ only 8 threads in x ⇒ 128-thread blocks); with 32 KB shared that **crashed occupancy 92.7%→35.5%**, and the lost warps outweighed the instruction savings → net slower.
- Lesson: an optimization that improves one metric can regress a more important one. Vectorization here is only worth it if occupancy is preserved — e.g. CH=128 with `int` loads keeps 32 threads/x (512-thread blocks) but needs 128 KB **dynamic** shared (opt-in via `cudaFuncSetAttribute`). Not pursued now (budget).

Decision: **keep v5a (0.359 ms, 143×) as the best version.** Reverted `submission.cu` to v5a.

---

### Entry 8 — Grid/block sweep (launch-config tuning) — FINAL

Date: 2026-06-25

Thinking: at v5a, occupancy was already ~93%, so the launch params (CH=32, blockDim.y=16, gridDim.y=128, UNROLL=8) were untuned guesses. Swept them to find the sweet spot. Made `blockDim.y`/`gridDim.y` env-configurable (one compile, sweep via env); CH via recompile.

Sweep (best per channel-tile size, benchmark mean):

| CH | best (BY, GY) | best time |
|----|---------------|-----------|
| 16 | (32, 32) | 0.391 ms (worse) |
| **32** | **(32, 32)** | **0.343 ms** ✅ |
| 64 (dynamic shmem) | (16, 32) | 0.349 ms |

- **Best: CH=32, blockDim.y=32, gridDim.y=32 → 0.343 ms** (vs untuned v5a 0.359 ms, ~4%).
- Pattern: **more row-lanes + fewer blocks** wins (fewer blocks ⇒ fewer global flush atomics, parallelism still sufficient). CH=32 is the sweet spot — 16 is worse (more blocks, 16-wide coalescing), 64 not better.
- As predicted, grid tuning is a single-digit-% gain (occupancy was already saturated), not a step change.

Profiler at the best point (`hist_kernel`):

| Metric | Value | |
|--------|-------|--|
| **L1TEX throughput** | **91.6%** | ← near-saturated: the **new bottleneck** |
| SM / L2 throughput | 64.7% / 57.7% | |
| **DRAM throughput** | **40.7%** | not bandwidth-bound |
| Achieved occupancy | 90.4% | |
| stall long_scoreboard | 6.95 | residual load latency |
| stall mio_throttle | 2.56 | shared-atomic pipe |

Observation — bottleneck has migrated to the **L1TEX / shared-memory pipe (91.6%)**: the 537 M shared `atomicAdd`s + global loads now saturate that pipe. This is the *first* time a real resource is saturated (v3/v4 were latency-bound with nothing saturated). DRAM at 40.7% ⇒ ~60% bandwidth is unusable because the shared-atomic counting work gates it. The L1TEX cost was always there (inherent to "one atomic per element"); earlier bottlenecks (L2, load latency) masked it until they were cleared.

Conclusion (at the time) — **STOP here.** Higher occupancy / more grid tuning won't help: the limiter is a saturated pipe doing the algorithm's intrinsic work, not lack of warps. Beating it needs *fewer* shared atomics (warp-aggregation — ineffective here since a warp spans 32 distinct channels, or a sort/reduce-based count) — an algorithmic change with low ROI.

> **Superseded by Entry 9.** The "L1TEX is a hard wall" call was half right: L1TEX
> *was* saturated (91.6%), but that pipe carries **both** global loads **and** shared
> atomics. The load half was reducible after all — a *vectorized* load (1 uint16 = 2
> channels) halves the load transactions through L1TEX. Entry 9 does exactly that
> (without the occupancy crash of Entry 7) and beats 0.343 → **0.297 ms**.

---

### Entry 9 — CH=64 + uint16 vectorized load + lane-major conflict-free shared — WIN

Date: 2026-06-28

Thinking (consensus synthesis): the L1TEX-bound state (Entry 8, 91.6%) is load+atomic
traffic combined. Two levers, applied together:
1. **Vectorized load that preserves the warp shape.** Entry 7 proved vectorization
   helps the load (`long_scoreboard` 6.01→4.72) but crashed occupancy (92.7%→35.5%)
   because it used 4 ch/thread → 8 threads in x → 128-thread blocks. Instead keep
   `threadIdx.x = 0..31` and let each lane read **one uint16 = 2 consecutive channels**
   (`CH=64`), so the block stays `(32,32)` = 1024 threads → occupancy holds. Halves the
   load transactions through L1TEX and doubles bytes/request (64 B/warp vs 32 B).
2. **Lane-major conflict-free shared layout.** A naive vectorized version raises shared
   stalls (Entry 7's `short_scoreboard` 0.95→2.65). Store with `slot = j*32 + lane`:
   `s[v*CH + j*32 + lane]` ⇒ bank = `(v*64 + j*32 + lane) % 32 = lane` for every j ⇒
   conflict-free within a warp regardless of the random value v. (This is the *correct*
   swizzle — unlike Entry 5's 257-padding, which left bank = `(lc+v)%32` still random.)
   Needs 64 KiB **dynamic** shared (opt-in via `cudaFuncSetAttribute`); 2 blocks/SM fit.

Code version: `versions/v8_ch64_vec2.cu` (CH=64, VEC=2, UNROLL=8, block=(32,32),
grid=(num_channels/64, 32), 64 KiB dynamic shared, `s[bin*64 + j*32 + lane]`).

Correctness: **pass** (`check: pass`).

Performance:
- **Runtime: 0.297 ms** (mean of 30, best 0.294) vs same-session Entry 8 baseline
  **0.342 ms** → **1.15× over Entry 8**, **~173× over baseline**, **~1.86× off the
  160 µs roofline** (was 2.14×).
- UNROLL sweep on CH=64 (8/12/16): **flat** — 0.297 / 0.296 / 0.298 ms. Load latency is
  already largely hidden (`long_scoreboard` 4.22), so deeper unroll buys nothing. Kept
  UNROLL=8 as canonical.

Profiler (ncu) — Entry 9 vs Entry 8, all success criteria met:

| Metric | Entry 8 | Entry 9 | criterion | |
|--------|--------:|--------:|-----------|--|
| Runtime | 0.342 ms | **0.297 ms** | < 0.342 | ✅ −13% |
| **L1TEX throughput** | **91.6%** | **42.35%** | not worse | ✅ pipe **un-saturated** |
| DRAM throughput | 40.7% | **46.98%** | > 40.7% | ✅ |
| L2 throughput | 57.7% | 63.86% | — | up |
| SM throughput | 64.7% | 68.85% | — | up |
| Achieved occupancy | 90.4% | **79.05%** | ≥ 75–80% | ✅ (held, vs Entry 7's 35.5%) |
| stall long_scoreboard | 6.95 | **4.22** | < 6.95 | ✅ |
| stall mio_throttle (shared-atomic pipe) | 2.56 | **0.55** | not worse | ✅ |
| stall short_scoreboard (bank/shared) | (0.95; naive-vec→2.65) | **0.68** | not blown up | ✅ lane-major worked |

Observation:
- **The "intrinsic L1TEX wall" was beatable.** Vectorizing the load dropped L1TEX
  91.6%→42.35% — i.e. roughly half of that saturation was load-transaction traffic,
  not shared-atomic work. The pipe is no longer the bottleneck.
- **The lane-major layout did its job:** `short_scoreboard` stayed at 0.68 instead of
  blowing up to ~2.65 the way a naive bin-major/uint16 version did in Entry 7. So the
  two levers are genuinely synergistic — vectorize without the conflict-free layout
  would have given back much of the win to shared-memory stalls.
- New state: **nothing saturated again** (L1TEX 42%, DRAM 47%, occ 79%), back to mildly
  latency-bound (`long_scoreboard` 4.22 is the top stall). DRAM at 47% ⇒ there is still
  headroom toward the 160 µs roofline.

Method note / who-was-right: in review, two analyses disagreed. (a) "Bottleneck is
shared-atomic/L1TEX; do a standalone bank-conflict swizzle first" — correctly read the
91.6% L1TEX as the best kernel's (an earlier read had wrongly dismissed it as a
mis-attributed Triton number), but its #1 standalone-swizzle pick targeted
`short_scoreboard` (~0.7) which is *not* the limiter. (b) "It's load-latency limited; the
occupancy-preserving vectorized load is the real lever" — right that Entry 7's vectorize
failed only on occupancy. The win came from **combining both**: occupancy-preserving
vectorized load (lever b) + lane-major conflict-free layout (the *useful* form of lever a,
folded in for free).

Next step (deferred — needs a decision): **CH=128 + uchar4** would halve load
transactions again, but needs 128 KiB shared ⇒ ~1 block/SM ⇒ occupancy risk (the same
trap as Entry 7). Only worth it because CH=64 held occupancy *and* is still load-limited
(`long_scoreboard` top); abort if occupancy collapses or L1TEX/`mio_throttle` spikes.

---

### Entry 10 — CH=128 + uint32 (uchar4) vectorized load — WIN (despite 50% occupancy)

Date: 2026-06-28

Thinking: push Entry 9 one more notch — each lane reads ONE uint32 = **4 consecutive
channels** (`CH=128`, `VEC=4`), 4× fewer load instructions than the byte version, 128 B
/warp/instr. Same lane-major conflict-free layout (`bank = lane` still holds for VEC=4).
Known risk: 128 KiB dynamic shared ⇒ only 1 block/SM ⇒ occupancy capped ~50%.

Code version: `versions/v9_ch128_vec4.cu` (CH=128, VEC=4, UNROLL=8, block=(32,32),
grid=(num_channels/128, 32), 128 KiB dynamic shared, `s[bin*128 + j*32 + lane]`).

Correctness: **pass**.

Performance:
- **Runtime: 0.270 ms** (mean of 4, best 0.270) vs Entry 9 0.297 ms → **1.10× over
  Entry 9**, **~190× over baseline**, **~1.69× off the 160 µs roofline**.

Profiler (ncu) — Entry 10 vs Entry 9:

| Metric | Entry 9 | Entry 10 | |
|--------|--------:|---------:|--|
| Runtime | 0.297 ms | **0.270 ms** | ✅ −9% |
| **Achieved occupancy** | 79.05% | **49.91%** | ⚠️ dropped to ~50% (1 block/SM, 128 KiB shared) — *as predicted* |
| DRAM throughput | 46.98% | **53.94%** | ✅ climbing toward bandwidth |
| L1TEX throughput | 42.35% | 41.82% | flat — not the wall |
| L2 / SM throughput | 63.86 / 68.85% | 65.27 / 66.37% | — |
| stall long_scoreboard | 4.22 | **2.69** | ✅ 4× fewer loads → latency better hidden |
| stall mio_throttle | 0.55 | 0.01 | shared-atomic pipe idle |
| stall short_scoreboard | 0.68 | 1.02 | up slightly (more atomics/thread); lane-major still holding |

Observation:
- **The predicted occupancy trap did NOT bite.** Occupancy halved to 49.9% (1 block/SM),
  yet the kernel got *faster*, because it was never occupancy-starved: UNROLL=8 × 4
  loads-in-flight supplies enough ILP to hide the load latency on its own
  (`long_scoreboard` actually *fell* 4.22→2.69, since there are 4× fewer load
  instructions to wait on). Occupancy is a means (latency hiding); ILP substituted for it.
- **DRAM throughput climbed 47%→54%** — the kernel is now the closest it has been to
  bandwidth-bound. This is the end of the vectorize-the-load line: `CH=256` (uint64,
  8 ch/lane) would need 256 KiB shared > 228 KiB/SM, so it won't fit.
- Final state: DRAM 54%, occupancy capped at 50% by the 128 KiB histogram, top stall
  `long_scoreboard` 2.69 (small). The residual gap to 160 µs is the occupancy cap (can't
  hide latency further) plus the intrinsic shared-atomic work — closing it needs an
  algorithmic change (fewer atomics per element), not more launch/layout tuning.

Conclusion: **0.270 ms is the practical floor of the one-atomic-per-element form.** Three
levers compounded from Entry 8's 0.343 ms: uint16 vec load (0.297), then uint32 vec load
(0.270), each un-saturating a pipe and lowering load-stall, with the lane-major layout
keeping bank conflicts negligible throughout. ~190× over baseline, 1.69× off roofline.

---

## Summary

| Version | Runtime | vs baseline | Bottleneck addressed |
|---------|---------|-------------|----------------------|
| Entry 0 baseline (PyTorch) | 51.31 ms | 1× | — |
| Entry 1 transpose (PyTorch) | 45.36 ms | 1.13× | (failed: over-fetch relocated) |
| Entry 2 best-effort Triton (C: `tl.histogram`) | 4.107 ms | 12.5× | Triton's ceiling — L2-bound, no shared-mem privatization |
| Entry 3 fused CUDA, global atomics | 6.77 ms | 7.6× | strided over-fetch → read once |
| Entry 4 shared-mem privatized | 0.816 ms | 63× | global-atomic L2 traffic |
| Entry 5 bank-conflict padding | 0.813 ms | 63× | (failed: no effect) |
| Entry 6 row-unroll ×8 | 0.359 ms | 143× | global-load latency (MLP) |
| Entry 7 vectorized int loads | 0.400 ms | 128× | (failed: occupancy crash) |
| Entry 8 grid/block tuning | 0.343 ms | ~150× | flush-atomic count (launch config) |
| Entry 9 CH=64 uint16 vec load + lane-major shared | 0.297 ms | ~173× | un-saturated L1TEX (halved load transactions, conflict-free) |
| **Entry 10 CH=128 uint32 vec load + lane-major shared** | **0.270 ms** | **~190×** | DRAM 54% (4× fewer loads); won at 50% occupancy via ILP |

(Entry 2 is a side branch — best-effort Triton, not on the CUDA optimization
line. Its three variants: A global-atomics 11.0 ms, B one-hot privatize 629 ms,
C `tl.histogram` 4.107 ms. Best Triton ≈ 4.1 ms, ~15× behind the CUDA best.)

**Final: 0.270 ms, ~190× over baseline, ~1.69× off the 160 µs read roofline** (code: `versions/v9_ch128_vec4.cu`).

Where it stands: Entry 8's "L1TEX wall (91.6%) is intrinsic, STOP" call was beatable.
That pipe carried both global loads and shared atomics; warp-shape-preserving vectorized
loads halved then quartered the load transactions — **uint16 (Entry 9, 0.297 ms)** then
**uint32 (Entry 10, 0.270 ms)** — un-saturating L1TEX (91.6%→42%) and pushing DRAM
40.7%→54%, while a **lane-major conflict-free** shared layout kept bank conflicts
negligible throughout (and ILP from UNROLL=8 covered the drop to 50% occupancy at CH=128).
This is the practical floor of the one-atomic-per-element form (CH=256 won't fit shared).
The residual ~1.7× to the 160 µs roofline is the occupancy cap + intrinsic atomic work;
closing it needs an algorithmic change (**fewer shared atomics** per element), not more
launch/layout tuning.

Key lessons recorded along the way: (1) strided column access over-fetches ~64×; read once, coalesced. (2) a physical transpose just relocates the over-fetch. (3) when nothing is saturated, read **warp stall reasons** before naming a bottleneck — guessing cost two wrong calls (bank conflicts, "shared-atomic throughput"). (4) padding only fixes bank conflicts for correlated writes, not random data. (5) occupancy is a means (latency hiding) with diminishing returns — 90%+ occupancy didn't prevent being latency-bound; ILP (unroll) fixed it. (6) optimization migrates the bottleneck until you hit an intrinsic resource limit. (7) Triton (Entry 2) has no cheap privatized scatter — without user-managed shared memory it tops out ~12× behind hand CUDA on histogram: global atomics hit the L2 wall, one-hot privatization is a 256× compute disaster, and the built-in `tl.histogram` is fast but forces strided per-channel reads. Pick the tool to the access pattern: histogram's scatter wants explicit SRAM, which is CUDA's turf.
