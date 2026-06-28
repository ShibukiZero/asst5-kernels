# FlashAttention — version history (code for each worklog Entry)

`submission.*` is gitignored; these preserve the code for each step in `../worklog.md`.
Large case (4,64,8192,128) on H100 is the benchmark; baseline naive attention = 115 ms.

| File | Entry | Runtime (large) | Notes |
|------|-------|-----------------|-------|
| (baseline) `reference.ref_kernel` | 0 | 115.04 ms | naive 3-op attention; materializes 34 GB S×S (memory wall) |
| `v1_sdpa.py` | 1 | 12.76 ms | ✅ `F.scaled_dot_product_attention` (cuDNN backend, ~70% MFU). Bar for hand-written kernels |
| `v2_triton.py` | 2 | 18.86 ms | ✅ hand-written Triton FA-2 (tiling + online softmax), best tile 128×128×8w×3s. 6.1× over baseline, beats FA-2 backend, 1.48× off cuDNN. ncu: DRAM 3% (memory wall gone), 47% MFU, register-limited |
| **`v3_cutlass_fa3.py`** | 4 | **12.79 ms** | ✅ **BEST** — CUTLASS FA-3 (Hopper FMHA, example 88: warp-spec+TMA+wgmma). Matches cuDNN (12.76). ncu: 76% SM (vs Triton 52%) — warp-spec payoff. 9.0× over baseline |
| `v4_fa3_cached.py` | 6 | 12.78 ms | ⚖️ cached/slim CUTLASS FA-3 (no per-call alloc/init) + device fix. WASH — wrapper overhead isn't in the timed region; SDPA=CUTLASS=cached all ~12.78 |
| `v5_fa3_fp8.py` | 7 | (won't build) | ⚖️ FP8 e4m3 FA-3. Numerically VIABLE (sim: maxdiff 0.009<0.01, 0 viol, needs per-row P scaling) but FmhaBuilder FP8 path doesn't build in this CUTLASS (MixedInput mainloop / StageCount mismatch) |
