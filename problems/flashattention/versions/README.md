# FlashAttention — version history (code for each worklog Entry)

`submission.*` is gitignored; these preserve the code for each step in `../worklog.md`.
Large case (4,64,8192,128) on H100 is the benchmark; baseline naive attention = 115 ms.

| File | Entry | Runtime (large) | Notes |
|------|-------|-----------------|-------|
| (baseline) `reference.ref_kernel` | 0 | 115.04 ms | naive 3-op attention; materializes 34 GB S×S (memory wall) |
| `v1_sdpa.py` | 1 | 12.76 ms | ✅ `F.scaled_dot_product_attention` (cuDNN backend, ~70% MFU). Bar for hand-written kernels |
| `v2_triton.py` | 2 | 18.86 ms | ✅ hand-written Triton FA-2 (tiling + online softmax), best tile 128×128×8w×3s. 6.1× over baseline, beats FA-2 backend, 1.48× off cuDNN. ncu: DRAM 3% (memory wall gone), 47% MFU, register-limited |
| **`v3_cutlass_fa3.py`** | 4 | **12.79 ms** | ✅ **BEST** — CUTLASS FA-3 (Hopper FMHA, example 88: warp-spec+TMA+wgmma). Matches cuDNN (12.76). ncu: 76% SM (vs Triton 52%) — warp-spec payoff. 9.0× over baseline |
