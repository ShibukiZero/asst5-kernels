# FlashAttention — version history (code for each worklog Entry)

`submission.*` is gitignored; these preserve the code for each step in `../worklog.md`.
Large case (4,64,8192,128) on H100 is the benchmark; baseline naive attention = 115 ms.

| File | Entry | Runtime (large) | Notes |
|------|-------|-----------------|-------|
| (baseline) `reference.ref_kernel` | 0 | 115.04 ms | naive 3-op attention; materializes 34 GB S×S (memory wall) |
| **`v1_sdpa.py`** | 1 | **12.76 ms** | ✅ `F.scaled_dot_product_attention` (cuDNN backend, ~70% MFU). Bar for hand-written kernels |
