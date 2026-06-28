# 1D Occupancy Decoder — version history (code for each worklog Entry)

`submission.py` is what the harness runs; these files preserve the code for each
step in `../worklog.md`. Benchmark: B=1, 250k queries, 1024 latents, width 768,
12 heads, q_in_dim 3, seed 5531. Tolerance rtol=atol=1e-2. Nebius H100, torch 2.12.1+cu130.

| File | Entry | Runtime | Notes |
|------|-------|---------|-------|
| `v0_baseline.py` | 0 | 18.789 ms | stock PyTorch; rebuilds nn.Module each call. ~6.2 ms GPU work + ~12 ms construction/launch bubbles. SDPA (cudnn flash) = 35% of GPU work |
| `v1_functional.py` | 1 | ~5.35 ms | functional forward (F.linear/SDPA/F.layer_norm), no Module rebuild. **3.5×**. Removed all per-call construction + 14 .copy_() bubbles; GPU math unchanged. SDPA now 40% |
| `v2_compile.py` | 2 | 4.070 ms | torch.compile (inductor, fullgraph). Fused head-reshape (-620us) + LayerNorm/casts (1143→251us) + in_layer. **4.6×**. SDPA (2.17ms, cudnn) now 43% |
| `v3_compile_modes.py` | 3 | 4.667 ms (max-autotune) | mode sweep: max-autotune LOSS (+15%, replaces cuBLAS GEMM w/ slower triton); reduce-overhead/cudagraphs WASH (4.079). Keep default (Entry 2) |
