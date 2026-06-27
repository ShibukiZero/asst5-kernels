# SwiGLU — version history (code for each worklog Entry)

`submission.*` is gitignored; these preserve the code for each step in `../worklog.md`.

| File | Entry | Runtime | Notes |
|------|-------|---------|-------|
| (baseline) `reference.ref_kernel` | 0 | 12.16 ms | FP32 GEMMs on CUDA cores (no Tensor Cores) |
| `v1_tf32.py` | 1 | 2.94 ms | TF32 Tensor Cores (4.1×); BF16 fails tolerance |
| `v2_torchcompile.py` | 2 | 2.036 ms | TF32 + F.silu + torch.compile (Inductor fuses epilogue); pure-cuBLAS optimum |
| `v3_triton_fused.py` | 3 | 11.1 ms | ❌ hand Triton fused GEMM: 5.5× slower + shmem-OOM on tuning + fails strict tol |
| `v4_cutlass.py` | 4 | 2.501 ms | ⚠️ CUTLASS 3.x sm90 GEMM **beats cuBLAS** (0.817 vs 0.93 ms); full double-CUTLASS pipeline slower + "fails" (but that 2.6% was the flag-OFF artifact — see Entry 5) |
| **`v5_hybrid_h2.py`** | 5 | **1.954 ms** | ✅ **BEST** — CUTLASS gate + cuBLAS value + compiled epilogue + runtime guard. First hand kernel to beat baseline (~4% over Entry 2) |
| `v6_evt_fused.py` | 6 | 1.972 ms | ⚖️ CUTLASS EVT fuses silu(gate+b)*(value+c) into the gate GEMM (correct), but a wash: fused gate gains the value-read (+0.25ms) ≈ the epilogue it removes |
