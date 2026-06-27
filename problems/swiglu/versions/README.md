# SwiGLU — version history (code for each worklog Entry)

`submission.*` is gitignored; these preserve the code for each step in `../worklog.md`.

| File | Entry | Runtime | Notes |
|------|-------|---------|-------|
| (baseline) `reference.ref_kernel` | 0 | 12.16 ms | FP32 GEMMs on CUDA cores (no Tensor Cores) |
| `v1_tf32.py` | 1 | 2.94 ms | TF32 Tensor Cores (4.1×); BF16 fails tolerance |
| **`v2_torchcompile.py`** | 2 | **2.036 ms** | ✅ BEST — TF32 + F.silu + torch.compile (Inductor fuses epilogue) |
| `v3_triton_fused.py` | 3 | 11.1 ms | ❌ hand Triton fused GEMM: 5.5× slower + shmem-OOM on tuning + fails strict tol |
| `v4_cutlass.py` | 4 | 2.501 ms | ❌ CUTLASS 3.x sm90: GEMM **beats cuBLAS** (0.817 vs 0.93 ms) but full pipeline slower + fails 1e-2 (2.6% violations). Test is self-referential to cuBLAS-TF32 |
