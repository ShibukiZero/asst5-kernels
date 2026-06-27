# SwiGLU — version history (code for each worklog Entry)

`submission.*` is gitignored; these preserve the code for each step in `../worklog.md`.

| File | Entry | Runtime | Notes |
|------|-------|---------|-------|
| (baseline) `reference.ref_kernel` | 0 | 12.16 ms | FP32 GEMMs on CUDA cores (no Tensor Cores) |
| `v1_tf32.py` | 1 | 2.94 ms | TF32 Tensor Cores (4.1×); BF16 fails tolerance |
| **`v2_torchcompile.py`** | 2 | **2.036 ms** | ✅ TF32 + F.silu + torch.compile (Inductor fuses epilogue) |
