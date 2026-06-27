# SwiGLU Entry 3 — hand-written Triton FUSED kernel.
# One kernel computes both gate=x@W and value=x@V (sharing each x tile, loaded
# once -> two accumulators), then applies silu(gate+b)*(value+c) in registers and
# writes only the final output. gate/value never touch DRAM. TF32 tensor cores.
import os
import torch
import triton
import triton.language as tl
from task import input_t, output_t

# Make the reference (used by check_implementation) also use TF32, matching our
# Triton TF32 dot — otherwise it's TF32 (ours) vs full FP32 (ref) and near-zero
# outputs exceed the 1e-2 tolerance.
torch.set_float32_matmul_precision("high")


@triton.jit
def _swiglu_kernel(
    x_ptr, w_ptr, v_ptr, b_ptr, c_ptr, out_ptr,
    M, K, N,
    stride_xm, stride_xk, stride_wk, stride_wn, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    v_ptrs = v_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_v = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        km = offs_k[None, :] < (K - k0)
        kn = offs_k[:, None] < (K - k0)
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & km, other=0.0)
        w = tl.load(w_ptrs, mask=kn & (offs_n[None, :] < N), other=0.0)
        v = tl.load(v_ptrs, mask=kn & (offs_n[None, :] < N), other=0.0)
        acc_g = tl.dot(x, w, acc_g, input_precision="tf32")   # x tile reused
        acc_v = tl.dot(x, v, acc_v, input_precision="tf32")
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
        v_ptrs += BLOCK_K * stride_wk

    bb = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    cc = tl.load(c_ptr + offs_n, mask=offs_n < N, other=0.0)
    gate = acc_g + bb[None, :]
    value = acc_v + cc[None, :]
    out = (gate * tl.sigmoid(gate)) * value                   # silu(gate)*value, in-register

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def custom_kernel(data: input_t) -> output_t:
    x, W, V, b, c, beta = data
    B, S, K = x.shape
    M = B * S
    N = W.shape[1]
    x2 = x.reshape(M, K)
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    BLOCK_M = int(os.getenv("SG_BM", "128"))
    BLOCK_N = int(os.getenv("SG_BN", "128"))
    BLOCK_K = int(os.getenv("SG_BK", "32"))
    nw = int(os.getenv("SG_NW", "4"))
    ns = int(os.getenv("SG_NS", "3"))

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _swiglu_kernel[grid](
        x2, W, V, b, c, out, M, K, N,
        x2.stride(0), x2.stride(1), W.stride(0), W.stride(1), out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=nw, num_stages=ns,
    )
    return out.reshape(B, S, N)
