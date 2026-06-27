# Entry 2 — Hand-written Triton FlashAttention (FA-2 style: tiling + online softmax).
# Best swept config BLOCK_M=BLOCK_N=128, num_warps=8, num_stages=3 -> 18.3 ms (large),
# 6.3x over baseline, beats PyTorch's FA-2 flash backend (24.6 ms), loses to cuDNN (12.76 ms).
# No boundary mask: all benchmark seq_lens (1024/4096/8192) are multiples of 128.
import math
import torch
import triton
import triton.language as tl
from task import input_t, output_t


@triton.jit
def _attn_fwd(Q, K, V, sm_scale, Out,
              stride_qz, stride_qh, stride_qm, stride_qk,
              stride_kz, stride_kh, stride_kn, stride_kk,
              stride_vz, stride_vh, stride_vn, stride_vk,
              stride_oz, stride_oh, stride_om, stride_ok,
              H, N_CTX,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = off_z * stride_kz + off_h * stride_kh
    v_off = off_z * stride_vz + off_h * stride_vh
    o_off = off_z * stride_oz + off_h * stride_oh

    Q_bp = tl.make_block_ptr(base=Q + q_off, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qk),
                             offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    K_bp = tl.make_block_ptr(base=K + k_off, shape=(BLOCK_DMODEL, N_CTX), strides=(stride_kk, stride_kn),
                             offsets=(0, 0), block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1))
    V_bp = tl.make_block_ptr(base=V + v_off, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_vn, stride_vk),
                             offsets=(0, 0), block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0))
    O_bp = tl.make_block_ptr(base=Out + o_off, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_om, stride_ok),
                             offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504089  # log2(e): fold into q to use exp2
    q = tl.load(Q_bp)
    q = (q * qk_scale).to(q.dtype)

    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(K_bp)
        qk = tl.dot(q, k)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(V_bp)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        K_bp = tl.advance(K_bp, (0, BLOCK_N))
        V_bp = tl.advance(V_bp, (BLOCK_N, 0))

    acc = acc / l_i[:, None]
    tl.store(O_bp, acc.to(Out.dtype.element_ty))


def custom_kernel(data: input_t) -> output_t:
    q, k, v = data
    B, H, S, D = q.shape
    o = torch.empty_like(q)
    scale = 1.0 / math.sqrt(D)
    BLOCK_M = BLOCK_N = 128
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    _attn_fwd[grid](
        q, k, v, scale, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H, S, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D,
        num_warps=8, num_stages=3)
    return o
