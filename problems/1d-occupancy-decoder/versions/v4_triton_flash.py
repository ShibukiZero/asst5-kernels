# Entry 4 - hand Triton flash attention for the SDPA (NOT ADOPTED).
# Best tile config from the sweep: BM=128, BN=64, num_warps=8, num_stages=4,
# exp2 softmax, no N-mask (LKV % BN == 0). Isolated attention = 2.017 ms = 0.90x
# cudnn (1.81 ms) on this shape (q 250k x kv 1024, 12 heads, hd 64). LOSES to cudnn:
# ncu shows 24.7% occupancy / 0.58 IPC -> can't feed tensor cores like cudnn's
# wgmma+warp-spec+TMA pipeline. warp_specialize=True crashes (needs TMA/block-ptr
# loads). Kept for the record; submission stays on Entry 2 (torch.compile, cudnn SDPA).
import torch, triton, triton.language as tl
B, H, LQ, LKV, D = 1, 12, 250000, 1024, 64
Z = B * H
scale = 1.0 / (D ** 0.5); LOG2E = 1.4426950408889634
q = torch.randn(Z, LQ, D, device='cuda', dtype=torch.float16)
k = torch.randn(Z, LKV, D, device='cuda', dtype=torch.float16)
v = torch.randn(Z, LKV, D, device='cuda', dtype=torch.float16)

@triton.jit
def _flash(Q, K, V, Out, sqz, sqm, sqd, skz, skn, skd, svz, svn, svd, soz, som, sod,
           LQ, LKV, qkscale, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr):
    start_m = tl.program_id(0); off_z = tl.program_id(1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N); offs_d = tl.arange(0, D)
    m_mask = offs_m < LQ
    qt = tl.load(Q + off_z*sqz + offs_m[:, None]*sqm + offs_d[None, :]*sqd, mask=m_mask[:, None], other=0.0)
    m_i = tl.full([BLOCK_M], -float('inf'), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32); acc = tl.zeros([BLOCK_M, D], tl.float32)
    for start_n in range(0, LKV, BLOCK_N):
        offs_nc = start_n + offs_n
        kt = tl.load(K + off_z*skz + offs_d[:, None]*skd + offs_nc[None, :]*skn)
        qk = tl.dot(qt, kt) * qkscale
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - m_new[:, None]); alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1); acc = acc * alpha[:, None]
        vt = tl.load(V + off_z*svz + offs_nc[:, None]*svn + offs_d[None, :]*svd)
        acc += tl.dot(p.to(tl.float16), vt); m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(Out + off_z*soz + offs_m[:, None]*som + offs_d[None, :]*sod, acc.to(tl.float16), mask=m_mask[:, None])

out = torch.empty_like(q)
BM, BN = 128, 64
grid = (triton.cdiv(LQ, BM), Z)
_flash[grid](q, k, v, out, q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
             v.stride(0), v.stride(1), v.stride(2), out.stride(0), out.stride(1), out.stride(2),
             LQ, LKV, scale*LOG2E, BLOCK_M=BM, BLOCK_N=BN, D=D, num_warps=8, num_stages=4)
torch.cuda.synchronize()
