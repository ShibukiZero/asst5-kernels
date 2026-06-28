# Entry 7 - Entry 6 (q-fold) + fused LayerNorm->out_proj tail (Triton).
# The output is [B, 250k, 1], so there is no need to materialize the normalized
# [B, 250k, 768] tensor and then GEMM it to a scalar. One Triton kernel per row:
#   mean/var (fp32) -> normalize -> cast to fp16 (matches LayerNorm.type_as) ->
#   dot with out_proj weight (fp32 accumulate) + bias -> one fp16 scalar.
# Replaces the Entry-6 tail (Inductor LayerNorm ~250us + out_proj GEMM ~130us, both
# streaming the 250k x 768 tensor) with a single ~bandwidth-bound read of y.
# Builds on Entry 6 (q-fold, correctness already validated). Goal: < 3.59 ms.
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from task import input_t, output_t

NUM_HEADS = 12
LN_EPS = 1.0e-6
WIDTH = 768
_NUM_WARPS = 2   # swept {1,2,4,8}: 1/2 tie ~3.41ms, 4=3.49, 8=3.65 (768-elem rows want few warps)


@triton.jit
def _ln_out_kernel(Y, W, BIAS, OUT, D: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(Y + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    rstd = tl.rsqrt(var + EPS)
    norm = (x - mean) * rstd
    norm_h = norm.to(tl.float16)                       # LayerNorm casts back to fp16
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    acc = tl.sum(norm_h.to(tl.float32) * w, axis=0)    # out_proj dot (fp32 accumulate)
    bias = tl.load(BIAS).to(tl.float32)
    tl.store(OUT + row, (acc + bias).to(tl.float16))


def _run_ln_out(y, op_w, op_b):
    # y: [B, S, 768] contiguous -> per-row fused LN + out_proj -> [B, S, 1]
    B, S, D = y.shape
    out = torch.empty((B * S,), device=y.device, dtype=y.dtype)
    yf = y.reshape(B * S, D)
    _ln_out_kernel[(B * S,)](yf, op_w.reshape(-1), op_b.reshape(-1), out,
                             D=D, EPS=LN_EPS, BLOCK=1024, num_warps=_NUM_WARPS)
    return out.view(B, S, 1)


def _prefix_qfold(queries, latents,
                  qi_w, qi_b, qfold_w, qfold_b,
                  ck_w, ck_b, cv_w, cv_b, cp_w, cp_b):
    h = F.linear(queries, qi_w, qi_b)
    h = F.silu(h)
    qh = F.linear(h, qfold_w, qfold_b)

    kh = F.linear(latents, ck_w, ck_b)
    vh = F.linear(latents, cv_w, cv_b)

    b, l, d = qh.shape
    s = kh.shape[1]
    qh = qh.view(b, l, NUM_HEADS, -1).transpose(1, 2)
    kh = kh.view(b, s, NUM_HEADS, -1).transpose(1, 2)
    vh = vh.view(b, s, NUM_HEADS, -1).transpose(1, 2)

    y = F.scaled_dot_product_attention(qh, kh, vh)

    y = y.transpose(1, 2).contiguous().view(b, l, d)
    y = F.linear(y, cp_w, cp_b)                        # c_proj -> y [b, l, 768]
    return y


_compiled_prefix = torch.compile(_prefix_qfold, fullgraph=True)
_qfold = {}


def _get_qfold(qo_w, qo_b, cq_w, cq_b):
    key = tuple((t.data_ptr(), t._version) for t in (qo_w, qo_b, cq_w, cq_b))
    c = _qfold.get(key)
    if c is None:
        _qfold.clear()
        with torch.no_grad():
            qfold_w = torch.mm(cq_w.float(), qo_w.float()).to(qo_w.dtype).contiguous()
            qfold_b = F.linear(qo_b.float(), cq_w.float(), cq_b.float()).to(qo_b.dtype).contiguous()
        c = (qo_w, qo_b, cq_w, cq_b, qfold_w, qfold_b)
        _qfold[key] = c
    return c[4], c[5]


def custom_kernel(data: input_t) -> output_t:
    queries, latents, w = data
    qfold_w, qfold_b = _get_qfold(
        w['query_in_out_layer_weight'], w['query_in_out_layer_bias'],
        w['attn_c_q_weight'], w['attn_c_q_bias'],
    )
    y = _compiled_prefix(
        queries, latents,
        w['query_in_in_layer_weight'], w['query_in_in_layer_bias'],
        qfold_w, qfold_b,
        w['attn_c_k_weight'], w['attn_c_k_bias'],
        w['attn_c_v_weight'], w['attn_c_v_bias'],
        w['attn_c_proj_weight'], w['attn_c_proj_bias'],
    )
    return _run_ln_out(y.contiguous(), w['out_proj_weight'], w['out_proj_bias'])
