# Entry 6 - q-fold: collapse query_in.out_layer + attn.c_q into ONE linear.
# out_layer and c_q are consecutive affine maps with NO nonlinearity between them
# (SiLU sits before out_layer), so:
#   qh = (h @ qo_w^T + qo_b) @ cq_w^T + cq_b
#      = h @ (cq_w @ qo_w)^T + (qo_b @ cq_w^T + cq_b)
# => qfold_w = cq_w @ qo_w ,  qfold_b = F.linear(qo_b, cq_w, cq_b).
# This deletes one 250k x 768 x 768 cuBLAS GEMM (~0.49 ms): the three big 768->768
# GEMMs (out_layer, c_q, c_proj) become two (qfold, c_proj). The fold depends only on
# weights -> computed once and cached (NOT recomputed per call, NOT input-dependent).
# Folded weight is computed in fp32 then cast back to fp16 to minimise the folding
# rounding error (the only correctness risk vs the reference's unfolded fp16 path).
# Builds on Entry 2 (torch.compile of the rest). Goal: beat Entry 2's 4.070 ms (~3.6 ms).
import torch
import torch.nn.functional as F
from task import input_t, output_t

NUM_HEADS = 12          # problem constant (width 768 / head_dim 64)
LN_EPS = 1.0e-6         # OneDOccupancyDecoder default


def _forward_qfold(queries, latents,
                   qi_w, qi_b, qfold_w, qfold_b,
                   ck_w, ck_b, cv_w, cv_b, cp_w, cp_b,
                   op_w, op_b):
    h = F.linear(queries, qi_w, qi_b)
    h = F.silu(h)
    qh = F.linear(h, qfold_w, qfold_b)          # replaces out_layer GEMM + c_q GEMM

    kh = F.linear(latents, ck_w, ck_b)
    vh = F.linear(latents, cv_w, cv_b)

    b, l, d = qh.shape
    s = kh.shape[1]
    qh = qh.view(b, l, NUM_HEADS, -1).transpose(1, 2)
    kh = kh.view(b, s, NUM_HEADS, -1).transpose(1, 2)
    vh = vh.view(b, s, NUM_HEADS, -1).transpose(1, 2)

    y = F.scaled_dot_product_attention(qh, kh, vh)

    y = y.transpose(1, 2).contiguous().view(b, l, d)
    y = F.linear(y, cp_w, cp_b)

    y = F.layer_norm(y.float(), (d,), None, None, LN_EPS).type_as(queries)
    return F.linear(y, op_w, op_b)


_compiled = torch.compile(_forward_qfold, fullgraph=True)
_qfold = {}   # weight-identity key -> (refs..., qfold_w, qfold_b)


def _get_qfold(qo_w, qo_b, cq_w, cq_b):
    key = tuple((t.data_ptr(), t._version) for t in (qo_w, qo_b, cq_w, cq_b))
    c = _qfold.get(key)
    if c is None:
        _qfold.clear()
        with torch.no_grad():
            qfold_w = torch.mm(cq_w.float(), qo_w.float()).to(qo_w.dtype).contiguous()
            qfold_b = F.linear(qo_b.float(), cq_w.float(), cq_b.float()).to(qo_b.dtype).contiguous()
        c = (qo_w, qo_b, cq_w, cq_b, qfold_w, qfold_b)   # hold refs -> no ptr reuse
        _qfold[key] = c
    return c[4], c[5]


def custom_kernel(data: input_t) -> output_t:
    queries, latents, w = data
    qfold_w, qfold_b = _get_qfold(
        w['query_in_out_layer_weight'], w['query_in_out_layer_bias'],
        w['attn_c_q_weight'], w['attn_c_q_bias'],
    )
    return _compiled(
        queries, latents,
        w['query_in_in_layer_weight'], w['query_in_in_layer_bias'],
        qfold_w, qfold_b,
        w['attn_c_k_weight'], w['attn_c_k_bias'],
        w['attn_c_v_weight'], w['attn_c_v_bias'],
        w['attn_c_proj_weight'], w['attn_c_proj_bias'],
        w['out_proj_weight'], w['out_proj_bias'],
    )
