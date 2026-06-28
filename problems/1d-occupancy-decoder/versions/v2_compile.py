# Entry 2 - torch.compile over the functional forward.
# Same graph as Entry 1, but the elementwise glue (SiLU, LayerNorm fp32 up/down
# casts, biases, the head-reshape contiguous) is handed to TorchInductor to fuse.
# GEMMs stay on cuBLAS and SDPA stays on cudnn flash (inductor lowers them to the
# same library kernels). First call compiles (absorbed by the harness warmup /
# correctness call); shapes are static (250k / 1024) so no recompiles.
import torch
import torch.nn.functional as F
from task import input_t, output_t

NUM_HEADS = 12          # problem constant (width 768 / head_dim 64)
LN_EPS = 1.0e-6         # OneDOccupancyDecoder default


def _forward(queries, latents,
             qi_w, qi_b, qo_w, qo_b,
             cq_w, cq_b, ck_w, ck_b, cv_w, cv_b, cp_w, cp_b,
             op_w, op_b):
    q = F.linear(queries, qi_w, qi_b)
    q = F.silu(q)
    q = F.linear(q, qo_w, qo_b)

    qh = F.linear(q, cq_w, cq_b)
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


_compiled = torch.compile(_forward, fullgraph=True)


def custom_kernel(data: input_t) -> output_t:
    queries, latents, w = data
    return _compiled(
        queries, latents,
        w['query_in_in_layer_weight'], w['query_in_in_layer_bias'],
        w['query_in_out_layer_weight'], w['query_in_out_layer_bias'],
        w['attn_c_q_weight'], w['attn_c_q_bias'],
        w['attn_c_k_weight'], w['attn_c_k_bias'],
        w['attn_c_v_weight'], w['attn_c_v_bias'],
        w['attn_c_proj_weight'], w['attn_c_proj_bias'],
        w['out_proj_weight'], w['out_proj_bias'],
    )
