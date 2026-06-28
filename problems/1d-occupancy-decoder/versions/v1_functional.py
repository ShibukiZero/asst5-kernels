# Entry 1 - Functional forward. No nn.Module rebuild per call: compute the whole
# graph directly from the weights dict with F.linear / F.scaled_dot_product_attention.
# Kills the per-call ctor random-init (baseline kernels 0,1) and the CPU construction
# bubbles that stretched the baseline to 18.8 ms over its ~6.2 ms of real GPU work.
# Same math as reference: SiLU MLP -> 12-head cross-attention (fp32 softmax) ->
# fp32 LayerNorm (eps 1e-6, no affine) -> out_proj.
import torch
import torch.nn.functional as F
from task import input_t, output_t

NUM_HEADS = 12          # problem constant (width 768 / head_dim 64)
LN_EPS = 1.0e-6         # OneDOccupancyDecoder default


def custom_kernel(data: input_t) -> output_t:
    queries, latents, w = data

    # --- query embedding: Linear(3->768) -> SiLU -> Linear(768->768) ---
    q = F.linear(queries, w['query_in_in_layer_weight'], w['query_in_in_layer_bias'])
    q = F.silu(q)
    q = F.linear(q, w['query_in_out_layer_weight'], w['query_in_out_layer_bias'])

    # --- cross-attention: q attends to latents (k, v) ---
    qh = F.linear(q, w['attn_c_q_weight'], w['attn_c_q_bias'])
    kh = F.linear(latents, w['attn_c_k_weight'], w['attn_c_k_bias'])
    vh = F.linear(latents, w['attn_c_v_weight'], w['attn_c_v_bias'])

    b, l, d = qh.shape
    s = kh.shape[1]
    qh = qh.view(b, l, NUM_HEADS, -1).transpose(1, 2)   # (B, nh, L, hs)
    kh = kh.view(b, s, NUM_HEADS, -1).transpose(1, 2)   # (B, nh, S, hs)
    vh = vh.view(b, s, NUM_HEADS, -1).transpose(1, 2)

    y = F.scaled_dot_product_attention(qh, kh, vh)      # fp32 softmax internally

    y = y.transpose(1, 2).contiguous().view(b, l, d)
    y = F.linear(y, w['attn_c_proj_weight'], w['attn_c_proj_bias'])

    # --- LayerNorm (fp32 internal, no affine) -> out_proj(768->1) ---
    y = F.layer_norm(y.float(), (d,), None, None, LN_EPS).type_as(queries)
    out = F.linear(y, w['out_proj_weight'], w['out_proj_bias'])
    return out
