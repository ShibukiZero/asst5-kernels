# Entry 1 — F.scaled_dot_product_attention (PyTorch's library FlashAttention).
# Drop-in replacement for the naive 3-op reference. SDPA defaults match the
# reference exactly: scale = 1/sqrt(head_dim), is_causal=False (non-causal).
# On H100 + fp16 + head_dim=128 it dispatches to the fused flash/cuDNN backend,
# which never materializes the S×S matrix -> kills the O(N^2) memory wall.
import torch
import torch.nn.functional as F
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    q, k, v = data
    return F.scaled_dot_product_attention(q, k, v)
