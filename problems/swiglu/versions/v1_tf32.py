# Entry 1 — TF32 Tensor Cores. 2.94 ms (4.1x over baseline). Correct within 1e-2.
# (BF16 was tried too — faster but fails the 1e-2 tolerance, so TF32 it is.)
import torch
from task import input_t, output_t

torch.set_float32_matmul_precision("high")  # use TF32 tensor cores for fp32 matmul


def custom_kernel(data: input_t) -> output_t:
    x, W, V, b, c, beta = data
    gate = x @ W + b
    value = x @ V + c
    swish = gate * torch.sigmoid(beta * gate)
    return swish * value
