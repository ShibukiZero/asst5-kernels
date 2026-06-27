# Triton best-effort — Variant C: the built-in tl.histogram.
# tl.histogram works on a flattened 1D block, so it cannot keep channels
# separate -> each program must own ONE channel and read that channel's column.
# In the [length, C] row-major layout a column is strided by C=512, so this
# reintroduces the ~64x strided over-fetch that Entry 0/1 identified as the
# root cause. We accumulate per-chunk tl.histogram results, then store once.
import torch
import triton
import triton.language as tl
from task import input_t, output_t


@triton.jit
def _hist_builtin(
    data_ptr,                 # *u8  [length, C]
    hist_ptr,                 # *i32 [C, num_bins]
    length, C, num_bins,
    BLOCK_R: tl.constexpr,
    NUM_BINS: tl.constexpr,
):
    c = tl.program_id(0)                                   # one channel per program
    acc = tl.zeros((NUM_BINS,), dtype=tl.int32)
    for r0 in range(0, length, BLOCK_R):
        rows = r0 + tl.arange(0, BLOCK_R)
        rmask = rows < length
        ptrs = data_ptr + rows * C + c                     # STRIDED by C -> over-fetch
        vals = tl.load(ptrs, mask=rmask, other=0).to(tl.int32)
        acc += tl.histogram(vals, NUM_BINS, mask=rmask)
    bins = tl.arange(0, NUM_BINS)
    tl.store(hist_ptr + c * num_bins + bins, acc, mask=bins < num_bins)


def custom_kernel(data: input_t) -> output_t:
    array, num_bins = data
    length, C = array.shape
    hist = torch.zeros(C, num_bins, dtype=torch.int32, device=array.device)
    BLOCK_R = 2048
    grid = (C,)
    _hist_builtin[grid](array, hist, length, C, num_bins,
                        BLOCK_R=BLOCK_R, NUM_BINS=num_bins)
    return hist
