# Triton best-effort — Variant A: fused single-pass, coalesced read, GLOBAL atomics.
# Mirrors CUDA Entry 2 (v2). Each program owns a block of consecutive channels
# (coalesced load), grid-strides over rows, scatters with one tl.atomic_add per
# element into the global [C, num_bins] histogram.
import torch
import triton
import triton.language as tl
from task import input_t, output_t


@triton.jit
def _hist_atomics(
    data_ptr,                 # *u8  [length, C], row-major
    hist_ptr,                 # *i32 [C, num_bins]
    length, C, num_bins,
    BLOCK_C: tl.constexpr,    # consecutive channels per program -> coalesced
    BLOCK_R: tl.constexpr,    # rows per step
):
    pid_c = tl.program_id(0)
    pid_r = tl.program_id(1)
    nblk_r = tl.num_programs(1)

    cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)        # [BLOCK_C]
    cmask = cols < C

    for r0 in range(pid_r * BLOCK_R, length, nblk_r * BLOCK_R):
        rows = r0 + tl.arange(0, BLOCK_R)                 # [BLOCK_R]
        rmask = rows < length
        mask = rmask[:, None] & cmask[None, :]
        ptrs = data_ptr + rows[:, None] * C + cols[None, :]
        vals = tl.load(ptrs, mask=mask, other=0).to(tl.int32)   # [BLOCK_R, BLOCK_C]
        hptrs = hist_ptr + cols[None, :] * num_bins + vals
        tl.atomic_add(hptrs, 1, mask=mask)


def custom_kernel(data: input_t) -> output_t:
    array, num_bins = data
    length, C = array.shape
    hist = torch.zeros(C, num_bins, dtype=torch.int32, device=array.device)
    BLOCK_C, BLOCK_R = 32, 16
    grid = (triton.cdiv(C, BLOCK_C), 128)
    _hist_atomics[grid](array, hist, length, C, num_bins,
                        BLOCK_C=BLOCK_C, BLOCK_R=BLOCK_R)
    return hist
