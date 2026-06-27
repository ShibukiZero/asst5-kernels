# Triton best-effort — Variant B: one-hot privatization, NO per-element atomics.
# The "Triton-idiomatic" answer to CUDA Entry 3's shared-mem privatization.
# Each program owns BLOCK_C channels x a row-block; it builds a private
# [NUM_BINS, BLOCK_C] histogram in registers by one-hot comparing the loaded
# tile against all bins (3D broadcast) and tl.sum-reducing over rows, then
# flushes once with atomic_add. No shared-mem atomics needed -> but pays a
# NUM_BINS x compute blow-up (256 comparisons per element).
import torch
import triton
import triton.language as tl
from task import input_t, output_t


@triton.jit
def _hist_onehot(
    data_ptr,                 # *u8  [length, C]
    hist_ptr,                 # *i32 [C, num_bins]
    length, C, num_bins,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    NUM_BINS: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_r = tl.program_id(1)
    nblk_r = tl.num_programs(1)

    cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)            # [BLOCK_C]
    cmask = cols < C
    bins = tl.arange(0, NUM_BINS)                             # [NUM_BINS]

    acc = tl.zeros((NUM_BINS, BLOCK_C), dtype=tl.int32)
    for r0 in range(pid_r * BLOCK_R, length, nblk_r * BLOCK_R):
        rows = r0 + tl.arange(0, BLOCK_R)                     # [BLOCK_R]
        rmask = rows < length
        ld_mask = rmask[:, None] & cmask[None, :]
        ptrs = data_ptr + rows[:, None] * C + cols[None, :]
        vals = tl.load(ptrs, mask=ld_mask, other=NUM_BINS).to(tl.int32)  # [BLOCK_R, BLOCK_C]
        # one-hot: [NUM_BINS, BLOCK_R, BLOCK_C], reduce over rows -> [NUM_BINS, BLOCK_C]
        oneh = (vals[None, :, :] == bins[:, None, None]).to(tl.int32)
        acc += tl.sum(oneh, axis=1)

    # flush once: hist[cols, bins] += acc
    hptrs = hist_ptr + cols[None, :] * num_bins + bins[:, None]   # [NUM_BINS, BLOCK_C]
    tl.atomic_add(hptrs, acc, mask=cmask[None, :])


def custom_kernel(data: input_t) -> output_t:
    array, num_bins = data
    length, C = array.shape
    hist = torch.zeros(C, num_bins, dtype=torch.int32, device=array.device)
    BLOCK_C, BLOCK_R = 16, 8
    grid = (triton.cdiv(C, BLOCK_C), 256)
    _hist_onehot[grid](array, hist, length, C, num_bins,
                       BLOCK_C=BLOCK_C, BLOCK_R=BLOCK_R, NUM_BINS=num_bins)
    return hist
