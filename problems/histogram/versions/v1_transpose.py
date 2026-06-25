# Entry 1 — Physical transpose (PyTorch). 45.36 ms (1.13x). FAILED: over-fetch
# relocated into .contiguous() (34 GB strided read).
import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    array, num_bins = data
    length, num_channels = array.shape

    arr_t = array.t().contiguous()  # physical transpose: channels now contiguous

    histogram = torch.zeros(num_channels, num_bins, dtype=torch.int32, device=array.device)
    for c in range(num_channels):
        histogram[c] = torch.bincount(arr_t[c], minlength=num_bins)[:num_bins].to(torch.int32)
    return histogram
