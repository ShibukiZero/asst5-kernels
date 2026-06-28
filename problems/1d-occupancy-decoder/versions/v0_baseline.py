# Entry 0 - Baseline. Stock PyTorch reference: rebuilds the nn.Module every call.
# 18.789 ms (CUDA-event mean, 100 runs) on Nebius H100 / torch 2.12.1+cu130.
# ~6.2 ms is real GPU work; the other ~12 ms is per-call Module construction +
# 14 .copy_() launches + ctor random-init starving the GPU (see worklog Entry 0).
from reference import ref_kernel

custom_kernel = ref_kernel
