#include <cuda_runtime.h>

// Entry 2 — fused single kernel, coalesced read, GLOBAL atomics. 6.77 ms (7.6x).
// thread.x -> channel (consecutive threads -> consecutive channels -> coalesced);
// grid-stride over rows; atomicAdd into the global [channels, bins] histogram.
__global__ void hist_kernel(
    const uint8_t* __restrict__ data,
    int* __restrict__ hist,
    int length, int num_channels, int num_bins)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_channels) return;
    for (int r = blockIdx.y; r < length; r += gridDim.y) {
        uint8_t v = data[(long long)r * num_channels + c];
        atomicAdd(&hist[c * num_bins + (int)v], 1);
    }
}

torch::Tensor histogram_kernel(torch::Tensor data, int num_bins) {
    TORCH_CHECK(data.device().is_cuda(), "Tensor data must be a CUDA tensor");
    TORCH_CHECK(data.is_contiguous(), "data must be contiguous");
    const int length = data.size(0);
    const int num_channels = data.size(1);
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(data.device());
    torch::Tensor histogram = torch::zeros({num_channels, num_bins}, options);
    const int block = 256;
    dim3 grid((num_channels + block - 1) / block, 2048);
    hist_kernel<<<grid, block>>>(
        data.data_ptr<uint8_t>(), histogram.data_ptr<int>(),
        length, num_channels, num_bins);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return histogram;
}
