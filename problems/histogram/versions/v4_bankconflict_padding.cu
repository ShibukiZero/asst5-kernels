#include <cuda_runtime.h>

// Entry 4 — v3 + bank-conflict padding (stride 256 -> 257). 0.813 ms. FAILED:
// no effect. Padding only fixes bank conflicts for correlated writes; for
// uniform-random values bank=(lc+v)%32 is still random. (And bank conflicts
// were not the real bottleneck anyway — it's global-load latency.)
#define CH    32
#define BINS  256
#define SPADC (BINS + 1)   // padded per-channel stride

__global__ void hist_kernel(
    const uint8_t* __restrict__ data,
    int* __restrict__ hist,
    int length, int num_channels, int num_bins)
{
    __shared__ int s[CH * SPADC];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y;
    const int spad = num_bins + 1;

    for (int i = tid; i < CH * spad; i += nthreads) s[i] = 0;
    __syncthreads();

    const int lc = threadIdx.x;
    const int c  = blockIdx.x * CH + lc;
    if (c < num_channels) {
        const int row0      = blockIdx.y * blockDim.y + threadIdx.y;
        const int rowStride = gridDim.y * blockDim.y;
        for (int r = row0; r < length; r += rowStride) {
            uint8_t v = data[(long long)r * num_channels + c];
            atomicAdd(&s[lc * spad + (int)v], 1);
        }
    }
    __syncthreads();

    for (int i = tid; i < CH * num_bins; i += nthreads) {
        int local_c = i / num_bins;
        int bin     = i % num_bins;
        int gc      = blockIdx.x * CH + local_c;
        if (gc < num_channels) {
            int val = s[local_c * spad + bin];
            if (val) atomicAdd(&hist[gc * num_bins + bin], val);
        }
    }
}

torch::Tensor histogram_kernel(torch::Tensor data, int num_bins) {
    TORCH_CHECK(data.device().is_cuda(), "Tensor data must be a CUDA tensor");
    TORCH_CHECK(data.is_contiguous(), "data must be contiguous");
    TORCH_CHECK(num_bins <= BINS, "num_bins exceeds static shared size");
    const int length = data.size(0);
    const int num_channels = data.size(1);
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(data.device());
    torch::Tensor histogram = torch::zeros({num_channels, num_bins}, options);
    dim3 block(CH, 16);
    dim3 grid((num_channels + CH - 1) / CH, 128);
    hist_kernel<<<grid, block>>>(
        data.data_ptr<uint8_t>(), histogram.data_ptr<int>(),
        length, num_channels, num_bins);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return histogram;
}
