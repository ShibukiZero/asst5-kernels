#include <cuda_runtime.h>

// Entry 7 (BEST) — v5a + tuned launch config from a grid sweep.
// CH=32 channels/block, blockDim.y=32 row-lanes, gridDim.y=32, UNROLL=8.
// 0.343 ms, ~150x over baseline. Sweep found CH=32 is the sweet spot (16 worse,
// 64 not better) and "more row-lanes + fewer blocks" wins (fewer global flushes).
#define CH     32
#define BINS   256
#define UNROLL 8
#define BY     32   // blockDim.y (row-lanes)
#define GY     32   // gridDim.y  (row parallelism)

__global__ void hist_kernel(
    const uint8_t* __restrict__ data,
    int* __restrict__ hist,
    int length, int num_channels, int num_bins)
{
    __shared__ int s[CH * BINS];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y;

    for (int i = tid; i < CH * num_bins; i += nthreads) s[i] = 0;
    __syncthreads();

    const int lc = threadIdx.x;
    const int c  = blockIdx.x * CH + lc;
    if (c < num_channels) {
        const int row0      = blockIdx.y * blockDim.y + threadIdx.y;
        const int rowStride = gridDim.y * blockDim.y;

        int r = row0;
        for (; r + (UNROLL - 1) * rowStride < length; r += UNROLL * rowStride) {
            uint8_t v[UNROLL];
            #pragma unroll
            for (int k = 0; k < UNROLL; k++)
                v[k] = data[(long long)(r + k * rowStride) * num_channels + c];
            #pragma unroll
            for (int k = 0; k < UNROLL; k++)
                atomicAdd(&s[lc * num_bins + (int)v[k]], 1);
        }
        for (; r < length; r += rowStride) {
            uint8_t v = data[(long long)r * num_channels + c];
            atomicAdd(&s[lc * num_bins + (int)v], 1);
        }
    }
    __syncthreads();

    for (int i = tid; i < CH * num_bins; i += nthreads) {
        int local_c = i / num_bins;
        int bin     = i % num_bins;
        int gc      = blockIdx.x * CH + local_c;
        if (gc < num_channels) {
            int val = s[local_c * num_bins + bin];
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

    dim3 block(CH, BY);
    dim3 grid((num_channels + CH - 1) / CH, GY);

    hist_kernel<<<grid, block>>>(
        data.data_ptr<uint8_t>(), histogram.data_ptr<int>(),
        length, num_channels, num_bins);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return histogram;
}
