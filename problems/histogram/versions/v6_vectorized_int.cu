#include <cuda_runtime.h>
#include <cstdint>

// Entry 6 — vectorized int loads (4 channels/thread) + unroll. 0.400 ms. FAILED:
// vectorization hid the load even better (long_scoreboard 6->4.7) but forced
// 128-thread blocks (8 threads in x), crashing occupancy 92%->35% -> net slower.
#define CH     32
#define CHX    (CH / 4)
#define BINS   256
#define UNROLL 8

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

    const int lc0 = threadIdx.x * 4;
    const int c0  = blockIdx.x * CH + lc0;
    const uint32_t* __restrict__ p32 = reinterpret_cast<const uint32_t*>(data);

    if (c0 < num_channels) {
        const int row0      = blockIdx.y * blockDim.y + threadIdx.y;
        const int rowStride = gridDim.y * blockDim.y;

        int r = row0;
        for (; r + (UNROLL - 1) * rowStride < length; r += UNROLL * rowStride) {
            uint32_t w[UNROLL];
            #pragma unroll
            for (int k = 0; k < UNROLL; k++) {
                long long byteOff = (long long)(r + k * rowStride) * num_channels + c0;
                w[k] = p32[byteOff >> 2];
            }
            #pragma unroll
            for (int k = 0; k < UNROLL; k++) {
                uint32_t packed = w[k];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int v = (packed >> (8 * j)) & 0xFF;
                    atomicAdd(&s[(lc0 + j) * num_bins + v], 1);
                }
            }
        }
        for (; r < length; r += rowStride) {
            long long byteOff = (long long)r * num_channels + c0;
            uint32_t packed = p32[byteOff >> 2];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int v = (packed >> (8 * j)) & 0xFF;
                atomicAdd(&s[(lc0 + j) * num_bins + v], 1);
            }
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
    TORCH_CHECK(num_channels % 4 == 0, "num_channels must be a multiple of 4");
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(data.device());
    torch::Tensor histogram = torch::zeros({num_channels, num_bins}, options);
    dim3 block(CHX, 16);
    dim3 grid((num_channels + CH - 1) / CH, 256);
    hist_kernel<<<grid, block>>>(
        data.data_ptr<uint8_t>(), histogram.data_ptr<int>(),
        length, num_channels, num_bins);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return histogram;
}
