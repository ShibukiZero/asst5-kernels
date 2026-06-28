#include <cuda_runtime.h>

// Entry 10 — CH=128 + uint32 (uchar4) vectorized load + lane-major conflict-free shared.
// Pushes Entry 9's idea one step: each lane reads ONE uint32 = 4 consecutive channels
// per load (128 bytes/warp/instr, 4x the byte version), 4 shared atomics per load.
// Lane-major slot = j*32 + lane ⇒ s[v*CH + slot] has bank = (v*128 + j*32 + lane)%32
// = lane ⇒ conflict-free for every j. RISK: 128 KiB dynamic shared ⇒ only 1 block/SM
// (228 KiB/SM), so block=(32,32)=1024 threads caps occupancy at ~50% (Entry 7 trap).
#define CH     128
#define BINS   256
#define VEC    4
#define UNROLL 8
#define BY     32   // blockDim.y (row-lanes)
#define GY     32   // gridDim.y  (row parallelism)

__global__ void hist_kernel(
    const uint8_t* __restrict__ data,
    int* __restrict__ hist,
    int length, int num_channels, int num_bins)
{
    extern __shared__ int s[];   // CH * num_bins ints, lane-major: s[bin*CH + j*32 + lane]

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y;

    for (int i = tid; i < CH * num_bins; i += nthreads) s[i] = 0;
    __syncthreads();

    const int lane = threadIdx.x;                 // 0..31
    const int c0   = blockIdx.x * CH + lane * VEC; // first of this lane's 4 channels

    if (c0 < num_channels) {
        const int row0      = blockIdx.y * blockDim.y + threadIdx.y;
        const int rowStride = gridDim.y * blockDim.y;
        const bool h1 = (c0 + 1) < num_channels;
        const bool h2 = (c0 + 2) < num_channels;
        const bool h3 = (c0 + 3) < num_channels;

        int r = row0;
        for (; r + (UNROLL - 1) * rowStride < length; r += UNROLL * rowStride) {
            uint32_t v[UNROLL];
            #pragma unroll
            for (int k = 0; k < UNROLL; k++) {
                long long off = (long long)(r + k * rowStride) * num_channels + c0;
                v[k] = *reinterpret_cast<const uint32_t*>(data + off);
            }
            #pragma unroll
            for (int k = 0; k < UNROLL; k++) {
                int v0 =  v[k]        & 0xff;
                int v1 = (v[k] >>  8) & 0xff;
                int v2 = (v[k] >> 16) & 0xff;
                int v3 = (v[k] >> 24) & 0xff;
                atomicAdd(&s[v0 * CH + 0 * 32 + lane], 1);
                if (h1) atomicAdd(&s[v1 * CH + 1 * 32 + lane], 1);
                if (h2) atomicAdd(&s[v2 * CH + 2 * 32 + lane], 1);
                if (h3) atomicAdd(&s[v3 * CH + 3 * 32 + lane], 1);
            }
        }
        for (; r < length; r += rowStride) {
            long long off = (long long)r * num_channels + c0;
            uint32_t pv = *reinterpret_cast<const uint32_t*>(data + off);
            int v0 =  pv        & 0xff;
            int v1 = (pv >>  8) & 0xff;
            int v2 = (pv >> 16) & 0xff;
            int v3 = (pv >> 24) & 0xff;
            atomicAdd(&s[v0 * CH + 0 * 32 + lane], 1);
            if (h1) atomicAdd(&s[v1 * CH + 1 * 32 + lane], 1);
            if (h2) atomicAdd(&s[v2 * CH + 2 * 32 + lane], 1);
            if (h3) atomicAdd(&s[v3 * CH + 3 * 32 + lane], 1);
        }
    }
    __syncthreads();

    // flush: reverse lane-major map. slot = j*32 + lane2 ; local_c = lane2*VEC + j
    for (int i = tid; i < CH * num_bins; i += nthreads) {
        int bin   = i / CH;
        int slot  = i % CH;
        int j     = slot / 32;
        int lane2 = slot % 32;
        int local_c = lane2 * VEC + j;
        int gc = blockIdx.x * CH + local_c;
        if (gc < num_channels) {
            int val = s[bin * CH + slot];
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
    // uint32 vectorized load requires num_channels % 4 == 0 (4-byte aligned offsets).
    TORCH_CHECK((num_channels % 4) == 0, "v9 requires num_channels % 4 == 0");

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(data.device());
    torch::Tensor histogram = torch::zeros({num_channels, num_bins}, options);

    dim3 block(32, BY);
    dim3 grid((num_channels + CH - 1) / CH, GY);
    size_t shmem = (size_t)CH * num_bins * sizeof(int);

    cudaFuncSetAttribute(hist_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);

    hist_kernel<<<grid, block, shmem>>>(
        data.data_ptr<uint8_t>(), histogram.data_ptr<int>(),
        length, num_channels, num_bins);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return histogram;
}
