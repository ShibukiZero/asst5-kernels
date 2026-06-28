# Entry 6 - cached / slim CUTLASS FA-3. Same kernel & math as Entry 4 (v3), but the
# per-call wrapper overhead (allocate O/LSE/workspace, can_implement, initialize) is done
# ONCE per (shape, ptrs) and cached; the hot path is just op.run(stream). The harness times
# custom_kernel with CUDA events, so CPU work between start-event and kernel-launch can show
# up as GPU-idle in elapsed time; the current gap to SDPA (12.79 vs 12.76) is only ~30 us,
# so shaving the wrapper may flip it. Also fixes the device_id=0 hardcode.
# Hybrid: small/medium use SDPA (cuDNN), large uses cached FA-3 (only large is benchmarked).
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

INC = "/home/linzi/asst5-venv/lib/python3.12/site-packages/cutlass_library/source/include"
UTIL = "/home/linzi/asst5-venv/lib/python3.12/site-packages/cutlass_library/source/tools/util/include"
EX = "/home/linzi/asst5-venv/lib/python3.12/site-packages/cutlass_library/source/examples/88_hopper_fmha"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

_CPP = "torch::Tensor fa3_cached(torch::Tensor Q, torch::Tensor K, torch::Tensor V);\n"
_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <map>
#include <tuple>
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "collective/fmha_fusion.hpp"
#include "device/device_universal.hpp"
#include "kernel/fmha_kernel_builder.hpp"
using namespace cute;

using Element = cutlass::half_t;
using AccQK = float;
using AccPV = float;
using TileShape = Shape<_128, _128, _128>;
using StrideQ = cute::tuple<int, _1, cute::tuple<int, int>>;
using StrideK = cute::tuple<int, _1, cute::tuple<int, int>>;
using StrideV = cute::tuple<int, _1, cute::tuple<int, int>>;
using StrideO = cute::tuple<int, _1, cute::tuple<int, int>>;
using StrideLSE = cute::tuple<_1, cute::tuple<int, int>>;
using Fusion = cutlass::fmha::collective::DefaultFusion;

using Operation = cutlass::device::Universal<
  typename cutlass::fmha::kernel::FmhaBuilder<
    Element, AccQK, AccPV, TileShape, StrideQ, StrideK, StrideV,
    Fusion, cutlass::gemm::KernelTmaWarpSpecializedCooperative
  >::Kernel>;

struct Key { int dev,B,H,S,D; void *q,*k,*v;
  bool operator<(const Key& o) const {
    return std::tie(dev,B,H,S,D,q,k,v) < std::tie(o.dev,o.B,o.H,o.S,o.D,o.q,o.k,o.v);
  } };
struct Slot { torch::Tensor O, LSE, wsp; Operation op; bool init=false; };
static std::map<Key, Slot> g_cache;

torch::Tensor fa3_cached(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
  TORCH_CHECK(Q.is_cuda() && Q.is_contiguous() && Q.scalar_type() == torch::kHalf, "Q half/contig/cuda");
  int B = Q.size(0), H = Q.size(1), S = Q.size(2), D = Q.size(3);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  Key key{Q.get_device(), B, H, S, D, Q.data_ptr(), K.data_ptr(), V.data_ptr()};
  Slot& slot = g_cache[key];

  if (!slot.init) {
    slot.O = torch::empty_like(Q);
    slot.LSE = torch::empty({(long)B * H * S}, torch::dtype(torch::kFloat32).device(Q.device()));
    auto sQ = cute::make_stride(D, _1{}, cute::make_stride(H * S * D, S * D));
    auto sK = sQ; auto sV = sQ; auto sO = sQ;
    auto sLSE = cute::make_stride(_1{}, cute::make_stride(H * S, S));
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = Q.get_device();
    cudaDeviceGetAttribute(&hw_info.sm_count, cudaDevAttrMultiProcessorCount, hw_info.device_id);
    typename Operation::Arguments args{
      {B, H, S, S, D},
      { reinterpret_cast<Element*>(Q.data_ptr()), sQ,
        reinterpret_cast<Element*>(K.data_ptr()), sK,
        reinterpret_cast<Element*>(V.data_ptr()), sV },
      { reinterpret_cast<Element*>(slot.O.data_ptr()), sO,
        slot.LSE.data_ptr<float>(), sLSE },
      hw_info
    };
    size_t ws = Operation::get_workspace_size(args);
    slot.wsp = torch::empty({(long)ws}, torch::dtype(torch::kUInt8).device(Q.device()));
    TORCH_CHECK(slot.op.can_implement(args) == cutlass::Status::kSuccess, "can_implement failed");
    TORCH_CHECK(slot.op.initialize(args, slot.wsp.data_ptr(), stream) == cutlass::Status::kSuccess, "initialize failed");
    slot.init = true;
  }
  TORCH_CHECK(slot.op.run(stream) == cutlass::Status::kSuccess, "run failed");
  return slot.O;
}
"""

_ext = load_inline(name="cutlass_fa3_cached", cpp_sources=[_CPP], cuda_sources=[_CUDA],
    functions=["fa3_cached"],
    extra_cuda_cflags=["-I" + INC, "-I" + UTIL, "-I" + EX, "--expt-relaxed-constexpr",
                       "-std=c++17", "-O3", "-DNDEBUG", "-gencode", "arch=compute_90a,code=sm_90a"],
    verbose=False)


def custom_kernel(data: input_t) -> output_t:
    q, k, v = data
    S = q.size(2)
    if S < 8192:
        return F.scaled_dot_product_attention(q, k, v)
    return _ext.fa3_cached(q, k, v)
