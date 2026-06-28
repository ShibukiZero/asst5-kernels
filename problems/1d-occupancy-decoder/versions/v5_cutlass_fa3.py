# Entry 5 - CUTLASS FA-3 (example 88 Hopper warp-spec FMHA) for the SDPA. NOT ADOPTED.
# Adapted from flashattention/v3_cutlass_fa3.py for this shape: head_dim 64
# (TileShape K-mode=_64), separate Q vs K/V strides (q_len 250k != kv_len 1024).
# Best variant = WarpSpec Cooperative TileShape<128,128,64> = 2.084 ms isolated
# attention = 0.85x cudnn (1.78 ms). LOSES to cudnn AND to the hand Triton (0.90x).
# Why: kv_len=1024 is too short to fill FA-3's warp-spec producer/consumer pipeline
# -> occupancy collapses to 13.6% (ncu). Real FA-3 is for long/balanced kv, not the
# huge-Q/tiny-KV shape. Submission stays Entry 2 (torch.compile, cudnn SDPA, 4.070 ms).
# (Full buildable source: tmp/decoder_fa3_128n.py. Sweep: cooperative/pingpong x
#  128x64x64 / 128x128x64, all 0.76-0.85x cudnn.)
import os, sys, torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

INC = "/home/linzi/asst5-venv/lib/python3.12/site-packages/cutlass_library/source/include"
UTIL = "/home/linzi/asst5-venv/lib/python3.12/site-packages/cutlass_library/source/tools/util/include"
EX = "/home/linzi/asst5-venv/lib/python3.12/site-packages/cutlass_library/source/examples/88_hopper_fmha"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")

_CPP = "torch::Tensor fa3(torch::Tensor Q, torch::Tensor K, torch::Tensor V);\n"
_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "collective/fmha_fusion.hpp"
#include "device/device_universal.hpp"
#include "kernel/fmha_kernel_builder.hpp"
using namespace cute;

using Element = cutlass::half_t;
using AccQK = float;
using AccPV = float;
using TileShape = Shape<_128, _128, _64>;   // head_dim=64 cooperative (example 88 run_fwd_64)
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

torch::Tensor fa3(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
  TORCH_CHECK(Q.is_cuda() && Q.is_contiguous() && Q.scalar_type() == torch::kHalf, "Q half/contig/cuda");
  int B = Q.size(0), H = Q.size(1), Sq = Q.size(2), D = Q.size(3);
  int Sk = K.size(2);
  auto O = torch::empty_like(Q);
  auto LSE = torch::empty({(long)B * H * Sq}, torch::dtype(torch::kFloat32).device(Q.device()));

  auto sQ = cute::make_stride(D, _1{}, cute::make_stride(H * Sq * D, Sq * D));
  auto sK = cute::make_stride(D, _1{}, cute::make_stride(H * Sk * D, Sk * D));
  auto sV = sK;
  auto sO = sQ;
  auto sLSE = cute::make_stride(_1{}, cute::make_stride(H * Sq, Sq));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  cudaDeviceGetAttribute(&hw_info.sm_count, cudaDevAttrMultiProcessorCount, 0);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  typename Operation::Arguments args{
    {B, H, Sq, Sk, D},
    { reinterpret_cast<Element*>(Q.data_ptr()), sQ,
      reinterpret_cast<Element*>(K.data_ptr()), sK,
      reinterpret_cast<Element*>(V.data_ptr()), sV },
    { reinterpret_cast<Element*>(O.data_ptr()), sO,
      LSE.data_ptr<float>(), sLSE },
    hw_info
  };

  Operation op;
  size_t ws = Operation::get_workspace_size(args);
  auto wsp = torch::empty({(long)ws}, torch::dtype(torch::kUInt8).device(Q.device()));
  TORCH_CHECK(op.can_implement(args) == cutlass::Status::kSuccess, "can_implement failed");
  TORCH_CHECK(op.initialize(args, wsp.data_ptr(), stream) == cutlass::Status::kSuccess, "initialize failed");
  TORCH_CHECK(op.run(stream) == cutlass::Status::kSuccess, "run failed");
  return O;
}
"""

