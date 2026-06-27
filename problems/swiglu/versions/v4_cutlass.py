# Entry 4 — Hand-written CUTLASS 3.x (Hopper sm90) TF32 GEMM. RECORD ARTIFACT.
#
# The GEMM itself BEATS cuBLAS: 0.817 ms vs ~0.93 ms (1.14x) for [16384,2048]x[2048,4096],
# using tile 128x256x32 / cluster 2x1x1, Cooperative mainloop PAIRED with a
# TMA-Cooperative epilogue, built with -O3 -DNDEBUG. (Default builder gave 1.04 ms;
# the win came from the matched warp-specialized schedules + opt flags, which kill the
# ptxas C7510 "wgmma serialized" stall.)
#
# But this DOES NOT PASS correctness, and that is fundamental, not a bug:
#   - ref_kernel uses plain `x@W` with no set_float32_matmul_precision, so the reference
#     runs in TRUE FP32 here (this file deliberately does not set the global flag).
#   - CUTLASS-TF32 vs FP32 -> 2.625% of elements exceed allclose(rtol=1e-2, atol=1e-2)
#     (max abs diff 12.2), because the nonlinear silu(gate)*value epilogue + tight atol
#     punish TF32's ~1e-3 error wherever the output is small but the factors aren't.
#   - Setting the flag instead makes the reference cuBLAS-TF32, which CUTLASS-TF32 also
#     fails (not bit-identical) -- the Entry 3 / Triton case.
#   => The only fast path that passes is cuBLAS-with-the-flag = Entry 2. The test is
#      self-referential to cuBLAS's exact TF32 numerics. Kept Entry 2 (2.036 ms) as best.
#
# Full custom_kernel here (2 GEMMs + 2 transposes + non-fused torch epilogue) = 2.501 ms,
# slower than Entry 2 anyway; an EVT-fused epilogue would reach ~1.8 ms but still fail
# correctness, so it was not pursued. See worklog.md Entry 4.
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

INC = "/home/linzi/asst5-venv/lib/python3.12/site-packages/cutlass_library/source/include"
UTIL = "/home/linzi/asst5-venv/lib/python3.12/site-packages/cutlass_library/source/tools/util/include"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")

# eval runs tests in spawned workers where sys.stdout/stderr can be None;
# torch's JIT build touches them -> guard against 'NoneType'.flush.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

_CPP = "torch::Tensor cutlass_gemm(torch::Tensor A, torch::Tensor B);\n"

_CUDA = r"""
#include <torch/extension.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
using namespace cute;

using MyTile    = Shape<_128,_256,_32>;
using MyCluster = Shape<_2,_1,_1>;
using LA = cutlass::layout::RowMajor;
using LB = cutlass::layout::ColumnMajor;   // feed B = W.t().contiguous() ([N,K] K-major / TN)
using LC = cutlass::layout::RowMajor;
using Arch = cutlass::arch::Sm90;
using Op   = cutlass::arch::OpClassTensorOp;       // float + this -> TF32 wgmma

using Epi = cutlass::epilogue::collective::CollectiveBuilder<
    Arch, Op, MyTile, MyCluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, float, LC, 4, float, LC, 4,
    cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
using Main = cutlass::gemm::collective::CollectiveBuilder<
    Arch, Op, float, LA, 4, float, LB, 4, float, MyTile, MyCluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

// A:[M,K] rowmajor, B:[N,K] rowmajor (K-major). Returns C:[M,N] = A @ W  (B == W.t()).
torch::Tensor cutlass_gemm(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.is_contiguous() && B.is_contiguous());
  int M = A.size(0), K = A.size(1), N = B.size(0);
  auto C = torch::empty({M, N}, A.options());
  using SA = typename Gemm::GemmKernel::StrideA;
  using SB = typename Gemm::GemmKernel::StrideB;
  using SC = typename Gemm::GemmKernel::StrideC;
  SA sA = cutlass::make_cute_packed_stride(SA{}, make_shape(M, K, 1));
  SB sB = cutlass::make_cute_packed_stride(SB{}, make_shape(N, K, 1));
  SC sC = cutlass::make_cute_packed_stride(SC{}, make_shape(M, N, 1));
  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
    { A.data_ptr<float>(), sA, B.data_ptr<float>(), sB },
    { {1.0f, 0.0f}, C.data_ptr<float>(), sC, C.data_ptr<float>(), sC } };
  Gemm gemm;
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "can_implement");
  size_t ws = Gemm::get_workspace_size(args);
  auto wsp = torch::empty({(long)ws}, torch::dtype(torch::kUInt8).device(A.device()));
  TORCH_CHECK(gemm.initialize(args, wsp.data_ptr()) == cutlass::Status::kSuccess, "init");
  TORCH_CHECK(gemm.run() == cutlass::Status::kSuccess, "run");
  return C;
}
"""

_ext = load_inline(
    name="cutlass_swiglu_gemm",
    cpp_sources=[_CPP],
    cuda_sources=[_CUDA],
    functions=["cutlass_gemm"],
    extra_cuda_cflags=["-I" + INC, "-I" + UTIL, "--expt-relaxed-constexpr",
                       "-std=c++17", "-O3", "-DNDEBUG",
                       "-gencode", "arch=compute_90a,code=sm_90a"],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    x, W, V, b, c, beta = data
    B, S, K = x.shape
    N = W.size(1)
    xm = x.reshape(B * S, K).contiguous()
    Wt = W.t().contiguous()   # [N,K] K-major (TN transpose tax)
    Vt = V.t().contiguous()
    gate = _ext.cutlass_gemm(xm, Wt)    # x @ W   (TF32)
    value = _ext.cutlass_gemm(xm, Vt)   # x @ V   (TF32)
    out = F.silu(gate + b) * (value + c)
    return out.reshape(B, S, N)
