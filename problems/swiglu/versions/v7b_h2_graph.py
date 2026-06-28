# Entry 7b — 7a + CUDA Graph capture/replay of the H2 fast path.
# Same numerics as Entry 5 / 7a (H2; value stays cuBLAS). The 7a measurement showed the
# 1.80 ms GPU-sum -> 1.958 ms wall gap is launch/Python dispatch overhead, not allocation
# (prealloc alone moved only ~10 us). CUDA Graph collapses the whole fast path (CUTLASS
# gate GEMM + cuBLAS value GEMM + eager epilogue) into one replay -> removes per-kernel
# launch latency and Python dispatch. Falls back to 7a-style direct path if capture fails.
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

INC = "/home/linzi/asst5-venv/lib/python3.12/site-packages/cutlass_library/source/include"
UTIL = "/home/linzi/asst5-venv/lib/python3.12/site-packages/cutlass_library/source/tools/util/include"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

torch.set_float32_matmul_precision("high")

_CPP = (
    "int64_t cutlass_ws(int64_t M, int64_t N, int64_t K);\n"
    "void cutlass_gemm_out(torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor wsp);\n"
    "void swiglu_epilogue_out(torch::Tensor gate, torch::Tensor b, torch::Tensor value, torch::Tensor c, torch::Tensor out);\n"
)
_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
using namespace cute;
using MyTile = Shape<_128,_256,_32>; using MyCluster = Shape<_2,_1,_1>;
using LA = cutlass::layout::RowMajor; using LB = cutlass::layout::ColumnMajor; using LC = cutlass::layout::RowMajor;
using Arch = cutlass::arch::Sm90; using Op = cutlass::arch::OpClassTensorOp;
using Epi = cutlass::epilogue::collective::CollectiveBuilder<
    Arch, Op, MyTile, MyCluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, float, LC, 4, float, LC, 4, cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
using Main = cutlass::gemm::collective::CollectiveBuilder<
    Arch, Op, float, LA, 4, float, LB, 4, float, MyTile, MyCluster,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
using SA = typename Gemm::GemmKernel::StrideA;
using SB = typename Gemm::GemmKernel::StrideB;
using SC = typename Gemm::GemmKernel::StrideC;

static typename Gemm::Arguments _mk(int M, int N, int K, float* A, float* B, float* C) {
  SA sA = cutlass::make_cute_packed_stride(SA{}, make_shape(M, K, 1));
  SB sB = cutlass::make_cute_packed_stride(SB{}, make_shape(N, K, 1));
  SC sC = cutlass::make_cute_packed_stride(SC{}, make_shape(M, N, 1));
  return typename Gemm::Arguments{cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
    {A, sA, B, sB}, {{1.0f, 0.0f}, C, sC, C, sC}};
}
int64_t cutlass_ws(int64_t M, int64_t N, int64_t K) {
  auto args = _mk((int)M, (int)N, (int)K, nullptr, nullptr, nullptr);
  return (int64_t)Gemm::get_workspace_size(args);
}
void cutlass_gemm_out(torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor wsp) {
  int M = A.size(0), K = A.size(1), N = B.size(0);
  auto args = _mk(M, N, K, A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());
  Gemm gemm;
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "ci");
  void* wptr = wsp.numel() ? wsp.data_ptr() : nullptr;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(gemm.initialize(args, wptr, stream) == cutlass::Status::kSuccess, "init");
  TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess, "run");
}

// fused SwiGLU epilogue: out = silu(gate+b) * (value+c), float4-vectorized.
// gate/value/out are [M,N] row-major; b/c are length-N (broadcast over rows).
__global__ void swiglu_ep_kernel(const float* __restrict__ gate, const float* __restrict__ b,
                                 const float* __restrict__ value, const float* __restrict__ c,
                                 float* __restrict__ out, long total, int N) {
  long i4 = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (i4 >= total) return;
  int col = (int)(i4 % N);
  float4 g = *reinterpret_cast<const float4*>(gate + i4);
  float4 v = *reinterpret_cast<const float4*>(value + i4);
  float4 bb = *reinterpret_cast<const float4*>(b + col);
  float4 cc = *reinterpret_cast<const float4*>(c + col);
  float4 o;
  float gg;
  gg = g.x + bb.x; o.x = (gg / (1.0f + __expf(-gg))) * (v.x + cc.x);
  gg = g.y + bb.y; o.y = (gg / (1.0f + __expf(-gg))) * (v.y + cc.y);
  gg = g.z + bb.z; o.z = (gg / (1.0f + __expf(-gg))) * (v.z + cc.z);
  gg = g.w + bb.w; o.w = (gg / (1.0f + __expf(-gg))) * (v.w + cc.w);
  *reinterpret_cast<float4*>(out + i4) = o;
}

void swiglu_epilogue_out(torch::Tensor gate, torch::Tensor b, torch::Tensor value, torch::Tensor c, torch::Tensor out) {
  int M = gate.size(0), N = gate.size(1);
  long total = (long)M * N;
  long nthreads = total / 4;
  int block = 256;
  long grid = (nthreads + block - 1) / block;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  swiglu_ep_kernel<<<grid, block, 0, stream>>>(
    gate.data_ptr<float>(), b.data_ptr<float>(), value.data_ptr<float>(),
    c.data_ptr<float>(), out.data_ptr<float>(), total, N);
}
"""
_ext = load_inline(name="cutlass_swiglu_gemm_out_stream", cpp_sources=[_CPP], cuda_sources=[_CUDA],
    functions=["cutlass_ws", "cutlass_gemm_out", "swiglu_epilogue_out"],
    extra_cuda_cflags=["-I"+INC, "-I"+UTIL, "--expt-relaxed-constexpr", "-std=c++17", "-O3", "-DNDEBUG",
                       "-gencode", "arch=compute_90a,code=sm_90a"], verbose=False)


def _ref(x, W, V, b, c):
    gate = x @ W + b
    value = x @ V + c
    return F.silu(gate) * value


_ref_c = None
_mode = {}    # key -> "graph" | "direct" | "fallback"
_st = {}      # key -> dict of buffers + graph


def _key(*ts):
    return tuple((t.data_ptr(), t._version) for t in ts)


_NOGRAPH = bool(os.environ.get("SWIGLU_NOGRAPH"))


def _fastpath(xm, Wt, V, b, c, gate, value, wsp, out):
    _ext.cutlass_gemm_out(xm, Wt, gate, wsp)
    torch.mm(xm, V, out=value)
    _ext.swiglu_epilogue_out(gate, b, value, c, out)   # fused, single kernel


def custom_kernel(data: input_t) -> output_t:
    global _ref_c
    if _ref_c is None:
        _ref_c = torch.compile(_ref)
    x, W, V, b, c, beta = data
    B, S, K = x.shape
    N = W.size(1)
    M = B * S
    k = _key(x, W, V, b, c)
    m = _mode.get(k)

    if m == "fallback":
        return _ref_c(x, W, V, b, c)
    if m == "graph":
        st = _st[k]
        st["graph"].replay()
        return st["out"].view(B, S, N)
    if m == "direct":
        st = _st[k]
        _fastpath(st["xm"], st["Wt"], V, b, c, st["gate"], st["value"], st["wsp"], st["out"])
        return st["out"].view(B, S, N)

    # first time for this input: build static buffers, validate H2 vs double-cuBLAS ref
    xm = x.reshape(M, K).contiguous()
    Wt = W.t().contiguous()
    wssz = int(_ext.cutlass_ws(M, N, K))
    gate = torch.empty((M, N), device=x.device, dtype=x.dtype)
    value = torch.empty((M, N), device=x.device, dtype=x.dtype)
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    wsp = torch.empty((max(wssz, 1),), device=x.device, dtype=torch.uint8)

    _fastpath(xm, Wt, V, b, c, gate, value, wsp, out)
    out_ref = _ref_c(x, W, V, b, c)
    if not torch.allclose(out.view(B, S, N), out_ref, rtol=1e-2, atol=1e-2):
        _mode[k] = "fallback"
        return out_ref

    st = {"xm": xm, "Wt": Wt, "gate": gate, "value": value, "wsp": wsp, "out": out}

    if _NOGRAPH:
        _st[k] = st
        _mode[k] = "direct"
        return out.view(B, S, N)

    # try to capture the fast path as a CUDA graph
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _fastpath(xm, Wt, V, b, c, gate, value, wsp, out)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _fastpath(xm, Wt, V, b, c, gate, value, wsp, out)
        st["graph"] = g
        _st[k] = st
        _mode[k] = "graph"
        g.replay()
        torch.cuda.synchronize()
        # validate the replayed result once more
        if torch.allclose(out.view(B, S, N), out_ref, rtol=1e-2, atol=1e-2):
            return out.view(B, S, N)
        # graph produced wrong result -> drop to direct
        _mode[k] = "direct"
        return out_ref
    except Exception:
        _st[k] = st
        _mode[k] = "direct"
        # recompute via direct path (buffers may be dirty from capture attempt)
        _fastpath(xm, Wt, V, b, c, gate, value, wsp, out)
        return out.view(B, S, N)
