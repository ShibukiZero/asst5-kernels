# Entry 7 - per-z-plane shared-memory x/y tile (NO z-march). The real attempt at the
# L2 wall (Entry 6 showed it's at 97%, driven by y/z strided loads).
# This is NOT Entry 4: Entry 4 died from z-march (serial z, 2 barriers/z, ~12% occupancy).
# Here each block does ONE z-plane, one thread per output point, block count stays in the
# hundreds of thousands, ONE __syncthreads. Shared tile holds the (BX+8)x(BY+8) plane
# neighbourhood -> x/y stencil from SRAM (17 taps -> ~2.25 field loads/point); z's 8 taps
# still come from global/L2. Same Entry-3 dataflow (4 kernels/step, same RK4).
# Success = L2 drops AND occupancy holds (>=75%) AND runtime < 85 ms.
import os
import sys
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

_CPP = "#include <ATen/cuda/CUDAContext.h>\ntorch::Tensor rk4(torch::Tensor u0, double a, double hx, double hy, double hz, int n);\n"
_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#ifndef BX
#define BX 64
#endif
#ifndef BY
#define BY 8
#endif
#define H 4
#define SMW (BX + 2*H)
#define SMH (BY + 2*H)
static const float C0=-205.0/72.0, C1=8.0/5.0, C2=-1.0/5.0, C3=8.0/315.0, C4=-1.0/560.0;

// fld and ub may alias (stage 1: both = u) -> NOT __restrict__.
__device__ __forceinline__ void load_tile(const float* fld, float sm[SMH][SMW],
        int x0, int y0, int z, int sz, int Nx, int Ny){
  int tid = threadIdx.y * BX + threadIdx.x;
  for (int idx = tid; idx < SMW*SMH; idx += BX*BY){
    int sj = idx / SMW, si = idx - sj*SMW;
    int lx = x0 + si - H, ly = y0 + sj - H;
    float v = 0.0f;
    if (lx >= 0 && lx < Nx && ly >= 0 && ly < Ny) v = fld[z*sz + ly*Nx + lx];
    sm[sj][si] = v;
  }
}

__device__ __forceinline__ float lap_tile(float sm[SMH][SMW], int si, int sj,
        const float* fld, int b, int sz, float ihx, float ihy, float ihz){
  float uc = sm[sj][si];
  float xx=(C0*uc + C1*(sm[sj][si+1]+sm[sj][si-1]) + C2*(sm[sj][si+2]+sm[sj][si-2]) + C3*(sm[sj][si+3]+sm[sj][si-3]) + C4*(sm[sj][si+4]+sm[sj][si-4]))*ihx;
  float yy=(C0*uc + C1*(sm[sj+1][si]+sm[sj-1][si]) + C2*(sm[sj+2][si]+sm[sj-2][si]) + C3*(sm[sj+3][si]+sm[sj-3][si]) + C4*(sm[sj+4][si]+sm[sj-4][si]))*ihy;
  float zz=(C0*uc + C1*(fld[b+sz]+fld[b-sz]) + C2*(fld[b+2*sz]+fld[b-2*sz]) + C3*(fld[b+3*sz]+fld[b-3*sz]) + C4*(fld[b+4*sz]+fld[b-4*sz]))*ihz;
  return xx+yy+zz;
}

__global__ void stage_k(const float* fld,const float* ub,float* ko,float* uo,float a,float dt,float coef,int Nz,int Ny,int Nx,float ihx,float ihy,float ihz){
  int x0=blockIdx.x*BX, y0=blockIdx.y*BY, z=blockIdx.z;
  int x=x0+threadIdx.x, y=y0+threadIdx.y;
  int sy=Nx, sz=Ny*Nx;
  bool valid=(x<Nx)&&(y<Ny);
  if (z<H || z>=Nz-H){ if(valid){ int b=z*sz+y*sy+x; uo[b]=ub[b]; } return; }
  __shared__ float sm[SMH][SMW];
  load_tile(fld, sm, x0, y0, z, sz, Nx, Ny);
  __syncthreads();
  if(!valid) return;
  int b=z*sz+y*sy+x; float uh=ub[b];
  bool in=(x>=H)&&(x<Nx-H)&&(y>=H)&&(y<Ny-H);
  if(in){ float k=a*lap_tile(sm, threadIdx.x+H, threadIdx.y+H, fld, b, sz, ihx, ihy, ihz); ko[b]=k; uo[b]=uh+coef*dt*k; }
  else uo[b]=uh;
}

__global__ void final_k(const float* fld,const float* ub,const float* k1,const float* k2,const float* k3,float* fo,float a,float dt,int Nz,int Ny,int Nx,float ihx,float ihy,float ihz){
  int x0=blockIdx.x*BX, y0=blockIdx.y*BY, z=blockIdx.z;
  int x=x0+threadIdx.x, y=y0+threadIdx.y;
  int sy=Nx, sz=Ny*Nx;
  bool valid=(x<Nx)&&(y<Ny);
  if (z<H || z>=Nz-H){ if(valid){ int b=z*sz+y*sy+x; fo[b]=ub[b]; } return; }
  __shared__ float sm[SMH][SMW];
  load_tile(fld, sm, x0, y0, z, sz, Nx, Ny);
  __syncthreads();
  if(!valid) return;
  int b=z*sz+y*sy+x; float uh=ub[b];
  bool in=(x>=H)&&(x<Nx-H)&&(y>=H)&&(y<Ny-H);
  if(in){ float k4=a*lap_tile(sm, threadIdx.x+H, threadIdx.y+H, fld, b, sz, ihx, ihy, ihz); fo[b]=uh+(dt/6.0f)*(k1[b]+2.0f*k2[b]+2.0f*k3[b]+k4); }
  else fo[b]=uh;
}

torch::Tensor rk4(torch::Tensor u0,double ad,double hxd,double hyd,double hzd,int n){
  int Nz=u0.size(0),Ny=u0.size(1),Nx=u0.size(2);
  float a=ad,hx=hxd,hy=hyd,hz=hzd,ihx=1.0f/(hx*hx),ihy=1.0f/(hy*hy),ihz=1.0f/(hz*hz),dt=0.05f/(a*(ihx+ihy+ihz));
  auto u=u0.clone(),k1=torch::empty_like(u),k2=torch::empty_like(u),k3=torch::empty_like(u);
  auto us=torch::empty_like(u),us2=torch::empty_like(u),us3=torch::empty_like(u),f=torch::empty_like(u);
  dim3 blk(BX,BY,1),grd((Nx+BX-1)/BX,(Ny+BY-1)/BY,Nz); auto st=at::cuda::getCurrentCUDAStream();
  for(int s=0;s<n;s++){
    stage_k<<<grd,blk,0,st>>>(u.data_ptr<float>(),u.data_ptr<float>(),k1.data_ptr<float>(),us.data_ptr<float>(),a,dt,0.5f,Nz,Ny,Nx,ihx,ihy,ihz);
    stage_k<<<grd,blk,0,st>>>(us.data_ptr<float>(),u.data_ptr<float>(),k2.data_ptr<float>(),us2.data_ptr<float>(),a,dt,0.5f,Nz,Ny,Nx,ihx,ihy,ihz);
    stage_k<<<grd,blk,0,st>>>(us2.data_ptr<float>(),u.data_ptr<float>(),k3.data_ptr<float>(),us3.data_ptr<float>(),a,dt,1.0f,Nz,Ny,Nx,ihx,ihy,ihz);
    final_k<<<grd,blk,0,st>>>(us3.data_ptr<float>(),u.data_ptr<float>(),k1.data_ptr<float>(),k2.data_ptr<float>(),k3.data_ptr<float>(),f.data_ptr<float>(),a,dt,Nz,Ny,Nx,ihx,ihy,ihz);
    auto t=u;u=f;f=t;
  }
  return u;
}
"""
_BX = int(os.environ.get("RK4_BX", "64"))
_BY = int(os.environ.get("RK4_BY", "8"))
_ext = load_inline(name=f"rk4_plane_{_BX}_{_BY}", cpp_sources=[_CPP], cuda_sources=[_CUDA], functions=["rk4"],
    extra_cuda_cflags=["-O3","--expt-relaxed-constexpr","-std=c++17",f"-DBX={_BX}",f"-DBY={_BY}",
                       "-gencode","arch=compute_90a,code=sm_90a"], verbose=False)


def custom_kernel(data: input_t) -> output_t:
    u0, alpha, hx, hy, hz, n_steps = data
    return _ext.rk4(u0.contiguous(), float(alpha), float(hx), float(hy), float(hz), n_steps)
