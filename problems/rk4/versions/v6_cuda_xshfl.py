# Entry 6 - Naive CUDA++ : warp-shuffle x-neighbor reuse + launch cleanup.
# Keeps Entry 3's dataflow EXACTLY (one thread / one point, 4 kernels/step, same RK4) -
# the opposite of Entry 4/5 which traded occupancy for locality. Only changes:
#   (1) x-direction stencil taps come from __shfl within the warp (lanes are contiguous
#       in x), not from L2/global re-loads; cross-warp-edge lanes fall back to global.
#       A boundary lane's uc IS the field value its interior neighbor needs, so any
#       same-warp source lane is valid -> the only fallback is lane < d (cross-warp).
#   (2) TPB swept (multiple of 32) to cut x-padding (Nx=600: 256->768 launched/row vs
#       160->640); (3) 32-bit indexing (600^3 = 216M < 2^31).
# Shuffles are issued UNCONDITIONALLY (before the interior branch) so all active lanes
# participate -> __shfl_*_sync correctness. Target: beat 85.5 ms (expect ~marginal; x is
# the contiguous/best-coalesced direction, so the L2 relief from x-reuse may be small -
# measure L2% to confirm the mechanism).
import os
import sys
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
# eval runs in spawned workers where sys.stdout/stderr can be None; load_inline touches them.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

_CPP = "#include <ATen/cuda/CUDAContext.h>\ntorch::Tensor rk4(torch::Tensor u0, double a, double hx, double hy, double hz, int n);\n"
_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#ifndef TPB
#define TPB 160
#endif
static const float C0=-205.0/72.0, C1=8.0/5.0, C2=-1.0/5.0, C3=8.0/315.0, C4=-1.0/560.0;

__device__ __forceinline__ float lap_sh(const float* __restrict__ f, int b, int sy, int sz, int lane,
    float uc, float um1,float um2,float um3,float um4, float up1,float up2,float up3,float up4,
    float ihx,float ihy,float ihz){
  float xm1=(lane>=1)?um1:f[b-1];
  float xm2=(lane>=2)?um2:f[b-2];
  float xm3=(lane>=3)?um3:f[b-3];
  float xm4=(lane>=4)?um4:f[b-4];
  float xp1=(lane<=30)?up1:f[b+1];
  float xp2=(lane<=29)?up2:f[b+2];
  float xp3=(lane<=28)?up3:f[b+3];
  float xp4=(lane<=27)?up4:f[b+4];
  float xx=(C0*uc + C1*(xp1+xm1) + C2*(xp2+xm2) + C3*(xp3+xm3) + C4*(xp4+xm4))*ihx;
  float yy=(C0*uc + C1*(f[b+sy]+f[b-sy]) + C2*(f[b+2*sy]+f[b-2*sy]) + C3*(f[b+3*sy]+f[b-3*sy]) + C4*(f[b+4*sy]+f[b-4*sy]))*ihy;
  float zz=(C0*uc + C1*(f[b+sz]+f[b-sz]) + C2*(f[b+2*sz]+f[b-2*sz]) + C3*(f[b+3*sz]+f[b-3*sz]) + C4*(f[b+4*sz]+f[b-4*sz]))*ihz;
  return xx+yy+zz;
}

__global__ void stage_k(const float* fld,const float* ub,float* ko,float* uo,float a,float dt,float coef,int Nz,int Ny,int Nx,float ihx,float ihy,float ihz){
  int x=blockIdx.x*TPB+threadIdx.x, y=blockIdx.y, z=blockIdx.z; if(x>=Nx)return;
  int sy=Nx, sz=Ny*Nx, b=z*sz+y*sy+x; float uh=ub[b];
  float uc=fld[b];
  unsigned m=__activemask(); int lane=threadIdx.x&31;
  float um1=__shfl_up_sync(m,uc,1),um2=__shfl_up_sync(m,uc,2),um3=__shfl_up_sync(m,uc,3),um4=__shfl_up_sync(m,uc,4);
  float up1=__shfl_down_sync(m,uc,1),up2=__shfl_down_sync(m,uc,2),up3=__shfl_down_sync(m,uc,3),up4=__shfl_down_sync(m,uc,4);
  bool in=(z>=4)&&(z<Nz-4)&&(y>=4)&&(y<Ny-4)&&(x>=4)&&(x<Nx-4);
  if(in){float k=a*lap_sh(fld,b,sy,sz,lane,uc,um1,um2,um3,um4,up1,up2,up3,up4,ihx,ihy,ihz); ko[b]=k; uo[b]=uh+coef*dt*k;} else uo[b]=uh;
}

__global__ void final_k(const float* fld,const float* ub,const float* k1,const float* k2,const float* k3,float* fo,float a,float dt,int Nz,int Ny,int Nx,float ihx,float ihy,float ihz){
  int x=blockIdx.x*TPB+threadIdx.x, y=blockIdx.y, z=blockIdx.z; if(x>=Nx)return;
  int sy=Nx, sz=Ny*Nx, b=z*sz+y*sy+x; float uh=ub[b];
  float uc=fld[b];
  unsigned m=__activemask(); int lane=threadIdx.x&31;
  float um1=__shfl_up_sync(m,uc,1),um2=__shfl_up_sync(m,uc,2),um3=__shfl_up_sync(m,uc,3),um4=__shfl_up_sync(m,uc,4);
  float up1=__shfl_down_sync(m,uc,1),up2=__shfl_down_sync(m,uc,2),up3=__shfl_down_sync(m,uc,3),up4=__shfl_down_sync(m,uc,4);
  bool in=(z>=4)&&(z<Nz-4)&&(y>=4)&&(y<Ny-4)&&(x>=4)&&(x<Nx-4);
  if(in){float k4=a*lap_sh(fld,b,sy,sz,lane,uc,um1,um2,um3,um4,up1,up2,up3,up4,ihx,ihy,ihz); fo[b]=uh+(dt/6.0f)*(k1[b]+2.0f*k2[b]+2.0f*k3[b]+k4);} else fo[b]=uh;
}

torch::Tensor rk4(torch::Tensor u0,double ad,double hxd,double hyd,double hzd,int n){
  int Nz=u0.size(0),Ny=u0.size(1),Nx=u0.size(2);
  float a=ad,hx=hxd,hy=hyd,hz=hzd,ihx=1.0f/(hx*hx),ihy=1.0f/(hy*hy),ihz=1.0f/(hz*hz),dt=0.05f/(a*(ihx+ihy+ihz));
  auto u=u0.clone(),k1=torch::empty_like(u),k2=torch::empty_like(u),k3=torch::empty_like(u);
  auto us=torch::empty_like(u),us2=torch::empty_like(u),us3=torch::empty_like(u),f=torch::empty_like(u);
  dim3 blk(TPB,1,1),grd((Nx+TPB-1)/TPB,Ny,Nz); auto st=at::cuda::getCurrentCUDAStream();
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
_TPB = int(os.environ.get("RK4_TPB", "160"))
_ext = load_inline(name=f"rk4_cuda_xshfl_{_TPB}", cpp_sources=[_CPP], cuda_sources=[_CUDA], functions=["rk4"],
    extra_cuda_cflags=["-O3","--expt-relaxed-constexpr","-std=c++17",f"-DTPB={_TPB}",
                       "-gencode","arch=compute_90a,code=sm_90a"], verbose=False)


def custom_kernel(data: input_t) -> output_t:
    u0, alpha, hx, hy, hz, n_steps = data
    return _ext.rk4(u0.contiguous(), float(alpha), float(hx), float(hy), float(hz), n_steps)
