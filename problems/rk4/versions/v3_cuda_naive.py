# Entry 3 — Naive CUDA (per-thread 25-tap Laplacian from global, cached) + fused RK4.
# Correctness gate for the CUDA path: passes 1e-6 with BOTH -fmad=true and -fmad=false
# (max diff 4.77e-7, identical) -> the FMA contraction worry was unfounded here.
# 85.5 ms (ties Triton Entry 2; beats README's naive CUDA 148 ms via fused stage combines).
# Each thread = one (z,y,x) point; 4 kernels/step (3 stage + 1 final). No shared-memory
# reuse yet -> still ~2.3x over roofline (z-neighbors re-read). 2.5D blocking is Entry 4.
import os
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")

_CPP = "#include <ATen/cuda/CUDAContext.h>\ntorch::Tensor rk4(torch::Tensor u0, double a, double hx, double hy, double hz, int n);\n"
_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
static const float C0=-205.0/72.0, C1=8.0/5.0, C2=-1.0/5.0, C3=8.0/315.0, C4=-1.0/560.0;
__device__ __forceinline__ float lap(const float* f, long b, long sy, long sz, float ihx,float ihy,float ihz){
  float uc=f[b];
  float xx=(C0*uc + C1*(f[b+1]+f[b-1]) + C2*(f[b+2]+f[b-2]) + C3*(f[b+3]+f[b-3]) + C4*(f[b+4]+f[b-4]))*ihx;
  float yy=(C0*uc + C1*(f[b+sy]+f[b-sy]) + C2*(f[b+2*sy]+f[b-2*sy]) + C3*(f[b+3*sy]+f[b-3*sy]) + C4*(f[b+4*sy]+f[b-4*sy]))*ihy;
  float zz=(C0*uc + C1*(f[b+sz]+f[b-sz]) + C2*(f[b+2*sz]+f[b-2*sz]) + C3*(f[b+3*sz]+f[b-3*sz]) + C4*(f[b+4*sz]+f[b-4*sz]))*ihz;
  return xx+yy+zz;
}
__global__ void stage_k(const float* fld,const float* ub,float* ko,float* uo,float a,float dt,float coef,int Nz,int Ny,int Nx,float ihx,float ihy,float ihz){
  int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y,z=blockIdx.z; if(x>=Nx)return;
  long sy=Nx,sz=(long)Ny*Nx,b=(long)z*sz+(long)y*sy+x; float uh=ub[b];
  bool in=(z>=4)&&(z<Nz-4)&&(y>=4)&&(y<Ny-4)&&(x>=4)&&(x<Nx-4);
  if(in){float k=a*lap(fld,b,sy,sz,ihx,ihy,ihz); ko[b]=k; uo[b]=uh+coef*dt*k;} else uo[b]=uh;
}
__global__ void final_k(const float* fld,const float* ub,const float* k1,const float* k2,const float* k3,float* fo,float a,float dt,int Nz,int Ny,int Nx,float ihx,float ihy,float ihz){
  int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y,z=blockIdx.z; if(x>=Nx)return;
  long sy=Nx,sz=(long)Ny*Nx,b=(long)z*sz+(long)y*sy+x; float uh=ub[b];
  bool in=(z>=4)&&(z<Nz-4)&&(y>=4)&&(y<Ny-4)&&(x>=4)&&(x<Nx-4);
  if(in){float k4=a*lap(fld,b,sy,sz,ihx,ihy,ihz); fo[b]=uh+(dt/6.0f)*(k1[b]+2.0f*k2[b]+2.0f*k3[b]+k4);} else fo[b]=uh;
}
torch::Tensor rk4(torch::Tensor u0,double ad,double hxd,double hyd,double hzd,int n){
  int Nz=u0.size(0),Ny=u0.size(1),Nx=u0.size(2);
  float a=ad,hx=hxd,hy=hyd,hz=hzd,ihx=1.0f/(hx*hx),ihy=1.0f/(hy*hy),ihz=1.0f/(hz*hz),dt=0.05f/(a*(ihx+ihy+ihz));
  auto u=u0.clone(),k1=torch::empty_like(u),k2=torch::empty_like(u),k3=torch::empty_like(u);
  auto us=torch::empty_like(u),us2=torch::empty_like(u),us3=torch::empty_like(u),f=torch::empty_like(u);
  dim3 blk(256,1,1),grd((Nx+255)/256,Ny,Nz); auto st=at::cuda::getCurrentCUDAStream();
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
_ext = load_inline(name="rk4_cuda_naive", cpp_sources=[_CPP], cuda_sources=[_CUDA], functions=["rk4"],
    extra_cuda_cflags=["-O3","--expt-relaxed-constexpr","-std=c++17","-gencode","arch=compute_90a,code=sm_90a"], verbose=False)


def custom_kernel(data: input_t) -> output_t:
    u0, alpha, hx, hy, hz, n_steps = data
    return _ext.rk4(u0.contiguous(), float(alpha), float(hx), float(hy), float(hz), n_steps)
