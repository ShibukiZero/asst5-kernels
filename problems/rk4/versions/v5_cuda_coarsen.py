# Entry 5 — CUDA x-register-coarsening (each thread does CO consecutive x, x-run
# reused in registers to cut the L2-bound naive's load count). RECORD ARTIFACT:
# correct (1e-6) but SLOWER than naive. CO=4: 145-177 ms; CO=8: 340-377 ms; naive 85.
# ncu: occupancy 66% (vs naive 80%), DRAM 6.6% / L2 58% / SM 40% -> NOTHING saturated
# = latency-bound. Coarsening cut L2 traffic but the 4x-work/thread + register pressure
# starved parallelism -> can't hide memory latency. Same lesson as the 2.5D attempt:
# this problem wants massive parallelism, not locality tricks. Kept naive (85 ms). See E5.
import os, sys, torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
if sys.stdout is None: sys.stdout = open(os.devnull, "w")
if sys.stderr is None: sys.stderr = open(os.devnull, "w")
_CPP = "#include <ATen/cuda/CUDAContext.h>\ntorch::Tensor rk4(torch::Tensor u0,double a,double hx,double hy,double hz,int n);\n"
_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#define BX 64
#define BY 8
#define CO 4
static const float C0=-205.0/72.0,C1=8.0/5.0,C2=-1.0/5.0,C3=8.0/315.0,C4=-1.0/560.0;
template<bool FINAL>
__global__ void march(const float* field,const float* ub,float* ko,float* uo,
                      const float* k1,const float* k2,const float* k3,
                      float alpha,float dt,float coef,int Nz,int Ny,int Nx,float ihx,float ihy,float ihz){
  int tx=threadIdx.x,ty=threadIdx.y,gy=blockIdx.y*BY+ty,z=blockIdx.z,gx0=(blockIdx.x*BX+tx)*CO;
  if(gy>=Ny||z>=Nz) return;
  long sy=Nx,sz=(long)Ny*Nx,row=(long)z*sz+(long)gy*sy;
  float r[CO+8];
  #pragma unroll
  for(int i=0;i<CO+8;i++){ int xx=gx0-4+i; r[i]=(xx>=0&&xx<Nx)?field[row+xx]:0.0f; }
  #pragma unroll
  for(int i=0;i<CO;i++){
    int gx=gx0+i; if(gx>=Nx) continue; long b=row+gx;
    bool in=(z>=4)&&(z<Nz-4)&&(gy>=4)&&(gy<Ny-4)&&(gx>=4)&&(gx<Nx-4);
    if(!in){ uo[b]=ub[b]; continue; }
    float uc=r[4+i];
    float xx=(C0*uc + C1*(r[5+i]+r[3+i]) + C2*(r[6+i]+r[2+i]) + C3*(r[7+i]+r[1+i]) + C4*(r[8+i]+r[0+i]))*ihx;
    float yy=(C0*uc + C1*(field[b+sy]+field[b-sy]) + C2*(field[b+2*sy]+field[b-2*sy]) + C3*(field[b+3*sy]+field[b-3*sy]) + C4*(field[b+4*sy]+field[b-4*sy]))*ihy;
    float zz=(C0*uc + C1*(field[b+sz]+field[b-sz]) + C2*(field[b+2*sz]+field[b-2*sz]) + C3*(field[b+3*sz]+field[b-3*sz]) + C4*(field[b+4*sz]+field[b-4*sz]))*ihz;
    float k=alpha*(xx+yy+zz);
    if(FINAL) uo[b]=ub[b]+(dt/6.0f)*(k1[b]+2.0f*k2[b]+2.0f*k3[b]+k);
    else { ko[b]=k; uo[b]=ub[b]+coef*dt*k; }
  }
}
torch::Tensor rk4(torch::Tensor u0,double ad,double hxd,double hyd,double hzd,int n){
  int Nz=u0.size(0),Ny=u0.size(1),Nx=u0.size(2);
  float a=ad,hx=hxd,hy=hyd,hz=hzd,ihx=1.0f/(hx*hx),ihy=1.0f/(hy*hy),ihz=1.0f/(hz*hz),dt=0.05f/(a*(ihx+ihy+ihz));
  auto u=u0.clone(),k1=torch::empty_like(u),k2=torch::empty_like(u),k3=torch::empty_like(u);
  auto us=torch::empty_like(u),us2=torch::empty_like(u),us3=torch::empty_like(u),f=torch::empty_like(u);
  dim3 blk(BX,BY),grd((Nx+BX*CO-1)/(BX*CO),(Ny+BY-1)/BY,Nz); auto st=at::cuda::getCurrentCUDAStream();
  for(int s=0;s<n;s++){
    march<false><<<grd,blk,0,st>>>(u.data_ptr<float>(),u.data_ptr<float>(),k1.data_ptr<float>(),us.data_ptr<float>(),nullptr,nullptr,nullptr,a,dt,0.5f,Nz,Ny,Nx,ihx,ihy,ihz);
    march<false><<<grd,blk,0,st>>>(us.data_ptr<float>(),u.data_ptr<float>(),k2.data_ptr<float>(),us2.data_ptr<float>(),nullptr,nullptr,nullptr,a,dt,0.5f,Nz,Ny,Nx,ihx,ihy,ihz);
    march<false><<<grd,blk,0,st>>>(us2.data_ptr<float>(),u.data_ptr<float>(),k3.data_ptr<float>(),us3.data_ptr<float>(),nullptr,nullptr,nullptr,a,dt,1.0f,Nz,Ny,Nx,ihx,ihy,ihz);
    march<true><<<grd,blk,0,st>>>(us3.data_ptr<float>(),u.data_ptr<float>(),nullptr,f.data_ptr<float>(),k1.data_ptr<float>(),k2.data_ptr<float>(),k3.data_ptr<float>(),a,dt,0.0f,Nz,Ny,Nx,ihx,ihy,ihz);
    auto t=u;u=f;f=t;
  }
  return u;
}
"""
_ext = load_inline(name="rk4_cuda_coarsen", cpp_sources=[_CPP], cuda_sources=[_CUDA], functions=["rk4"],
    extra_cuda_cflags=["-O3","--expt-relaxed-constexpr","-std=c++17","-gencode","arch=compute_90a,code=sm_90a"], verbose=False)
def custom_kernel(data: input_t) -> output_t:
    u0, alpha, hx, hy, hz, n_steps = data
    return _ext.rk4(u0.contiguous(), float(alpha), float(hx), float(hy), float(hz), n_steps)
