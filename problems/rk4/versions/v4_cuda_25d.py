# Entry 4 — CUDA 2.5D blocking (shared-mem (x,y) tile + register queue marching z),
# with z-chunking. RECORD ARTIFACT: correct (1e-6) but SLOWER than naive CUDA.
#   non-chunked 2.5D: 150 ms;  z-chunked (ZC=32): 134 ms;  naive CUDA: 85 ms; Triton: 88 ms.
# Why it loses (ncu): the shared-load tile+halo has ~1.5-2.5x in-plane redundancy
# (halo overlap between tiles), plus 2 __syncthreads per z-step; meanwhile H100's
# 50 MB L2 already gives the naive kernel its neighbor reuse for free. Occupancy was
# 12.5% (sync/serialization-bound). Kept naive CUDA / Triton as best. See worklog E4.
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
#define BX 32
#define BY 8
#define H 4
#define ZC 32
#define SMW (BX+2*H)
#define SMH (BY+2*H)
static const float C0=-205.0/72.0,C1=8.0/5.0,C2=-1.0/5.0,C3=8.0/315.0,C4=-1.0/560.0;
template<bool FINAL>
__global__ void march(const float* fld,const float* ub,float* ko,float* uo,
                      const float* k1,const float* k2,const float* k3,
                      float a,float dt,float coef,int Nz,int Ny,int Nx,float ihx,float ihy,float ihz){
  int tx=threadIdx.x, ty=threadIdx.y, bx0=blockIdx.x*BX, by0=blockIdx.y*BY, gx=bx0+tx, gy=by0+ty;
  long sy=Nx, sz=(long)Ny*Nx; bool colv=(gx<Nx)&&(gy<Ny); long col=colv?((long)gy*sy+gx):0;
  __shared__ float sm[SMH][SMW];
  int z0=H+blockIdx.z*ZC, z1=min(z0+ZC,Nz-H); if(z0>=Nz-H) return;
  float q[9];
  #pragma unroll
  for(int i=0;i<9;i++){ int zz=z0-H+i; q[i]=(colv&&zz>=0&&zz<Nz)?fld[(long)zz*sz+col]:0.0f; }
  bool inxy=(gx>=H)&&(gx<Nx-H)&&(gy>=H)&&(gy<Ny-H);
  for(int z=z0; z<z1; z++){
    for(int idx=ty*BX+tx; idx<SMW*SMH; idx+=BX*BY){
      int sr=idx/SMW, sc=idx%SMW, lx=bx0-H+sc, ly=by0-H+sr;
      sm[sr][sc]=(lx>=0&&lx<Nx&&ly>=0&&ly<Ny)?fld[(long)z*sz+(long)ly*sy+lx]:0.0f;
    }
    __syncthreads();
    if(inxy){
      int cx=tx+H, cy=ty+H; float uc=q[4];
      float xx=(C0*uc + C1*(sm[cy][cx+1]+sm[cy][cx-1]) + C2*(sm[cy][cx+2]+sm[cy][cx-2]) + C3*(sm[cy][cx+3]+sm[cy][cx-3]) + C4*(sm[cy][cx+4]+sm[cy][cx-4]))*ihx;
      float yy=(C0*uc + C1*(sm[cy+1][cx]+sm[cy-1][cx]) + C2*(sm[cy+2][cx]+sm[cy-2][cx]) + C3*(sm[cy+3][cx]+sm[cy-3][cx]) + C4*(sm[cy+4][cx]+sm[cy-4][cx]))*ihy;
      float zz=(C0*uc + C1*(q[5]+q[3]) + C2*(q[6]+q[2]) + C3*(q[7]+q[1]) + C4*(q[8]+q[0]))*ihz;
      float k=a*(xx+yy+zz); long b=(long)z*sz+col;
      if(FINAL) uo[b]=ub[b]+(dt/6.0f)*(k1[b]+2.0f*k2[b]+2.0f*k3[b]+k);
      else { ko[b]=k; uo[b]=ub[b]+coef*dt*k; }
    }
    __syncthreads();
    #pragma unroll
    for(int i=0;i<8;i++) q[i]=q[i+1];
    int zn=z+H+1; q[8]=(colv&&zn<Nz)?fld[(long)zn*sz+col]:0.0f;
  }
}
torch::Tensor rk4(torch::Tensor u0,double ad,double hxd,double hyd,double hzd,int n){
  int Nz=u0.size(0),Ny=u0.size(1),Nx=u0.size(2);
  float a=ad,hx=hxd,hy=hyd,hz=hzd,ihx=1.0f/(hx*hx),ihy=1.0f/(hy*hy),ihz=1.0f/(hz*hz),dt=0.05f/(a*(ihx+ihy+ihz));
  auto u=u0.clone(),k1=torch::empty_like(u),k2=torch::empty_like(u),k3=torch::empty_like(u);
  auto us=u0.clone(),us2=u0.clone(),us3=u0.clone(),f=u0.clone();
  dim3 blk(BX,BY),grd((Nx+BX-1)/BX,(Ny+BY-1)/BY,(Nz-2*H+ZC-1)/ZC); auto st=at::cuda::getCurrentCUDAStream();
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
_ext = load_inline(name="rk4_cuda_25d", cpp_sources=[_CPP], cuda_sources=[_CUDA], functions=["rk4"],
    extra_cuda_cflags=["-O3","--expt-relaxed-constexpr","-std=c++17","-gencode","arch=compute_90a,code=sm_90a"], verbose=False)
def custom_kernel(data: input_t) -> output_t:
    u0, alpha, hx, hy, hz, n_steps = data
    return _ext.rk4(u0.contiguous(), float(alpha), float(hx), float(hy), float(hz), n_steps)
