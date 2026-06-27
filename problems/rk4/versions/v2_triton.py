# Entry 2 — Hand-written Triton fused stencil + RK4.
# RK4 as 4 kernels/step (3 stage + 1 final); each computes the 25-tap Laplacian +
# the stage combine + boundary-copy in ONE pass (vs Inductor's separate temporaries).
# Reuse comes from L2 cache on the shifted loads; best tile BX=128 (wide contiguous x),
# BY=4, num_warps=4. 87.9 ms (16x over baseline, 2.4x over torch.compile). Passes 1e-6.
import torch
import triton
import triton.language as tl
from task import input_t, output_t

C0 = tl.constexpr(-205.0 / 72.0)
C1 = tl.constexpr(8.0 / 5.0)
C2 = tl.constexpr(-1.0 / 5.0)
C3 = tl.constexpr(8.0 / 315.0)
C4 = tl.constexpr(-1.0 / 560.0)
R = tl.constexpr(4)
_BX, _BY, _NW = 128, 4, 4


@triton.jit
def _lap(field, base, m, sz, sy, ihx, ihy, ihz):
    uc = tl.load(field + base, mask=m, other=0.0)
    uxx = (C0 * uc
        + C1 * (tl.load(field+base+1, mask=m, other=0.0) + tl.load(field+base-1, mask=m, other=0.0))
        + C2 * (tl.load(field+base+2, mask=m, other=0.0) + tl.load(field+base-2, mask=m, other=0.0))
        + C3 * (tl.load(field+base+3, mask=m, other=0.0) + tl.load(field+base-3, mask=m, other=0.0))
        + C4 * (tl.load(field+base+4, mask=m, other=0.0) + tl.load(field+base-4, mask=m, other=0.0))) * ihx
    uyy = (C0 * uc
        + C1 * (tl.load(field+base+sy, mask=m, other=0.0) + tl.load(field+base-sy, mask=m, other=0.0))
        + C2 * (tl.load(field+base+2*sy, mask=m, other=0.0) + tl.load(field+base-2*sy, mask=m, other=0.0))
        + C3 * (tl.load(field+base+3*sy, mask=m, other=0.0) + tl.load(field+base-3*sy, mask=m, other=0.0))
        + C4 * (tl.load(field+base+4*sy, mask=m, other=0.0) + tl.load(field+base-4*sy, mask=m, other=0.0))) * ihy
    uzz = (C0 * uc
        + C1 * (tl.load(field+base+sz, mask=m, other=0.0) + tl.load(field+base-sz, mask=m, other=0.0))
        + C2 * (tl.load(field+base+2*sz, mask=m, other=0.0) + tl.load(field+base-2*sz, mask=m, other=0.0))
        + C3 * (tl.load(field+base+3*sz, mask=m, other=0.0) + tl.load(field+base-3*sz, mask=m, other=0.0))
        + C4 * (tl.load(field+base+4*sz, mask=m, other=0.0) + tl.load(field+base-4*sz, mask=m, other=0.0))) * ihz
    return uxx + uyy + uzz


@triton.jit
def _stage(field, ubase, kout, usout, alpha, dt, coef, Nz, Ny, Nx,
           ihx, ihy, ihz, BY: tl.constexpr, BX: tl.constexpr):
    z = tl.program_id(0)
    oy = tl.program_id(1) * BY + tl.arange(0, BY)
    ox = tl.program_id(2) * BX + tl.arange(0, BX)
    sz = Ny * Nx; sy = Nx
    base = z * sz + oy[:, None] * sy + ox[None, :]
    valid = (oy[:, None] < Ny) & (ox[None, :] < Nx)
    interior = (z >= R) & (z < Nz-R) & (oy[:, None] >= R) & (oy[:, None] < Ny-R) & (ox[None, :] >= R) & (ox[None, :] < Nx-R)
    k = alpha * _lap(field, base, interior, sz, sy, ihx, ihy, ihz)
    tl.store(kout + base, k, mask=interior)
    ub = tl.load(ubase + base, mask=valid, other=0.0)
    tl.store(usout + base, tl.where(interior, ub + coef * dt * k, ub), mask=valid)


@triton.jit
def _final(field, ubase, k1, k2, k3, fout, alpha, dt, Nz, Ny, Nx,
           ihx, ihy, ihz, BY: tl.constexpr, BX: tl.constexpr):
    z = tl.program_id(0)
    oy = tl.program_id(1) * BY + tl.arange(0, BY)
    ox = tl.program_id(2) * BX + tl.arange(0, BX)
    sz = Ny * Nx; sy = Nx
    base = z * sz + oy[:, None] * sy + ox[None, :]
    valid = (oy[:, None] < Ny) & (ox[None, :] < Nx)
    interior = (z >= R) & (z < Nz-R) & (oy[:, None] >= R) & (oy[:, None] < Ny-R) & (ox[None, :] >= R) & (ox[None, :] < Nx-R)
    k4 = alpha * _lap(field, base, interior, sz, sy, ihx, ihy, ihz)
    a1 = tl.load(k1 + base, mask=interior, other=0.0)
    a2 = tl.load(k2 + base, mask=interior, other=0.0)
    a3 = tl.load(k3 + base, mask=interior, other=0.0)
    ub = tl.load(ubase + base, mask=valid, other=0.0)
    f = tl.where(interior, ub + (dt / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + k4), ub)
    tl.store(fout + base, f, mask=valid)


def custom_kernel(data: input_t) -> output_t:
    u0, alpha, hx, hy, hz, n_steps = data
    Nz, Ny, Nx = u0.shape
    af = float(alpha); ihx = 1.0/(float(hx)**2); ihy = 1.0/(float(hy)**2); ihz = 1.0/(float(hz)**2)
    dt = 0.05 / (af * (ihx + ihy + ihz))
    u = u0.clone()
    k1 = torch.empty_like(u); k2 = torch.empty_like(u); k3 = torch.empty_like(u)
    us = torch.empty_like(u); us2 = torch.empty_like(u); us3 = torch.empty_like(u); f = torch.empty_like(u)
    grid = (Nz, triton.cdiv(Ny, _BY), triton.cdiv(Nx, _BX))
    for _ in range(n_steps):
        _stage[grid](u,   u, k1, us,  af, dt, 0.5, Nz, Ny, Nx, ihx, ihy, ihz, BY=_BY, BX=_BX, num_warps=_NW)
        _stage[grid](us,  u, k2, us2, af, dt, 0.5, Nz, Ny, Nx, ihx, ihy, ihz, BY=_BY, BX=_BX, num_warps=_NW)
        _stage[grid](us2, u, k3, us3, af, dt, 1.0, Nz, Ny, Nx, ihx, ihy, ihz, BY=_BY, BX=_BX, num_warps=_NW)
        _final[grid](us3, u, k1, k2, k3, f, af, dt, Nz, Ny, Nx, ihx, ihy, ihz, BY=_BY, BX=_BX, num_warps=_NW)
        u, f = f, u
    return u
