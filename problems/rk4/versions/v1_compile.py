# Entry 1 — torch.compile (Inductor fuses the sliced stencil + RK4).
# Functional per-step rewrite (no in-place/copy_/swap, so it compiles cleanly),
# math identical to reference.ref_kernel. Watch: 1e-6 tolerance vs Inductor's
# fused fp32 codegen.
import torch
from task import input_t, output_t

_c0 = -205.0 / 72.0
_c1 = 8.0 / 5.0
_c2 = -1.0 / 5.0
_c3 = 8.0 / 315.0
_c4 = -1.0 / 560.0
_CFL = 0.05
_r = 4


def _lap(u, inv_hx2, inv_hy2, inv_hz2):
    r = _r
    zc = slice(r, -r); yc = slice(r, -r); xc = slice(r, -r)
    uc = u[zc, yc, xc]
    u_xx = (
        _c0 * uc
        + _c1 * (u[zc, yc, r + 1:-r + 1] + u[zc, yc, r - 1:-r - 1])
        + _c2 * (u[zc, yc, r + 2:-r + 2] + u[zc, yc, r - 2:-r - 2])
        + _c3 * (u[zc, yc, r + 3:-r + 3] + u[zc, yc, r - 3:-r - 3])
        + _c4 * (u[zc, yc, r + 4:] + u[zc, yc, :-r - 4])
    ) * inv_hx2
    u_yy = (
        _c0 * uc
        + _c1 * (u[zc, r + 1:-r + 1, xc] + u[zc, r - 1:-r - 1, xc])
        + _c2 * (u[zc, r + 2:-r + 2, xc] + u[zc, r - 2:-r - 2, xc])
        + _c3 * (u[zc, r + 3:-r + 3, xc] + u[zc, r - 3:-r - 3, xc])
        + _c4 * (u[zc, r + 4:, xc] + u[zc, :-r - 4, xc])
    ) * inv_hy2
    u_zz = (
        _c0 * uc
        + _c1 * (u[r + 1:-r + 1, yc, xc] + u[r - 1:-r - 1, yc, xc])
        + _c2 * (u[r + 2:-r + 2, yc, xc] + u[r - 2:-r - 2, yc, xc])
        + _c3 * (u[r + 3:-r + 3, yc, xc] + u[r - 3:-r - 3, yc, xc])
        + _c4 * (u[r + 4:, yc, xc] + u[:-r - 4, yc, xc])
    ) * inv_hz2
    return u_xx + u_yy + u_zz


def _rk4_step(u, alpha, dt, inv_hx2, inv_hy2, inv_hz2):
    r = _r
    zc = slice(r, -r); yc = slice(r, -r); xc = slice(r, -r)
    uc = u[zc, yc, xc]

    k1 = alpha * _lap(u, inv_hx2, inv_hy2, inv_hz2)
    us = u.clone(); us[zc, yc, xc] = uc + 0.5 * dt * k1

    k2 = alpha * _lap(us, inv_hx2, inv_hy2, inv_hz2)
    us = u.clone(); us[zc, yc, xc] = uc + 0.5 * dt * k2

    k3 = alpha * _lap(us, inv_hx2, inv_hy2, inv_hz2)
    us = u.clone(); us[zc, yc, xc] = uc + dt * k3

    k4 = alpha * _lap(us, inv_hx2, inv_hy2, inv_hz2)

    f = u.clone()
    f[zc, yc, xc] = uc + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return f


_compiled = None


def custom_kernel(data: input_t) -> output_t:
    global _compiled
    if _compiled is None:
        _compiled = torch.compile(_rk4_step)
    u0, alpha, hx, hy, hz, n_steps = data
    device, dtype = u0.device, u0.dtype
    alpha = torch.as_tensor(alpha, device=device, dtype=dtype)
    hx = torch.as_tensor(hx, device=device, dtype=dtype)
    hy = torch.as_tensor(hy, device=device, dtype=dtype)
    hz = torch.as_tensor(hz, device=device, dtype=dtype)
    inv_hx2 = 1.0 / (hx * hx)
    inv_hy2 = 1.0 / (hy * hy)
    inv_hz2 = 1.0 / (hz * hz)
    S = inv_hx2 + inv_hy2 + inv_hz2
    dt = _CFL / (alpha * S)
    u = u0.clone()
    for _ in range(n_steps):
        u = _compiled(u, alpha, dt, inv_hx2, inv_hy2, inv_hz2)
    return u
