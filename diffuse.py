"""
First-order diffuse reflections from room surfaces.

The base paper states on p.6 that it models shot and thermal noise only, and
that "ambient light and non-IRS multipath components could affect system
performance", arguing the IRS's directional reflections suppress both. This
module supplies the non-IRS multipath so that claim can be tested rather than
assumed.

Model: the standard two-stage Lambertian bounce (Barry et al.). Each surface is
divided into patches; a patch collects light from the LEDs, then re-radiates it
as a Lambertian source of order 1:

    h_diff = sum_j  rho_j A_j A_pd (m+1)
                    -----------------------------------------
                       2 pi^2 d1_j^2 d2_j^2
             * cos^m(phi_j) cos(theta_in_j) cos(theta_out_j) cos(psi_j)
             * T_s g(psi_j)

Only the first bounce is included. Second and higher order terms are roughly an
order of magnitude smaller in a room this size and are a known, stated
approximation -- the same one the open-source ray-tracing VLC simulators make.

Two further simplifications, both stated rather than hidden:
  * the IRS panels occupy part of the wall area and are treated here as if the
    wall were uniformly diffuse behind them, which slightly over-counts;
  * furniture neither blocks nor contributes diffuse paths.

The diffuse term does NOT depend on mirror orientation, so it adds to the LoS
term and forms part of the floor that remains when a steered configuration goes
stale. That floor is what an IRS is actually competing against.
"""

import numpy as np

from channel import Config, _unit


def build_patches(cfg: Config, n_wall=(8, 5), n_ceil=(8, 8), n_floor=(8, 8),
                  include=("walls", "ceiling", "floor")):
    """
    Surface patch centres, inward normals and areas.

    Returns pos (P, 3), nrm (P, 3), area (P,).
    """
    lx, ly, lz = cfg.room
    pos, nrm, area = [], [], []

    def grid(u0, u1, v0, v1, nu, nv):
        du, dv = (u1 - u0) / nu, (v1 - v0) / nv
        u = u0 + du * (np.arange(nu) + 0.5)
        v = v0 + dv * (np.arange(nv) + 0.5)
        U, V = np.meshgrid(u, v, indexing="xy")
        return U.ravel(), V.ravel(), du * dv

    if "walls" in include:
        nu, nv = n_wall
        for wall in ("y0", "y1", "x0", "x1"):
            if wall in ("y0", "y1"):
                U, V, a = grid(0, lx, 0, lz, nu, nv)
                y = 0.0 if wall == "y0" else ly
                p = np.stack([U, np.full(U.size, y), V], -1)
                nv_ = np.array([0., 1., 0.]) if wall == "y0" else np.array([0., -1., 0.])
            else:
                U, V, a = grid(0, ly, 0, lz, nu, nv)
                x = 0.0 if wall == "x0" else lx
                p = np.stack([np.full(U.size, x), U, V], -1)
                nv_ = np.array([1., 0., 0.]) if wall == "x0" else np.array([-1., 0., 0.])
            pos.append(p)
            nrm.append(np.tile(nv_, (len(p), 1)))
            area.append(np.full(len(p), a))

    if "ceiling" in include:
        U, V, a = grid(0, lx, 0, ly, *n_ceil)
        p = np.stack([U, V, np.full(U.size, lz)], -1)
        pos.append(p)
        nrm.append(np.tile([0., 0., -1.], (len(p), 1)))
        area.append(np.full(len(p), a))

    if "floor" in include:
        U, V, a = grid(0, lx, 0, ly, *n_floor)
        p = np.stack([U, V, np.zeros(U.size)], -1)
        pos.append(p)
        nrm.append(np.tile([0., 0., 1.], (len(p), 1)))
        area.append(np.full(len(p), a))

    return (np.concatenate(pos), np.concatenate(nrm), np.concatenate(area))


def diffuse_gain(cfg: Config, rx_pos, rx_normals, patches=None, rho=None,
                 chunk=256):
    """
    First-order diffuse channel gain at each receiver. Returns (R,).

    Chunked over patches so the (P, L, R) intermediate stays bounded.
    """
    if patches is None:
        patches = build_patches(cfg)
    ppos, pnrm, parea = patches
    rho = cfg.wall_rho if rho is None else rho

    rx = np.atleast_2d(np.asarray(rx_pos, dtype=float))
    rn = _unit(np.atleast_2d(np.asarray(rx_normals, dtype=float)))
    led = cfg.led_positions
    led_n = _unit(cfg.led_normal)
    cos_fov = np.cos(np.radians(cfg.fov_deg))

    total = np.zeros(len(rx))
    for s in range(0, len(ppos), chunk):
        P = ppos[s:s + chunk]
        N = pnrm[s:s + chunk]
        A = parea[s:s + chunk]

        # --- stage 1: LED -> patch ---
        v1 = led[None, :, :] - P[:, None, :]              # (p, L, 3)
        d1 = np.linalg.norm(v1, axis=-1)
        u1 = v1 / d1[..., None]
        cos_phi = np.clip(np.einsum("pld,d->pl", -u1, led_n), 0, None)
        cos_in = np.clip(np.einsum("pld,pd->pl", u1, N), 0, None)
        stage1 = ((cfg.m + 1) / (2 * np.pi * d1 ** 2)
                  * np.power(cos_phi, cfg.m) * cos_in * A[:, None])

        # --- stage 2: patch -> PD (patch is Lambertian order 1) ---
        v2 = rx[None, :, :] - P[:, None, :]               # (p, R, 3)
        d2 = np.linalg.norm(v2, axis=-1)
        u2 = v2 / d2[..., None]
        cos_out = np.clip(np.einsum("prd,pd->pr", u2, N), 0, None)
        cos_psi = np.einsum("prd,rd->pr", -u2, rn)
        cos_psi = np.where(cos_psi >= cos_fov, np.clip(cos_psi, 0, None), 0.0)
        stage2 = (rho * cfg.a_pd / (np.pi * d2 ** 2)
                  * cos_out * cos_psi * cfg.t_filter * cfg.g_concentrator)

        total += np.einsum("pl,pr->r", stage1, stage2)
    return total


def ambient_current(cfg: Config, p_ambient_w):
    """
    Background photocurrent for a given ambient optical power at the PD.

    No attempt is made here to convert lux to watts: that conversion depends on
    the source spectrum and on the receiver's optical bandpass filter, neither
    of which the base paper states. Their p.6 aside asserts 800 lux gives
    0.12 uW at the PD; that is taken as their number and converted, not
    re-derived.
    """
    return cfg.responsivity * p_ambient_w


AMBIENT_CASES = {
    # label: background photocurrent I_bg [A]
    "none (paper's baseline)": 0.0,
    "their p.6 aside, 0.12 uW": None,          # filled from ambient_current()
    "moderate indoor, 740 uA": 740e-6,
    "bright / skylight, 5100 uA": 5100e-6,
}


def ambient_cases(cfg: Config):
    """Resolve AMBIENT_CASES into concrete I_bg values."""
    out = dict(AMBIENT_CASES)
    out["their p.6 aside, 0.12 uW"] = ambient_current(cfg, 0.12e-6)
    return out
