import numpy as np

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from channel import Config, build_mirrors, snr_db, upward_normals, _unit
from fitness import GeometryCache, analytic_normals

N = 25
SEED = 20260922


def per_elem(cache, nrm, r=0):
    """Per-element reflected gain, both models, without the internal sum."""
    if cache.model == "cosine":
        ca = np.clip(np.einsum("krd,kd->kr", cache.u_out, nrm)[:, r], 0.0, None)
        return cache.base[:, :, r].sum(axis=1) * ca
    k = np.arange(cache.K)
    best = np.argmax(cache.led_score[:, :, r], axis=1)
    u_in = cache.u_in[k, best]
    ct = np.einsum("kd,kd->k", u_in, nrm)
    refl = 2.0 * ct[:, None] * nrm - u_in
    cd = np.einsum("kd,kd->k", refl, cache.u_out[:, r, :])
    tt = (1.0 - cd) / cache.one_minus_cos_acc[:, r]
    w = np.where(tt < 1.0, 0.5 * (1 + np.cos(np.pi * np.sqrt(np.clip(tt, 0, 1)))), 0.0)
    return cache.base[k, best, r] * np.clip(ct, 0.0, None) * w


def aperture_c(cache, nrm, r=0):
    """min(1, projected element area / cone cross-section at the element)."""
    k = np.arange(cache.K)
    best = np.argmax(cache.led_score[:, :, r], axis=1)
    d1 = cache.d1[k, best]
    d2 = cache.d2[:, r]
    ci = np.clip(np.einsum("kd,kd->k", cache.u_in[k, best], nrm), 0.0, None)
    A_m = (cache.cfg.panel_size[0] * cache.cfg.panel_size[1]
           / cache.cfg.mirrors_per_panel ** 2)
    beam = cache.cfg.a_pd * (d1 / (d1 + d2)) ** 2
    return np.minimum(1.0, A_m * ci / beam)


print("Divergence in the genuine aperture-stop regime")
print("(per-element c = min(1, projected area / cone), both models)\n")
print(f"{'n/panel':>8}{'K':>9}{'median c':>11}{'divergence':>13}{'IRS/LoS':>10}")

for mpp in (10, 50, 100, 150, 250):
    cfg = Config(mirrors_per_panel=mpp)
    mpos, mnrm = build_mirrors(cfg)
    K = len(mpos)
    r = np.random.default_rng(SEED)
    lx, ly, _ = cfg.room
    pos = np.stack([r.uniform(.25, lx - .25, N), r.uniform(.25, ly - .25, N),
                    np.full(N, cfg.rx_plane_z)], -1)
    up = upward_normals(N)
    dv = np.empty(N)
    ratio = np.empty(N)
    cmed = []
    for i in range(N):
        hc = hs = h0 = 0.0
        for model in ("cosine", "specular"):
            g = GeometryCache(cfg, mpos, mnrm, pos[i][None, :], up[i][None, :],
                              model=model)
            nrm = analytic_normals(g, 0)
            ck = aperture_c(g, nrm)
            tot = float((ck * per_elem(g, nrm)).sum())
            if model == "cosine":
                hc, h0 = tot, float(g.h_los[0])
                cmed.append(np.median(ck))
            else:
                hs = tot
        a = snr_db(cfg, np.array([h0 + hc]))[0]
        b = snr_db(cfg, np.array([h0 + hs]))[0]
        dv[i] = a - b
        ratio[i] = hs / h0 if h0 > 0 else np.nan
    print(f"{mpp:8d}{K:9d}{np.median(cmed):11.4f}{dv.mean():10.2f} dB"
          f"{np.nanmedian(ratio):10.1f}")

print("\n--- divergence against reflected-to-LoS ratio, c = 1, fixed panel ---")
print(f"{'K':>8}{'IRS/LoS':>10}{'divergence':>13}")
for mpp in (3, 6, 10, 14, 20):
    cfg = Config(mirrors_per_panel=mpp)
    mpos, mnrm = build_mirrors(cfg)
    r = np.random.default_rng(SEED)
    lx, ly, _ = cfg.room
    pos = np.stack([r.uniform(.25, lx - .25, 120), r.uniform(.25, ly - .25, 120),
                    np.full(120, cfg.rx_plane_z)], -1)
    up = upward_normals(120)
    dv, rt = np.empty(120), np.empty(120)
    for i in range(120):
        gc = GeometryCache(cfg, mpos, mnrm, pos[i][None, :], up[i][None, :], model="cosine")
        gs = GeometryCache(cfg, mpos, mnrm, pos[i][None, :], up[i][None, :], model="specular")
        hc = gc.irs_gain(analytic_normals(gc, 0))[0]
        hs = gs.irs_gain(analytic_normals(gs, 0))[0]
        h0 = gs.h_los[0]
        dv[i] = snr_db(cfg, np.array([h0 + hc]))[0] - snr_db(cfg, np.array([h0 + hs]))[0]
        rt[i] = hs / h0
    print(f"{len(mpos):8d}{np.median(rt):10.1f}{dv.mean():10.2f} dB")

print("\n--- aperture margin across the panel sweep (10x10 per panel) ---")
print(f"{'panel [m]':>10}{'A_m [cm^2]':>12}{'min margin':>12}")
for w in (0.5, 1.0, 2.0, 4.0):
    cfg = Config(panel_size=(w, w))
    mpos, mnrm = build_mirrors(cfg)
    A_m = w * w / cfg.mirrors_per_panel ** 2
    r = np.random.default_rng(SEED)
    lx, ly, _ = cfg.room
    pos = np.stack([r.uniform(.25, lx - .25, 60), r.uniform(.25, ly - .25, 60),
                    np.full(60, cfg.rx_plane_z)], -1)
    up = upward_normals(60)
    mn = np.inf
    for i in range(60):
        g = GeometryCache(cfg, mpos, mnrm, pos[i][None, :], up[i][None, :], model="specular")
        nrm = analytic_normals(g, 0)
        k = np.arange(g.K)
        best = np.argmax(g.led_score[:, :, 0], axis=1)
        d1, d2 = g.d1[k, best], g.d2[:, 0]
        ci = np.clip(np.einsum("kd,kd->k", g.u_in[k, best], nrm), 0, None)
        beam = cfg.a_pd * (d1 / (d1 + d2)) ** 2
        mn = min(mn, float(np.min(A_m * ci[ci > 0] / beam[ci > 0])))
    print(f"{w:10.1f}{A_m*1e4:12.2f}{mn:11.0f}x")
