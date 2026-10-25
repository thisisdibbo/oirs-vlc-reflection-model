import numpy as np

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from channel import Config, build_mirrors, snr_db, upward_normals, _unit
from fitness import GeometryCache, analytic_normals

cfg = Config()
mpos, mnrm = build_mirrors(cfg)
K = len(mpos)
n = cfg.mirrors_per_panel
w, h = cfg.panel_size
A_panel = w * h
A_m = A_panel / (n * n)
print(f"panel {w} x {h} m, {n}x{n} elements per panel, {K} elements total")
print(f"per-element area A_m = {A_m:.5f} m^2 = {A_m*1e4:.2f} cm^2 "
      f"(pitch {1000*w/(n-1):.1f} mm)")
print(f"detector area A_PD   = {cfg.a_pd:.5f} m^2 = {cfg.a_pd*1e4:.2f} cm^2\n")

# A flat mirror images the LED. The detector sees that virtual source at
# d1 + d2. The cone that reaches the detector has cross-section
# A_PD * (d1/(d1+d2))^2 where it crosses the mirror plane. The element passes
# all of it only if its projected area exceeds that; otherwise the element is
# the aperture stop and the collected power is reduced by the ratio.
r = np.random.default_rng(20260922)
lx, ly, _ = cfg.room
N = 400
pos = np.stack([r.uniform(.25, lx - .25, N), r.uniform(.25, ly - .25, N),
                np.full(N, cfg.rx_plane_z)], -1)
up = upward_normals(N)

ratios, cos_i = [], []
for i in range(N):
    g = GeometryCache(cfg, mpos, mnrm, pos[i][None, :], up[i][None, :],
                      model="specular")
    best = np.argmax(g.led_score[:, :, 0], axis=1)
    k = np.arange(K)
    d1 = g.d1[k, best]
    d2 = g.d2[:, 0]
    u_in = g.u_in[k, best]
    nh = _unit(u_in + g.u_out[:, 0, :])
    ci = np.einsum("kd,kd->k", u_in, nh)
    beam = cfg.a_pd * (d1 / (d1 + d2)) ** 2
    ratios.append(A_m * np.clip(ci, 0, None) / beam)
    cos_i.append(ci)

ratios = np.concatenate(ratios)
print("ratio  (projected element area) / (beam cross-section at the element)")
print(f"  median {np.median(ratios):.4g}   mean {ratios.mean():.4g}")
print(f"  5th pct {np.percentile(ratios,5):.4g}   "
      f"95th pct {np.percentile(ratios,95):.4g}")
print(f"  fraction >= 1 (element not the stop): "
      f"{100*np.mean(ratios>=1):.1f} %\n")

c_phys = np.minimum(1.0, ratios)
print(f"physical per-element c = min(1, ratio):")
print(f"  median {np.median(c_phys):.4g}   mean {c_phys.mean():.4g}")
print(f"\nThe model as written uses c = 1 for every element.")
print(f"Overstatement factor at the median: {1/np.median(c_phys):.3g}x "
      f"({10*np.log10(1/np.median(c_phys)):.1f} dB of reflected power)")

# what surface gain does the physical c give?
h0 = np.empty(N); gf = np.empty(N)
for i in range(N):
    g = GeometryCache(cfg, mpos, mnrm, pos[i][None, :], up[i][None, :],
                      model="specular")
    h0[i] = g.h_los[0]
    gf[i] = g.irs_gain(analytic_normals(g, 0))[0]


def m(x):
    s = snr_db(cfg, x)
    return s[s > 0].mean()


base = m(h0)
print(f"\n{'c':>12}{'surface gain':>15}")
for lab, c in (("1 (as published)", 1.0),
               ("36/K (previous anchor)", 36.0 / K),
               ("physical median", float(np.median(c_phys))),
               ("physical mean", float(c_phys.mean()))):
    print(f"{lab:>24}  {m(h0 + c * gf) - base:7.2f} dB   (c = {c:.4g})")
