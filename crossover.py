import numpy as np

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from channel import Config, build_mirrors, snr_db, upward_normals
from fitness import GeometryCache, analytic_normals

N = 400
cfg = Config()
mpos, mnrm = build_mirrors(cfg)
r = np.random.default_rng(20260922)
lx, ly, _ = cfg.room
pos = np.stack([r.uniform(0.25, lx - .25, N), r.uniform(0.25, ly - .25, N),
                np.full(N, cfg.rx_plane_z)], -1)
up = upward_normals(N)

h0 = np.empty(N); gf = np.empty(N); gs = np.empty(N)
for i in range(N):
    real = GeometryCache(cfg, mpos, mnrm, pos[i][None, :], up[i][None, :], model="specular")
    a = r.uniform(0, 2 * np.pi)
    pst = pos[i].copy(); pst[0] += 0.05 * np.cos(a); pst[1] += 0.05 * np.sin(a)
    plan = GeometryCache(cfg, mpos, mnrm, pst[None, :], up[i][None, :], model="specular")
    h0[i] = real.h_los[0]
    gf[i] = real.irs_gain(analytic_normals(real, 0))[0]
    gs[i] = real.irs_gain(analytic_normals(plan, 0))[0]


def m(x):
    s = snr_db(cfg, x)
    return s[s > 0].mean()


base = m(h0)
rows = []
for c in np.geomspace(0.02, 4.0, 40):
    full = m(h0 + c * gf)
    lat = full - m(h0 + c * gs)
    blk = full - m(c * gf)
    rows.append((c, full - base, lat, blk, lat - blk))

print(f"{'c':>8}{'IRS gain':>11}{'latency':>10}{'blockage':>11}{'lat-blk':>10}")
for c, g, l, b, d in rows[::4]:
    print(f"{c:8.4f}{g:8.2f} dB{l:7.2f} dB{b:8.2f} dB{d:7.2f} dB")

s = [r for r in rows if r[4] >= 0]
if s:
    kx = min(s, key=lambda r: r[4])
    print(f"\ncrossover (latency = blockage) at c = {kx[0]:.3f}, "
          f"where the IRS gain over the unaided link is {kx[1]:.1f} dB")
print(f"\npublished point c = 1: IRS gain {rows[-1][1]:.2f} dB is off-scale; "
      f"at c=1 exactly:")
full = m(h0 + gf)
print(f"  IRS gain {full - base:.2f} dB, latency {full - m(h0 + gs):.2f} dB, "
      f"blockage {full - m(gf):.2f} dB")
