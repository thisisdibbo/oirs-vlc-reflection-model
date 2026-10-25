import numpy as np

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from channel import Config, build_mirrors, snr_db, upward_normals
from fitness import GeometryCache, analytic_normals

N = 150
SEED = 20260922

print("Divergence in SNR dB against the gain divergence Delta_H.")
print("Delta_H = 20 log10(H_cos / H_spec), free of the noise regime.")
print("(The optical power ratio is 10 log10 of the same quantity, half of this.)")
print("Diagnostic settings: uniform positions, face-up detector, no diffuse.\n")
print(f"{'K':>6}{'SNR div':>10}{'Delta_H':>11}{'ratio':>8}{'IRS/LoS':>10}")

for mpp in (3, 6, 10, 14, 20):
    cfg = Config(mirrors_per_panel=mpp)
    mpos, mnrm = build_mirrors(cfg)
    r = np.random.default_rng(SEED)
    lx, ly, _ = cfg.room
    pos = np.stack([r.uniform(.25, lx - .25, N), r.uniform(.25, ly - .25, N),
                    np.full(N, cfg.rx_plane_z)], -1)
    up = upward_normals(N)
    ds, dp, rt = np.empty(N), np.empty(N), np.empty(N)
    for i in range(N):
        gc = GeometryCache(cfg, mpos, mnrm, pos[i][None, :], up[i][None, :],
                           model="cosine")
        gs = GeometryCache(cfg, mpos, mnrm, pos[i][None, :], up[i][None, :],
                           model="specular")
        h0 = float(gs.h_los[0])
        hc = h0 + float(gc.irs_gain(analytic_normals(gc, 0))[0])
        hs = h0 + float(gs.irs_gain(analytic_normals(gs, 0))[0])
        ds[i] = snr_db(cfg, np.array([hc]))[0] - snr_db(cfg, np.array([hs]))[0]
        dp[i] = 20.0 * np.log10(hc / hs)
        rt[i] = (hs - h0) / h0
    print(f"{len(mpos):6d}{ds.mean():8.2f} dB{dp.mean():9.2f} dB"
          f"{ds.mean()/dp.mean():8.2f}{np.median(rt):10.1f}")

print("\nSNR ~ H^2 when thermal noise dominates, so the SNR divergence")
print("equals Delta_H there; SNR ~ H when shot noise dominates, so it")
print("falls to Delta_H / 2. The ratio column is therefore 1.0 at the")
print("thermal limit and 0.5 at the shot limit. Delta_H rising while the")
print("SNR divergence falls is the link moving between the two.")
