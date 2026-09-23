"""
The central experiment: the base paper's reflection model against physical
specular reflection, on identical geometry and identical parameters.

Produces
  Fig 2  SNR maps, conventional / cosine-model IRS / specular IRS
  Fig 4  per-model gain gap across the room, and the searchability contrast

Run:  python3 compare_models.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotting import publication_style, save_fig

from channel import (Config, build_mirrors, receiver_grid, upward_normals,
                     snr_db, cov)
from fitness import GeometryCache, analytic_normals

np.random.seed(20260922)

publication_style()
cfg = Config()
GRID = 21
mpos, mnrm = build_mirrors(cfg)

print("BASE PAPER PARAMETERS (Table 3 / Fig 1 / p7)")
print(f"  room {cfg.room}, receiver plane {cfg.rx_plane_z} m")
print(f"  B = {cfg.bandwidth/1e6:.0f} MHz, delta = {cfg.mirror_rho}, "
      f"m = {cfg.m:.2f}, Gc = {cfg.g_concentrator:.2f}")
print(f"  ambient shot noise: {'on' if cfg.ambient_default else 'OFF (as in paper)'}")
print(f"  mirrors: {len(mpos)} = {cfg.mirrors_per_panel}^2 x "
      f"{len(cfg.panel_centres)} panels, panel {cfg.panel_size} m [ASSUMED]")
print(f"  fitness: f = {cfg.w_snr}*SNR_linear - {cfg.w_cov}*CoV   (Eq. 15)")

X, Y, pts = receiver_grid(cfg, GRID)
rx_n = upward_normals(len(pts))

# ---- conventional ---------------------------------------------------------
from channel import los_gain
h_los = los_gain(cfg, pts, rx_n).sum(axis=1)
snr_conv = snr_db(cfg, h_los)

# ---- both IRS models, each at its OWN analytic optimum, per receiver -------
results = {}
for model in ("cosine", "specular"):
    h = np.empty(len(pts))
    for i, p in enumerate(pts):
        c = GeometryCache(cfg, mpos, mnrm, p[None, :], rx_n[i][None, :], model=model)
        h[i] = c.total_gain(analytic_normals(c, 0))[0]
    results[model] = dict(h=h, snr=snr_db(cfg, h))

print(f"\n{'scheme':<26}{'SNR min':>9}{'SNR max':>9}{'mean':>8}"
      f"{'CoV(Prx)':>10}{'mean Prx':>11}")
rows = [("conventional (no IRS)", snr_conv, h_los),
        ("IRS, cosine model", results["cosine"]["snr"], results["cosine"]["h"]),
        ("IRS, specular model", results["specular"]["snr"], results["specular"]["h"])]
for name, s, h in rows:
    print(f"{name:<26}{s.min():9.2f}{s.max():9.2f}{s.mean():8.2f}"
          f"{cov(h):10.3f}{1e6*cfg.p_led*h.mean():10.2f}u")

gap = results["cosine"]["snr"] - results["specular"]["snr"]
print(f"\nOVERSTATEMENT by the cosine model: "
      f"{gap.min():.2f} to {gap.max():.2f} dB  (mean {gap.mean():.2f} dB)")

print("\nBase paper reports, for comparison:")
print("  conventional  43 - 51 dB      (their Fig 3a)")
print("  IRS + blockage 50 - 84 dB     (their Fig 3b)")

# ---- searchability: hit fraction for a random configuration ---------------
print("\nSearchability of each model (random mirror orientations):")
rng = np.random.default_rng(0)
user = np.array([[3.60, 1.40, cfg.rx_plane_z]])
for model in ("cosine", "specular"):
    c = GeometryCache(cfg, mpos, mnrm, user, upward_normals(1), model=model)
    opt = c.total_gain(analytic_normals(c, 0))[0]
    vals = []
    for _ in range(200):
        n = np.random.default_rng(rng.integers(1 << 30)).normal(size=(c.K, 3))
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        n *= np.sign(np.einsum("kd,kd->k", n, c.wall_normals))[:, None]
        vals.append(c.irs_gain(n)[0])
    vals = np.array(vals)
    frac = float(np.mean(vals > 0.01 * (opt - c.h_los[0])))
    print(f"  {model:<9} random config reaches >1% of optimum IRS gain "
          f"in {100*frac:5.1f}% of draws")

# ---- figures --------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
vmin = min(snr_conv.min(), results["specular"]["snr"].min())
vmax = max(snr_conv.max(), results["cosine"]["snr"].max())
panels = [(snr_conv, "Conventional VLC (no IRS)"),
          (results["cosine"]["snr"], "IRS, base-paper cosine model"),
          (results["specular"]["snr"], "IRS, specular reflection")]
for ax, (data, title) in zip(axes, panels):
    im = ax.pcolormesh(X, Y, data.reshape(X.shape), shading="gouraud",
                       vmin=vmin, vmax=vmax, cmap="viridis")
    ax.scatter(cfg.led_positions[:, 0], cfg.led_positions[:, 1], marker="*",
               s=120, c="white", edgecolors="k", zorder=3)
    ax.scatter(cfg.panel_centres[:, 0], cfg.panel_centres[:, 1], marker="s",
               s=60, c="orangered", edgecolors="k", zorder=3)
    ax.set_title(f"{title}\n{data.min():.1f} – {data.max():.1f} dB", fontsize=10)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
fig.colorbar(im, ax=axes, label="electrical SNR [dB]", shrink=0.9)
save_fig(fig, "fig2_model_comparison")

fig2, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
pc = ax.pcolormesh(X, Y, gap.reshape(X.shape), shading="gouraud", cmap="magma")
ax.scatter(cfg.panel_centres[:, 0], cfg.panel_centres[:, 1], marker="s",
           s=60, c="cyan", edgecolors="k", zorder=3, label="IRS panel")
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
ax.set_title("SNR overstated by omitting the law of reflection", fontsize=10)
ax.legend(fontsize=8)
fig2.colorbar(pc, ax=ax, label="cosine model − specular [dB]")
save_fig(fig2, "fig4_overstatement")
print("\nwrote fig2_model_comparison.* and fig4_overstatement.*")
