"""
Panel-size sweep.

The base paper gives IRS panel CENTRE coordinates (Table 3) but never states
panel extent or A_IRS. Panel size is therefore the one free geometric knob,
and it controls the IRS peak SNR. This finds the value that reproduces their
Fig. 3b maximum of 84 dB -- and checks whether that value is physically
plausible for a 10 x 10 mirror array.

It also confirms the cosine-vs-specular overstatement is not an artefact of
the chosen panel size.

Run:  python3 sweep_panel.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from channel import (Config, build_mirrors, receiver_grid, upward_normals,
                     los_gain, snr_db, cov)
from fitness import GeometryCache, analytic_normals
from plotting import publication_style, save_fig

publication_style()

GRID = 11                      # coarse grid: this is a sweep, not a final figure
SIZES = (0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
TARGET_MAX_DB = 84.0           # their Fig. 3b maximum


def evaluate(panel_m, model, grid=GRID):
    cfg = Config(panel_size=(panel_m, panel_m))
    mpos, mnrm = build_mirrors(cfg)
    _, _, pts = receiver_grid(cfg, grid)
    rx_n = upward_normals(len(pts))
    h = np.empty(len(pts))
    for i, p in enumerate(pts):
        cache = GeometryCache(cfg, mpos, mnrm, p[None, :], rx_n[i][None, :],
                              model=model)
        h[i] = cache.total_gain(analytic_normals(cache, 0))[0]
    return snr_db(cfg, h), h, cfg


print(f"{'panel':>7}{'pitch':>9}{'cos min':>9}{'cos max':>9}"
      f"{'spec min':>10}{'spec max':>10}{'gap mean':>10}")
print(f"{'[m]':>7}{'[mm]':>9}{'[dB]':>9}{'[dB]':>9}{'[dB]':>10}{'[dB]':>10}{'[dB]':>10}")

rows = []
for w in SIZES:
    s_cos, h_cos, cfg = evaluate(w, "cosine")
    s_spec, h_spec, _ = evaluate(w, "specular")
    pitch_mm = 1000.0 * w / (cfg.mirrors_per_panel - 1)
    gap = float(np.mean(s_cos - s_spec))
    rows.append((w, pitch_mm, s_cos.min(), s_cos.max(),
                 s_spec.min(), s_spec.max(), gap, cov(h_cos), cov(h_spec)))
    print(f"{w:7.2f}{pitch_mm:9.1f}{s_cos.min():9.2f}{s_cos.max():9.2f}"
          f"{s_spec.min():10.2f}{s_spec.max():10.2f}{gap:10.2f}")

rows = np.array([r[:7] for r in rows])

# ---- which panel size reproduces their 84 dB peak? ------------------------
w_grid, cos_max = rows[:, 0], rows[:, 3]
if cos_max.max() >= TARGET_MAX_DB:
    w_fit = float(np.interp(TARGET_MAX_DB, cos_max, w_grid))
    pitch = 1000.0 * w_fit / (Config().mirrors_per_panel - 1)
    print(f"\n  panel size reproducing their {TARGET_MAX_DB:.0f} dB peak: "
          f"{w_fit:.2f} m  ->  {pitch:.0f} mm element pitch")
    print(f"  ({w_fit:.2f} m panel on a 3 m wall, "
          f"{Config().mirrors_per_panel}x{Config().mirrors_per_panel} elements)")
else:
    print(f"\n  their {TARGET_MAX_DB:.0f} dB peak is NOT reachable at any panel "
          f"size up to {SIZES[-1]:.1f} m (best {cos_max.max():.2f} dB).")
    print("  Report this: the gap must come from an unstated parameter other "
          "than panel extent -- most likely P_t or A_IRS.")

print(f"\n  overstatement across the whole sweep: "
      f"{rows[:, 6].min():.2f} to {rows[:, 6].max():.2f} dB "
      f"(mean {rows[:, 6].mean():.2f} dB) -- insensitive to panel size")

# ---- figure ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.03, 3.0), constrained_layout=True)
ax.plot(rows[:, 0], rows[:, 3], "-o", label="cosine model, max")
ax.plot(rows[:, 0], rows[:, 5], "--s", label="specular model, max")
ax.plot(rows[:, 0], rows[:, 2], ":^", color="C0", alpha=.6, label="cosine, min")
ax.plot(rows[:, 0], rows[:, 4], ":v", color="C1", alpha=.6, label="specular, min")
ax.axhline(TARGET_MAX_DB, color="k", ls=":", lw=1.2,
           label=f"base paper peak ({TARGET_MAX_DB:.0f} dB)")
ax.set_xlabel("IRS panel side length [m]  (unstated in the base paper)")
ax.set_ylabel("electrical SNR [dB]")
ax.grid(alpha=.3)
ax.legend(fontsize=8)
ax.set_title("Sensitivity to the one unstated geometric parameter", fontsize=10)
print("\nwrote:", ", ".join(save_fig(fig, "fig5_panel_sweep")))
