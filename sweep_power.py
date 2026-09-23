"""
Transmit-power sweep: is the cosine-vs-specular overstatement an artefact of
the one calibrated parameter?

The reference system does not state its transmit optical power. Ours is set to
P_t = 1.75 W [CALIBRATED] because that reproduces the peak of its Fig. 3a. If
the measured overstatement moved with P_t, the whole result would be an artefact
of that calibration rather than a property of the reflection model, so this
sweeps P_t over a six-fold range and reports the overstatement at each point.

The two models share every term of the link budget except the orientation
factor, and P_t is a common multiplier, so a naive reading says the dB gap must
be constant. It is not quite: SNR is not linear in received power (shot noise
scales with it, thermal noise does not), so raising P_t moves the link towards
the shot-noise limit and compresses the gap. Measured, the gap runs from
7.13 dB at 0.5 W down to 5.95 dB at 3.0 W -- present and of the same order
throughout, which is the claim that matters.

Diffuse reflections are OFF here, as in every reproduction script: this
compares against the reference system on its own stated channel. See the
docstring of freeze.py.

Run:  python3 sweep_power.py      (~5 s)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import freeze
from channel import (Config, build_mirrors, receiver_grid, upward_normals,
                     los_gain, snr_db)
from fitness import GeometryCache, analytic_normals
from plotting import publication_style, save_fig

publication_style()

GRID = 11
POWERS_W = (0.5, 0.75, 1.0, 1.75, 2.5, 3.0)      # six-fold range about 1.75 W


def evaluate(p_led, model, grid=GRID):
    """Mean, min and max SNR over the receiver plane at one transmit power."""
    cfg = Config(p_led=p_led)
    mpos, mnrm = build_mirrors(cfg)
    _, _, pts = receiver_grid(cfg, grid)
    rx_n = upward_normals(len(pts))

    h = np.empty(len(pts))
    for i, pt in enumerate(pts):
        g = GeometryCache(cfg, mpos, mnrm, pt[None, :], rx_n[i][None, :],
                          model=model)
        h[i] = g.total_gain(analytic_normals(g, 0))[0]
    s = snr_db(cfg, h)
    return s.mean(), s.min(), s.max()


print(freeze.banner("sweep_power") + "   [reproduction script: diffuse OFF]")
print("\nThe base paper does not state P_t. Ours is calibrated to its Fig. 3a.")
print("If the overstatement moved with P_t it would be an artefact of that "
      "calibration.\n")
print(f"{'P_t':>7}{'cos mean':>11}{'spec mean':>11}{'gap mean':>10}"
      f"{'gap min':>9}{'gap max':>9}")
print(f"{'[W]':>7}{'[dB]':>11}{'[dB]':>11}{'[dB]':>10}{'[dB]':>9}{'[dB]':>9}")

rows = []
for p in POWERS_W:
    cm, cmin, cmax = evaluate(p, "cosine")
    sm, smin, smax = evaluate(p, "specular")
    rows.append((p, cm, sm, cm - sm, cmin - smin, cmax - smax))
    print(f"{p:7.2f}{cm:11.2f}{sm:11.2f}{cm-sm:10.2f}"
          f"{cmin-smin:9.2f}{cmax-smax:9.2f}")

gaps = np.array([r[3] for r in rows])
print(f"\noverstatement across a {max(POWERS_W)/min(POWERS_W):.0f}-fold range "
      f"of transmit power: {gaps.min():.2f} to {gaps.max():.2f} dB "
      f"(mean {gaps.mean():.2f} dB)")
print(f"  The gap varies by {gaps.max()-gaps.min():.2f} dB across that range and")
print("  decreases monotonically with P_t: raising the transmit power moves the")
print("  link towards the shot-noise limit, where SNR grows as P_t rather than")
print("  P_t^2, so a fixed ratio of received powers maps to a smaller dB gap.")
print("  The overstatement is therefore a property of the reflection model, not")
print("  an artefact of the calibrated P_t -- it is present, and of the same")
print("  order, at every power tested.")

# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), constrained_layout=True)
P = [r[0] for r in rows]

axes[0].plot(P, [r[1] for r in rows], "-o", ms=4, label="cosine model")
axes[0].plot(P, [r[2] for r in rows], "-s", ms=4, label="specular reflection")
axes[0].axvline(1.75, color="0.6", lw=1, ls=":")
axes[0].text(1.80, axes[0].get_ylim()[0], " calibrated $P_t$",
             fontsize=7, color="0.4", va="bottom")
axes[0].set_xlabel("transmit optical power $P_t$ [W]")
axes[0].set_ylabel("mean SNR [dB]")
axes[0].set_title("Both models scale together")
axes[0].legend()

axes[1].plot(P, gaps, "-o", ms=4, color="C3", label="mean over the plane")
axes[1].fill_between(P, [r[4] for r in rows], [r[5] for r in rows],
                     color="C3", alpha=.15, label="min to max over the plane")
axes[1].axvline(1.75, color="0.6", lw=1, ls=":")
axes[1].set_ylim(0, max(r[5] for r in rows) * 1.25)
axes[1].set_xlabel("transmit optical power $P_t$ [W]")
axes[1].set_ylabel("overstatement [dB]")
axes[1].set_title("The gap persists at every power")
axes[1].legend(loc="lower right")

print("\nwrote:", ", ".join(save_fig(fig, "fig12_power_sweep")))
