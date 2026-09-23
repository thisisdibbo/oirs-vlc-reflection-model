"""
Day 5: mobility traces and the stale-configuration penalty, multi-seed.

  Delta_stale(tau) = SNR(H(t), Omega*(H(t))) - SNR(H(t), Omega*(H(t - tau)))

Single-trace results vary substantially between walk realisations, so every
number here is a mean +/- standard deviation over N_SEEDS independent traces.
Report the spread in the paper; a single trace is not a result.

Run:  python3 day5_mobility.py      (~2 min)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from channel import Config, build_mirrors
from fitness import pointing_footprint
from mobility import make_trace
from plotting import publication_style, save_fig
from trace_eval import TraceEvaluator

import freeze
SEED0 = freeze.SEED0
N_SEEDS = freeze.N_SEEDS
DURATION = 20.0                  # s per trace
DT = 0.02                        # s
SPEEDS = (0.5, 1.0, 1.5)
TAUS_MS = (20, 50, 100, 200, 500, 1000)
ACTIVITY = "walking"
ACCEPTS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0)   # deg, robustness sweep

publication_style()
cfg = Config()
DIFF = freeze.diffuse_kw(cfg)
build_diff = freeze.diffuse_kw
mpos, mnrm = build_mirrors(cfg)
lags = [int(round(ms / 1000.0 / DT)) for ms in TAUS_MS]

print(freeze.banner("day5_mobility"))
print(f"{N_SEEDS} seeds x {DURATION:.0f} s at dt = {DT*1000:.0f} ms, "
      f"activity = {ACTIVITY}")
print(f"tau_c = 0.5 s [modelling choice]; accept_floor = 0.5 deg "
      f"(footprint {100*pointing_footprint(cfg, 3.0, 0.5):.1f} cm at 3 m)")
print(f"Base paper's stated mirror response: "
      f"{cfg.mirror_response_ms[0]:.0f}-{cfg.mirror_response_ms[1]:.0f} ms "
      f"(their p.8)\n")


def penalties(model, speed, seed, accept=0.5):
    """Penalty [dB] at each tau for one trace."""
    tr = make_trace(cfg, DURATION, DT, speed, ACTIVITY,
                    np.random.default_rng(seed))
    ev = TraceEvaluator(cfg, mpos, mnrm, tr, model=model,
                        accept_floor_deg=accept, **DIFF)
    fresh = ev.snr_db(0)
    return [fresh[l:].mean() - ev.snr_db(l).mean() for l in lags]


# ---- main sweep -----------------------------------------------------------
stats = {}
for model in ("cosine", "specular"):
    for speed in SPEEDS:
        runs = np.array([penalties(model, speed, SEED0 + s)
                         for s in range(N_SEEDS)])
        stats[(model, speed)] = (runs.mean(axis=0), runs.std(axis=0))
    print(f"  done {model}")

print(f"\nStale-configuration penalty [dB], mean +/- std over {N_SEEDS} seeds")
hdr = "".join(f"{ms:>14}" for ms in TAUS_MS)
print(f"{'model':<10}{'speed':>6}{hdr}")
print(f"{'':<10}{'[m/s]':>6}" + "".join(f"{'ms':>14}" for _ in TAUS_MS))
for model in ("cosine", "specular"):
    for speed in SPEEDS:
        mu, sd = stats[(model, speed)]
        print(f"{model:<10}{speed:6.1f}"
              + "".join(f"{m:8.2f}±{s:<5.2f}" for m, s in zip(mu, sd)))

# ---- headline number ------------------------------------------------------
i50 = TAUS_MS.index(50)
print(f"\nAt tau = 50 ms (the upper end of the base paper's own actuator spec):")
for model in ("cosine", "specular"):
    for speed in SPEEDS:
        mu, sd = stats[(model, speed)]
        print(f"  {model:<9}{speed:4.1f} m/s -> {mu[i50]:6.2f} ± {sd[i50]:.2f} dB")

# ---- robustness to the acceptance assumption ------------------------------
print(f"\nRobustness: penalty [dB] vs assumed mirror acceptance, "
      f"specular, 1.0 m/s, {freeze.N_SEEDS_SUB} seeds")
print(f"{'accept':>7}{'footprint':>11}" + "".join(f"{ms:>13}" for ms in TAUS_MS))
acc_stats = {}
for acc in ACCEPTS:
    runs = np.array([penalties("specular", 1.0, SEED0 + s, accept=acc)
                     for s in range(freeze.N_SEEDS_SUB)])
    acc_stats[acc] = (runs.mean(axis=0), runs.std(axis=0))
    fp = 100 * pointing_footprint(cfg, 3.0, acc)
    print(f"{acc:7.2f}{fp:9.1f}cm"
          + "".join(f"{m:7.2f}±{s:<5.2f}" for m, s in zip(*acc_stats[acc])))

cos_ref = stats[("cosine", 1.0)][0]
print(f"\n  cosine model, same conditions:      "
      + "".join(f"{m:7.2f}{'':6}" for m in cos_ref))
print("\n  The exact dB at short latency depends on the assumed acceptance,")
print("  which the base paper does not state. The qualitative result does not:")
print("  even at 5 deg (a 52 cm footprint, wider than a person) the specular")
print("  penalty is orders of magnitude above the cosine model's, which stays")
print("  near zero at every latency tested.")

# ---- figure ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.2), constrained_layout=True)

# (a) one example time series
tr = make_trace(cfg, DURATION, DT, 1.0, ACTIVITY, np.random.default_rng(SEED0))
ev = TraceEvaluator(cfg, mpos, mnrm, tr, model="specular", **DIFF)
axes[0].plot(tr.t, ev.snr_db(0), lw=1.2, label=r"$\tau$ = 0 (ideal)")
for ms, c in ((50, "C1"), (500, "C3")):
    l = int(round(ms / 1000.0 / DT))
    axes[0].plot(tr.t[l:], ev.snr_db(l), lw=1.0, color=c, alpha=.85,
                 label=rf"$\tau$ = {ms} ms")
axes[0].set_xlabel("time [s]")
axes[0].set_ylabel("electrical SNR [dB]")
axes[0].set_title("Specular, 1.0 m/s, one trace")
axes[0].grid(alpha=.3)
axes[0].legend(loc="lower right", handlelength=1.4, borderpad=0.3,
                labelspacing=0.25)
# A tilted PD can drop every ray outside the FoV, giving zero power and a
# -300 dB sentinel. Those are real total outages, but plotting them crushes
# the axis, so clip and mark their rate instead.
lo = np.floor(min(ev.snr_db(0).min(), 40.0))
axes[0].set_ylim(max(lo, 38), None)
dead = float(np.mean(ev.snr_db(0) <= 0.0))
if dead:
    axes[0].annotate(f"{100*dead:.1f}% of samples: total outage\n"
                     f"(all rays outside FoV, clipped)",
                     xy=(0.03, 0.04), xycoords="axes fraction",
                     fontsize=5.5, color="0.35")

# (b) staleness, mean +/- std
marks = {0.5: "o", 1.0: "s", 1.5: "^"}
for model in ("cosine", "specular"):
    col = "C0" if model == "cosine" else "C3"
    ls = "--" if model == "cosine" else "-"
    for speed in SPEEDS:
        mu, sd = stats[(model, speed)]
        axes[1].plot(TAUS_MS, mu, ls, marker=marks[speed], ms=3, color=col,
                     label=f"{model}, {speed} m/s" if speed == 1.0 else None)
        axes[1].fill_between(TAUS_MS, mu - sd, mu + sd, color=col, alpha=.12)
axes[1].axvspan(*cfg.mirror_response_ms, color="grey", alpha=.18)
axes[1].annotate("actuator\nspec", xy=(11, 8.0), fontsize=5.5, color="0.35")
axes[1].set_xscale("log")
axes[1].set_xlabel(r"control latency $\tau$ [ms]")
axes[1].set_ylabel(r"$\Delta_{\rm stale}$ [dB]")
axes[1].set_title(f"Mean $\\pm$ 1 s.d., {N_SEEDS} traces")
axes[1].grid(alpha=.3, which="both")
axes[1].legend(loc="upper left", handlelength=1.6, borderpad=0.3,
               labelspacing=0.25)
axes[1].text(0.97, 0.30, "markers: 0.5 / 1.0 / 1.5 m/s", fontsize=5.5,
             color="0.4", ha="right", transform=axes[1].transAxes)

# (c) robustness to acceptance
for acc in ACCEPTS:
    mu, _ = acc_stats[acc]
    axes[2].plot(TAUS_MS, mu, "-o", ms=2.5, label=f"{acc:g}°")
axes[2].plot(TAUS_MS, cos_ref, "k--", lw=1.2, label="cosine")
axes[2].axvspan(*cfg.mirror_response_ms, color="grey", alpha=.18)
axes[2].set_xscale("log")
axes[2].set_xlabel(r"control latency $\tau$ [ms]")
axes[2].set_ylabel(r"$\Delta_{\rm stale}$ [dB]")
axes[2].set_title("Sensitivity to mirror acceptance")
axes[2].grid(alpha=.3, which="both")
axes[2].legend(loc="lower right", ncol=2, title=r"$\alpha_{\min}$",
               title_fontsize=6, handlelength=1.2, borderpad=0.3,
               labelspacing=0.2, columnspacing=0.8)

print("\nwrote:", ", ".join(save_fig(fig, "fig7_staleness")))
