"""
Day 9: bit error rate, and the element-count sweep.

Two figures the optical-communications reader expects:

  (a) BER against transmit optical power for each control scheme, so the
      latency result is expressed in the currency the field actually uses.
  (b) BER and SNR against the number of mirror elements, which parallels
      Fig. 6 of the reference system directly and shows where the two
      reflection models diverge as the array grows.

OOK on an IM/DD link: P_b = Q(sqrt(SNR)).

Run:  python3 day9_ber.py      (~3 min)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from channel import Config, ber_ook, build_mirrors, snr_linear
from mobility import make_trace
from plotting import publication_style, save_fig
from steering import predict_pose, predictive_configs
from trace_eval import TraceEvaluator

import freeze
SEED0 = freeze.SEED0
N_SEEDS = freeze.N_SEEDS
DURATION = 12.0
DT = 0.01
SPEED = 1.0
SLEW_MS = 10
TAU_META_S = freeze.TAU_META_S   # measured by validate_optimizers.py, pinned in freeze.py
SIGMA = 0.05                     # realistic tracking error [m]
FEC_HD = 3.8e-3                  # 7% hard-decision FEC threshold
FEC_SD = 2.0e-2                  # 20% soft-decision FEC threshold

publication_style()
cfg = Config()
DIFF = freeze.diffuse_kw(cfg)
build_diff = freeze.diffuse_kw

# ==========================================================================
# (a) BER vs transmit power
# ==========================================================================
POWERS_W = np.logspace(np.log10(2e-3), np.log10(2.0), 22)

print(freeze.banner("day9_ber"))
print(f"(a) BER vs transmit power, {N_SEEDS} seeds, specular model, "
      f"slew {SLEW_MS} ms, sigma {100*SIGMA:.0f} cm")

mpos, mnrm = build_mirrors(cfg)
slew = int(round(SLEW_MS / 1000.0 / DT))
meta = int(round((TAU_META_S + SLEW_MS / 1000.0) / DT))

gains = {k: [] for k in ("no IRS", "metaheuristic", "reactive closed form",
                         "predictive", "genie")}
for s in range(N_SEEDS):
    tr = make_trace(cfg, DURATION, DT, SPEED, "walking",
                    np.random.default_rng(SEED0 + s))
    ev = TraceEvaluator(cfg, mpos, mnrm, tr, model="specular", **DIFF)
    best = np.argmax(ev.base.mean(axis=2), axis=1)
    u_in_sel = ev.u_in[np.arange(ev.K), best]

    pred = predict_pose(tr.pos, tr.vel, SLEW_MS / 1000.0, cfg.room,
                        rng=np.random.default_rng(SEED0 + s + 77),
                        sigma_pos=SIGMA, sigma_vel=SIGMA)
    Np = predictive_configs(mpos, u_in_sel, pred)
    Np = np.concatenate([Np[:, :1, :].repeat(slew, axis=1),
                         Np[:, :-slew, :]], axis=1)

    m = min(meta, ev.T - 2)
    # Every scheme must be averaged over the SAME timesteps. Each carries a
    # different lag, so all are sliced to the largest one (the metaheuristic's)
    # -- otherwise the comparison mixes different samples of the walk, which is
    # what produced the earlier artefact of the IRS appearing slightly worse
    # than no IRS at all.
    w = slice(m, ev.T)
    gains["no IRS"].append(ev.h_los[w].mean())
    gains["genie"].append(ev.gains(0)[w].mean())
    gains["metaheuristic"].append(ev.gains(m).mean())
    gains["reactive closed form"].append(ev.gains(slew)[m - slew:].mean())
    gains["predictive"].append(ev.gains(0, N=Np)[w].mean())

H = {k: float(np.mean(v)) for k, v in gains.items()}
print(f"\n{'scheme':<24}{'mean H':>12}{'P_t for BER = 3.8e-3':>24}")
ber_curves = {}
for k, h in H.items():
    b = np.array([ber_ook(snr_linear(Config(p_led=p), h)) for p in POWERS_W])
    ber_curves[k] = b
    ok = np.where(b <= FEC_HD)[0]
    thr = POWERS_W[ok[0]] if len(ok) else np.nan
    print(f"{k:<24}{h:12.3e}"
          + (f"{1e3*thr:21.2f} mW" if np.isfinite(thr) else f"{'not reached':>24}"))

base = POWERS_W[np.where(ber_curves["metaheuristic"] <= FEC_HD)[0][0]]
print("\nTransmit-power saving at the FEC threshold, vs the metaheuristic:")
for k in ("reactive closed form", "predictive", "genie"):
    idx = np.where(ber_curves[k] <= FEC_HD)[0]
    if len(idx):
        print(f"  {k:<24}{10*np.log10(base/POWERS_W[idx[0]]):6.2f} dB")

# ==========================================================================
# (b) element-count sweep -- parallels Fig. 6 of the reference system
# ==========================================================================
PER_PANEL = (3, 4, 5, 6, 8, 10)
print(f"\n(b) Element sweep (their Fig. 6 shows 48-50 dB -> 85-92 dB "
      f"over 0 -> 400 elements)")
print(f"\n{'K':>5}{'cosine SNR':>13}{'specular SNR':>15}{'gap':>8}")

sweep = {"cosine": [], "specular": [], "K": []}
for npp in PER_PANEL:
    c = Config(mirrors_per_panel=npp)
    mp, mn = build_mirrors(c)
    row = {}
    for model in ("cosine", "specular"):
        vals = []
        for s in range(freeze.N_SEEDS_SWEEP):
            tr = make_trace(c, 6.0, DT, SPEED, "walking",
                            np.random.default_rng(SEED0 + s))
            e = TraceEvaluator(c, mp, mn, tr, model=model, **build_diff(c))
            vals.append(e.snr_db(0).mean())
        row[model] = float(np.mean(vals))
    sweep["K"].append(len(mp))
    sweep["cosine"].append(row["cosine"])
    sweep["specular"].append(row["specular"])
    print(f"{len(mp):5d}{row['cosine']:13.2f}{row['specular']:15.2f}"
          f"{row['cosine']-row['specular']:8.2f}")

print("\n  The overstatement GROWS with array size: the cosine model lets every")
print("  mirror contribute whatever its orientation, so its gain scales with K")
print("  more favourably than physics allows. Any extrapolation to large arrays")
print("  made under that model is therefore optimistic by a growing margin.")

# ==========================================================================
# figures
# ==========================================================================
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7), constrained_layout=True)

styles = {"no IRS": ("-", "0.5"), "metaheuristic": ("--", "C3"),
          "reactive closed form": ("-", "C0"), "predictive": ("-.", "C1"),
          "genie": (":", "k")}
for k, (ls, c) in styles.items():
    axes[0].semilogy(1e3 * POWERS_W, np.clip(ber_curves[k], 1e-12, 0.5),
                     ls, color=c, lw=1.6, label=k)
axes[0].axhline(FEC_HD, color="grey", lw=1, ls=":")
axes[0].text(2.1, FEC_HD * 1.4, "HD-FEC", fontsize=7, color="0.35")
axes[0].axhline(FEC_SD, color="grey", lw=1, ls=":")
axes[0].text(2.1, FEC_SD * 1.4, "SD-FEC", fontsize=7, color="0.35")
axes[0].set_xscale("log")
axes[0].set_ylim(1e-10, 0.5)
axes[0].set_xlabel("transmit optical power [mW]")
axes[0].set_ylabel("BER")
axes[0].set_title(f"Specular model, slew {SLEW_MS} ms, "
                  f"$\\sigma$ = {100*SIGMA:.0f} cm", fontsize=10)
axes[0].grid(alpha=.3, which="both")
axes[0].legend(fontsize=7.5, loc="lower left")

axes[1].plot(sweep["K"], sweep["cosine"], "-o", ms=4,
             label="cosine model (base paper)")
axes[1].plot(sweep["K"], sweep["specular"], "-s", ms=4,
             label="specular reflection")
axes[1].fill_between(sweep["K"], sweep["specular"], sweep["cosine"],
                     color="C3", alpha=.12, label="overstatement")
axes[1].set_xlabel("number of mirror elements $K$")
axes[1].set_ylabel("mean SNR [dB]")
axes[1].set_title("Divergence grows with array size", fontsize=10)
axes[1].grid(alpha=.3)
axes[1].legend(fontsize=8)

print("\nwrote:", ", ".join(save_fig(fig, "fig11_ber")))
