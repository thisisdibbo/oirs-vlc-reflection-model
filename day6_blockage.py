"""
Day 6: blockage.

Obstacles are the base paper's own Table 3 furniture (table, sofa, human), as
axis-aligned boxes. Three cases:

  none     no obstacles
  static   Table 3 furniture exactly as published
  mobile   Table 3 table + sofa, with the human replaced by one who walks

All three links are tested for occlusion: LED->mirror, mirror->PD, LED->PD.

Run:  python3 day6_blockage.py      (~3 min)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from blockage import (boxes_from_config, inside_any, link_masks,
                      walking_box)
from channel import Config, build_mirrors
from mobility import make_trace
from plotting import publication_style, save_fig
from trace_eval import TraceEvaluator

import freeze
SEED0 = freeze.SEED0
N_SEEDS = freeze.N_SEEDS
DURATION = 20.0
DT = 0.02
SPEED = 1.0
TAUS_MS = (0, 20, 50, 100, 500)
THRESH_DB = 50.0

publication_style()
cfg = Config()
DIFF = freeze.diffuse_kw(cfg)
build_diff = freeze.diffuse_kw
mpos, mnrm = build_mirrors(cfg)
lags = [int(round(ms / 1000.0 / DT)) for ms in TAUS_MS]

print("Obstacles from the base paper's Table 3 "
      "[xmin, xmax, ymin, ymax, height, rho]:")
for name, x0, x1, y0, y1, h, rho in cfg.blockers:
    print(f"  {name:<6} [{x0}, {x1}, {y0}, {y1}, {h}, {rho}]")
print(freeze.banner("day6_blockage"))
print(f"\n{N_SEEDS} seeds x {DURATION:.0f} s, walking at {SPEED} m/s\n")

static_all = boxes_from_config(cfg)
static_wo_human = boxes_from_config(cfg, exclude=("human",))


def run(model, case, seed):
    rng = np.random.default_rng(seed)
    tr = make_trace(cfg, DURATION, DT, SPEED, "walking", rng)
    if case == "none":
        masks = None
    elif case == "static":
        masks = link_masks(cfg, mpos, tr.pos, static_all)
    else:
        human = walking_box(len(tr), DT, 1.0, cfg.room,
                            np.random.default_rng(seed + 9999))
        masks = link_masks(cfg, mpos, tr.pos, static_wo_human, boxes_t=human)
    ev = TraceEvaluator(cfg, mpos, mnrm, tr, model=model, masks=masks, **DIFF)
    # drop timesteps where the walker is standing inside a piece of furniture
    bad = inside_any(tr.pos, static_all)
    rows = {}
    for ms, l in zip(TAUS_MS, lags):
        s = ev.snr_db(l)
        keep = ~bad[l:]
        s = s[keep]
        live = s > 0.0                       # -300 dB sentinel = total outage
        rows[ms] = (s[live].mean() if live.any() else np.nan,
                    float(np.median(s[live])) if live.any() else np.nan,
                    float(np.mean(s < THRESH_DB)),
                    float(1.0 - live.mean()))
    return float(bad.mean()), rows


CASES = ("none", "static", "mobile")
res = {}
for model in ("cosine", "specular"):
    for case in CASES:
        runs = [run(model, case, SEED0 + i) for i in range(N_SEEDS)]
        arr = np.array([[runs[i][1][ms] for ms in TAUS_MS]
                        for i in range(N_SEEDS)])          # (S, T, 4)
        res[(model, case)] = (arr[:, :, 0].mean(0), arr[:, :, 0].std(0),
                              arr[:, :, 1].mean(0), arr[:, :, 2].mean(0),
                              arr[:, :, 3].mean(0))
        excl = float(np.mean([r[0] for r in runs]))
    print(f"  done {model}  ({100*excl:.1f}% of timesteps dropped: "
          f"walker inside furniture)")

print(f"\nMean SNR [dB] over surviving links, {N_SEEDS} seeds")
print(f"{'model':<10}{'obstacles':<9}" + "".join(f"{ms:>13}" for ms in TAUS_MS))
for model in ("cosine", "specular"):
    for case in CASES:
        mu, sd = res[(model, case)][0], res[(model, case)][1]
        print(f"{model:<10}{case:<9}"
              + "".join(f"{m:7.2f}±{s:<5.2f}" for m, s in zip(mu, sd)))

print(f"\nOutage P[SNR < {THRESH_DB:.0f} dB]   /   total link loss")
print(f"{'model':<10}{'obstacles':<9}" + "".join(f"{ms:>16}" for ms in TAUS_MS))
for model in ("cosine", "specular"):
    for case in CASES:
        r = res[(model, case)]
        print(f"{model:<10}{case:<9}"
              + "".join(f"{100*a:7.1f}% /{100*b:5.1f}%"
                        for a, b in zip(r[3], r[4])))

# ---- what blockage costs, at zero latency ---------------------------------
print("\nBlockage cost at tau = 0 [dB], relative to no obstacles")
for model in ("cosine", "specular"):
    base = res[(model, "none")][0][0]
    for case in ("static", "mobile"):
        print(f"  {model:<9}{case:<8}{base - res[(model, case)][0][0]:6.2f}")

# ---- and how it compounds with latency ------------------------------------
i50 = TAUS_MS.index(50)
print(f"\nCompounding: total loss at tau = 50 ms WITH mobile blockage, "
      f"vs the clean tau = 0 case")
for model in ("cosine", "specular"):
    clean = res[(model, "none")][0][0]
    worst = res[(model, "mobile")][0][i50]
    blk = clean - res[(model, "mobile")][0][0]
    lat = res[(model, "mobile")][0][0] - worst
    print(f"  {model:<9} blockage {blk:5.2f} dB + latency {lat:5.2f} dB "
          f"= {clean - worst:5.2f} dB")

# ---- figure ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
width = 0.25
x = np.arange(len(CASES))
for j, model in enumerate(("cosine", "specular")):
    ax = axes[j]
    for k, ms in enumerate((0, 50, 500)):
        i = TAUS_MS.index(ms)
        mu = [res[(model, c)][0][i] for c in CASES]
        sd = [res[(model, c)][1][i] for c in CASES]
        ax.bar(x + (k - 1) * width, mu, width, yerr=sd, capsize=3,
               label=rf"$\tau$ = {ms} ms")
    ax.set_xticks(x)
    ax.set_xticklabels(["no obstacles", "Table 3 static", "mobile human"])
    ax.set_ylabel("mean SNR [dB]")
    ax.set_ylim(40, None)
    ax.set_title(f"{model} reflection model", fontsize=10)
    ax.grid(alpha=.3, axis="y")
    ax.legend(fontsize=8)
fig.suptitle("Blockage and control latency compound differently in the "
             "two models", fontsize=11)
print("\nwrote:", ", ".join(save_fig(fig, "fig8_blockage")))
