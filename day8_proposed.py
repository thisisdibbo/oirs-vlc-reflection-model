"""
Day 8: the proposed scheme, scored against the base paper's optimisers.

Every scheme is charged its own real control latency:

    tau = tau_compute + tau_slew

  metaheuristics  tau_compute ~ 5 s measured on this machine (Day 3)
                  + tau_slew 10-50 ms (base paper, p.8)
  closed form     tau_compute = one O(K) update, measured below
                  + the same tau_slew
  predictive      as closed form, but aimed at the pose the user will have
                  when the mirrors finish moving

Run:  python3 day8_proposed.py     (~4 min)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from channel import Config, build_mirrors, snr_db
from mobility import make_trace
from plotting import publication_style, save_fig
from steering import (complexity_table, predict_pose,
                      predictive_configs, time_bisector)
from trace_eval import TraceEvaluator

import freeze
SEED0 = freeze.SEED0
N_SEEDS = freeze.N_SEEDS
DURATION = 15.0
DT = 0.01                      # 10 ms resolution: actuator spec starts at 10 ms
SPEED = 1.0
MODEL = "specular"
SLEWS_MS = (10, 20, 50)        # base paper's stated actuator range
TAU_META_S = freeze.TAU_META_S   # measured by validate_optimizers.py, pinned in freeze.py
SIGMAS = (0.0, 0.05, 0.10, 0.20)   # position estimation error [m]

publication_style()
cfg = Config()
DIFF = freeze.diffuse_kw(cfg)
build_diff = freeze.diffuse_kw
mpos, mnrm = build_mirrors(cfg)

# ---- measured cost of one closed-form update ------------------------------
probe = make_trace(cfg, 1.0, DT, SPEED, "walking", np.random.default_rng(0))
ev0 = TraceEvaluator(cfg, mpos, mnrm, probe, model=MODEL, **DIFF)
best = np.argmax(ev0.base[:, :, 0], axis=1)
u_in_sel = ev0.u_in[np.arange(ev0.K), best]
t_closed = time_bisector(mpos, u_in_sel, probe.pos[0])

print(f"CONTROL COST PER UPDATE, K = {len(mpos)} mirrors")
print(f"  closed-form bisector  {1e6*t_closed:9.1f} us   (measured)")
print(f"  GA (frozen run, measured) {1e6*TAU_META_S:9.0f} us   "
      f"= {TAU_META_S:.2f} s")
print(f"  ratio                 {TAU_META_S/t_closed:9.0f}x")

tbl = complexity_table(len(mpos), n_pop=30, n_iter=1500)
print(f"\n{'scheme':<22}{'order':<26}{'fitness evals/update':>21}")
for k, v in tbl.items():
    print(f"{k:<22}{v['order']:<26}{v['fitness_evals']:>21,}")

# ==========================================================================
# trace study
# ==========================================================================
print(freeze.banner("day8_proposed"))
print(f"\n{N_SEEDS} seeds x {DURATION:.0f} s, walking {SPEED} m/s, "
      f"model = {MODEL}, dt = {DT*1000:.0f} ms")


def build_seed(seed):
    """True trace + evaluator + LED assignment. Built once, reused for all
    slew/sigma combinations, which is what makes this tractable."""
    tr = make_trace(cfg, DURATION, DT, SPEED, "walking",
                    np.random.default_rng(seed))
    ev = TraceEvaluator(cfg, mpos, mnrm, tr, model=MODEL, **DIFF)
    best = np.argmax(ev.base.mean(axis=2), axis=1)          # (K,)
    u_in_sel = ev.u_in[np.arange(ev.K), best]
    return tr, ev, u_in_sel


def score(tr, ev, u_in_sel, seed, slew_ms, sigma):
    slew = int(round(slew_ms / 1000.0 / DT))
    meta = min(int(round((TAU_META_S + slew_ms / 1000.0) / DT)), ev.T - 2)

    out = {"no IRS": snr_db(cfg, ev.h_los).mean(),
           "genie (tau = 0)": ev.snr_db(0).mean(),
           "metaheuristic": ev.snr_db(meta).mean(),
           "closed form": ev.snr_db(slew).mean()}

    pred = predict_pose(tr.pos, tr.vel, slew_ms / 1000.0, cfg.room,
                        rng=np.random.default_rng(seed + 77),
                        sigma_pos=sigma, sigma_vel=sigma)
    Np = predictive_configs(mpos, u_in_sel, pred)
    # commanded at t was computed at t - slew, aimed at where the user will be
    if slew:
        Np = np.concatenate([Np[:, :1, :].repeat(slew, axis=1),
                             Np[:, :-slew, :]], axis=1)
    out["predictive"] = ev.snr_db(0, N=Np).mean()
    return out


SCHEMES = ["no IRS", "metaheuristic", "closed form", "predictive",
           "genie (tau = 0)"]
built = [build_seed(SEED0 + i) for i in range(N_SEEDS)]
print(f"  built {N_SEEDS} traces")
res = {}
for slew_ms in SLEWS_MS:
    for sigma in SIGMAS:
        rows = [score(*built[i], SEED0 + i, slew_ms, sigma)
                for i in range(N_SEEDS)]
        res[(slew_ms, sigma)] = {
            s: (np.mean([r[s] for r in rows]), np.std([r[s] for r in rows]))
            for s in SCHEMES}
    print(f"  done slew = {slew_ms} ms")

# ---- headline table -------------------------------------------------------
print(f"\nMean SNR [dB]. Position/velocity estimation error sigma = 0 is a "
      f"genie predictor;\nquote the sigma > 0 rows.")
print(f"\n{'slew':>5}{'sigma':>7}" + "".join(f"{s:>18}" for s in SCHEMES))
for slew_ms in SLEWS_MS:
    for sigma in SIGMAS:
        r = res[(slew_ms, sigma)]
        print(f"{slew_ms:4d}ms{sigma:6.2f}m"
              + "".join(f"{r[s][0]:11.2f}±{r[s][1]:<5.2f}" for s in SCHEMES))

# ---- what the proposal buys ----------------------------------------------
print("\nGain over the metaheuristic baseline [dB]")
print(f"{'slew':>5}{'sigma':>7}{'closed form':>14}{'predictive':>13}"
      f"{'genie':>9}")
for slew_ms in SLEWS_MS:
    for sigma in SIGMAS:
        r = res[(slew_ms, sigma)]
        b = r["metaheuristic"][0]
        print(f"{slew_ms:4d}ms{sigma:6.2f}m"
              + f"{r['closed form'][0]-b:14.2f}"
              + f"{r['predictive'][0]-b:13.2f}"
              + f"{r['genie (tau = 0)'][0]-b:9.2f}")

print("\nFraction of the genie bound recovered")
print(f"{'slew':>5}{'sigma':>7}{'closed form':>14}{'predictive':>13}")
for slew_ms in SLEWS_MS:
    for sigma in SIGMAS:
        r = res[(slew_ms, sigma)]
        lo, hi = r["metaheuristic"][0], r["genie (tau = 0)"][0]
        f = lambda s: 100 * (r[s][0] - lo) / (hi - lo)
        print(f"{slew_ms:4d}ms{sigma:6.2f}m{f('closed form'):13.0f}%"
              f"{f('predictive'):12.0f}%")

# ==========================================================================
# the design rule this implies
# ==========================================================================
from fitness import pointing_footprint
fp = pointing_footprint(cfg, 3.0, 0.5)
print("\n" + "=" * 70)
print("WHEN IS PREDICTION WORTH IT?")
print("=" * 70)
print(f"\nPointing footprint at 3 m: {100*fp:.1f} cm. Aim error of that order")
print("costs the whole IRS gain, whichever way the error arises.")
print("\n  reactive closed form : aim point is stale by v * tau_slew")
print("  predictive           : aim point carries the tracking error sigma")
print("\nSo prediction only pays when the staleness it removes exceeds the")
print("estimation error it introduces:\n")
print("      v * tau_slew  >  sigma\n")
for sl in SLEWS_MS:
    print(f"  slew {sl:3d} ms at {SPEED} m/s -> staleness "
          f"{100*SPEED*sl/1000:5.1f} cm; predict only if sigma < that")
print("\nMeasured: which scheme wins (mean SNR, dB)")
print(f"{'':>12}" + "".join(f"{f'sigma={s:.2f}':>14}" for s in SIGMAS))
for sl in SLEWS_MS:
    row = []
    for sg in SIGMAS:
        c = res[(sl, sg)]["closed form"][0]
        p = res[(sl, sg)]["predictive"][0]
        row.append(f"{'predict' if p > c else 'reactive':>9}"
                   f"{max(p, c) - min(p, c):5.1f}")
    print(f"  slew {sl:3d} ms" + "".join(f"{v:>14}" for v in row))
print("\n  Sub-centimetre tracking is not available from VLC positioning, so")
print("  the ROBUST recommendation is the reactive closed form: it recovers")
print(f"  {100*(res[(10,0.0)]['closed form'][0]-res[(10,0.0)]['metaheuristic'][0])/(res[(10,0.0)]['genie (tau = 0)'][0]-res[(10,0.0)]['metaheuristic'][0]):.0f}% "
      f"of the bound at 10 ms slew with no pose estimate at all, and its")
print("  performance does not depend on tracking accuracy. Prediction is an")
print("  upgrade for systems that can localise to a few centimetres, not the")
print("  headline contribution.")

# ---- figure ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7), constrained_layout=True)

x = np.arange(len(SLEWS_MS))
w = 0.18
for i, s in enumerate(SCHEMES):
    mu = [res[(sl, 0.10)][s][0] for sl in SLEWS_MS]
    sd = [res[(sl, 0.10)][s][1] for sl in SLEWS_MS]
    axes[0].bar(x + (i - 2) * w, mu, w, yerr=sd, capsize=2, label=s)
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"{s} ms" for s in SLEWS_MS])
axes[0].set_xlabel("actuator slew time (base paper, p.8)")
axes[0].set_ylabel("mean SNR [dB]")
axes[0].set_ylim(45, None)
axes[0].set_title(r"Each scheme charged its own latency ($\sigma$ = 10 cm)")
axes[0].grid(alpha=.3, axis="y")
axes[0].legend(handlelength=1.2, borderpad=0.3, labelspacing=0.25)

for sl, mk in zip(SLEWS_MS, "os^"):
    axes[1].plot(SIGMAS, [res[(sl, sg)]["predictive"][0] for sg in SIGMAS],
                 "-", marker=mk, ms=4, label=f"predictive, slew {sl} ms")
    axes[1].plot(SIGMAS, [res[(sl, sg)]["closed form"][0] for sg in SIGMAS],
                 "--", marker=mk, ms=4, alpha=.6,
                 label=f"closed form, slew {sl} ms")
axes[1].axhline(res[(10, 0.0)]["metaheuristic"][0], color="k", ls=":",
                label="metaheuristic baseline")
axes[1].set_xlabel("position / velocity estimation error $\\sigma$ [m]")
axes[1].set_ylabel("mean SNR [dB]")
axes[1].set_title("Sensitivity to tracking accuracy")
axes[1].grid(alpha=.3)
axes[1].legend(ncol=2, loc="center right", handlelength=1.4,
               borderpad=0.3, labelspacing=0.2, columnspacing=0.8,
               fontsize=5.5)

# No suptitle: the caption carries the framing, and a long one forces the
# figure wider than the text block, which LaTeX then shrinks.
print("\nwrote:", ", ".join(save_fig(fig, "fig10_proposed")))
