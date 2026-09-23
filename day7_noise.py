"""
Day 7: the augmented channel -- ambient shot noise and first-order diffuse
reflections.

The base paper states on p.6 that it models shot and thermal noise only, and
argues the IRS's directional reflections "naturally suppress both diffuse
multipath components and off-angle ambient light". Both halves of that claim
are tested here rather than repeated.

  A. Ambient: sweep the background photocurrent, including the value implied by
     their own 0.12 uW aside, and measure what each costs.
  B. Diffuse: add first-order reflections from all six room surfaces and
     measure the contribution -- including what it does to the FLOOR that
     remains when a steered configuration goes stale.

Run:  python3 day7_noise.py      (~3 min)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from channel import (Config, build_mirrors, los_gain, receiver_grid, snr_db,
                     upward_normals)
from diffuse import ambient_cases, build_patches, diffuse_gain
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

publication_style()
cfg = Config()
mpos, mnrm = build_mirrors(cfg)
lags = [int(round(ms / 1000.0 / DT)) for ms in TAUS_MS]
patches = build_patches(cfg)

# ==========================================================================
# A. ambient shot noise
# ==========================================================================
print(freeze.banner("day7_noise") + "  [sweeps diffuse on/off by design]")
print("A. AMBIENT SHOT NOISE")
print(f"   patches: {len(patches[0])}, total surface {patches[2].sum():.0f} m^2")
_, _, pts = receiver_grid(cfg, 11)
rxn = upward_normals(len(pts))
h_los = los_gain(cfg, pts, rxn).sum(axis=1)

ref = snr_db(Config(ambient_default=False), h_los).mean()
print(f"\n{'ambient case':<28}{'I_bg':>11}{'mean SNR':>11}{'cost':>8}")
for label, ibg in ambient_cases(cfg).items():
    c = Config(i_bg=ibg, ambient_default=(ibg > 0))
    m = snr_db(c, h_los).mean()
    print(f"{label:<28}{1e6*ibg:9.2f}uA{m:11.2f}{ref - m:8.2f}")

print("\n   Their p.6 claim is that ambient adds ~0.49 dB and leaves >4.5 dB")
print("   of margin. Their own 0.12 uW figure implies I_bg = 0.065 uA, which")
print("   costs essentially nothing -- so the claim is self-consistent but")
print("   rests on an ambient level ~10,000x below the value standard VLC")
print("   noise models assume (740 uA). Which is right depends on the optical")
print("   bandpass filter, which the paper does not specify. Report the sweep.")

# ==========================================================================
# B. diffuse reflections
# ==========================================================================
h_dif = diffuse_gain(cfg, pts, rxn, patches=patches)
print(f"\nB. FIRST-ORDER DIFFUSE REFLECTIONS (wall rho = {cfg.wall_rho})")
print(f"   LoS     {1e6*cfg.p_led*h_los.mean():7.2f} uW")
print(f"   diffuse {1e6*cfg.p_led*h_dif.mean():7.2f} uW "
      f"({100*h_dif.mean()/h_los.mean():.1f}% of LoS)")
print(f"   SNR: LoS only {snr_db(cfg, h_los).mean():.2f} dB -> "
      f"with diffuse {snr_db(cfg, h_los + h_dif).mean():.2f} dB "
      f"(+{snr_db(cfg, h_los+h_dif).mean()-snr_db(cfg, h_los).mean():.2f} dB)")

# ==========================================================================
# C. effect on the staleness result
# ==========================================================================
print(f"\nC. EFFECT ON THE STALENESS RESULT ({N_SEEDS} seeds)")


def run(model, diffuse, ambient_ibg):
    c = Config(i_bg=ambient_ibg, ambient_default=(ambient_ibg > 0))
    rows = []
    for s in range(N_SEEDS):
        tr = make_trace(c, DURATION, DT, SPEED, "walking",
                        np.random.default_rng(SEED0 + s))
        ev = TraceEvaluator(c, mpos, mnrm, tr, model=model,
                            diffuse=diffuse, patches=patches)
        fresh = ev.snr_db(0)
        rows.append([fresh.mean()] +
                    [fresh[l:].mean() - ev.snr_db(l).mean() for l in lags[1:]])
    a = np.array(rows)
    return a.mean(axis=0), a.std(axis=0)


SCENARIOS = [
    ("paper's model", False, 0.0),
    ("+ diffuse", True, 0.0),
    ("+ diffuse + ambient", True, 740e-6),
]
print(f"\n{'model':<10}{'channel':<22}{'SNR@0':>9}"
      + "".join(f"{ms:>9}" for ms in TAUS_MS[1:]))
res = {}
for model in ("cosine", "specular"):
    for label, dif, ibg in SCENARIOS:
        mu, sd = run(model, dif, ibg)
        res[(model, label)] = (mu, sd)
        print(f"{model:<10}{label:<22}{mu[0]:9.2f}"
              + "".join(f"{v:9.2f}" for v in mu[1:]))

print("\n   Columns after SNR@0 are the stale-configuration penalty [dB].")

# ---- the floor argument ---------------------------------------------------
print("\nThe floor: what survives when the configuration is fully stale")
for model in ("cosine", "specular"):
    a = res[(model, "paper's model")][0]
    b = res[(model, "+ diffuse")][0]
    print(f"  {model:<9} no diffuse  {a[0]:6.2f} dB -> {a[0]-a[-1]:6.2f} dB "
          f"at tau = 500 ms")
    print(f"  {'':<9} + diffuse   {b[0]:6.2f} dB -> {b[0]-b[-1]:6.2f} dB")

print("\n   Diffuse reflection needs no steering and cannot go stale, so it")
print("   raises the floor a stale IRS falls back to. It slightly REDUCES the")
print("   measured staleness penalty -- adding realism here helps the base")
print("   paper's position, which is worth saying plainly.")

# ---- figure ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7), constrained_layout=True)
for j, model in enumerate(("cosine", "specular")):
    ax = axes[j]
    for label, _, _ in SCENARIOS:
        mu, sd = res[(model, label)]
        ax.errorbar(TAUS_MS[1:], mu[1:], yerr=sd[1:], marker="o", ms=4,
                    capsize=3, label=label)
    ax.axvspan(*cfg.mirror_response_ms, color="grey", alpha=.18)
    ax.set_xscale("log")
    ax.set_xlabel(r"control latency $\tau$ [ms]")
    ax.set_ylabel(r"$\Delta_{\rm stale}$ [dB]")
    ax.set_title(f"{model} reflection model", fontsize=10)
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8)
fig.suptitle("Augmented channel: ambient shot noise and first-order diffuse "
             "reflections", fontsize=11)
print("\nwrote:", ", ".join(save_fig(fig, "fig9_augmented")))
