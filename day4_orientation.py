"""
Day 4: random device orientation.

Three receiver assumptions, both reflection models:

  face-up   the base paper's assumption throughout
  genie     device randomly oriented, IRS controller KNOWS the true pose
  blind     device randomly oriented, controller assumes face-up (what the
            base paper's optimisation implicitly does)

Outputs
  Fig 6  CDF of SNR under each assumption
  outage probabilities at a stated threshold
  the effect decomposed into device tilt vs controller pose-blindness

Run:  python3 day4_orientation.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import freeze
from channel import Config, build_mirrors, snr_db, upward_normals
from diffuse import build_patches, diffuse_gain
from fitness import GeometryCache, analytic_normals
from orientation import POLAR_MODELS, describe, sample_pd_normals
from plotting import publication_style, save_fig

SEED = freeze.SEED0
N_DROPS = freeze.N_DROPS       # user positions x orientations per case
SNR_THRESHOLD_DB = 50.0        # outage threshold; LoS-only mean is ~49 dB

publication_style()
cfg = Config()
DIFF = freeze.diffuse_kw(cfg)
build_diff = freeze.diffuse_kw
mpos, mnrm = build_mirrors(cfg)
rng = np.random.default_rng(SEED)

print(freeze.banner("day4_orientation"))
print("Random device orientation model (Soltani et al. 2019)")
print(describe())
print(f"\nPD field of view: {cfg.fov_deg:.0f} deg (their Table 3)")
print(f"Monte Carlo: {N_DROPS} drops per case, seed {SEED}")

# user positions drawn uniformly over the receiver plane
lx, ly, _ = cfg.room
pos = np.stack([rng.uniform(0.25, lx - 0.25, N_DROPS),
                rng.uniform(0.25, ly - 0.25, N_DROPS),
                np.full(N_DROPS, cfg.rx_plane_z)], axis=-1)

up = upward_normals(N_DROPS)
normals = {"face-up": up}
for act in POLAR_MODELS:
    normals[act] = sample_pd_normals(act, N_DROPS, rng)


# First-order diffuse light does not depend on the mirror configuration, but it
# DOES depend on the true detector orientation, so it is computed once per set
# of true normals and cached. GeometryCache carries LoS + IRS only.
_PATCHES = build_patches(cfg) if freeze.DIFFUSE else None
_H_DIFFUSE = {}


def diffuse_for(true_normals):
    """First-order diffuse gain at every drop, for one set of true PD normals."""
    if not freeze.DIFFUSE:
        return 0.0
    key = id(true_normals)
    if key not in _H_DIFFUSE:
        _H_DIFFUSE[key] = diffuse_gain(cfg, pos, true_normals, patches=_PATCHES)
    return _H_DIFFUSE[key]


def run(model, true_normals, assumed_normals):
    """
    Steer the IRS using `assumed_normals`, evaluate against `true_normals`.

    Passing the same array for both is the genie case; passing face-up as the
    assumption while the device is tilted is the blind case.
    """
    h = np.empty(N_DROPS)
    for i in range(N_DROPS):
        # controller's view of the world -> mirror configuration
        plan = GeometryCache(cfg, mpos, mnrm, pos[i][None, :],
                             assumed_normals[i][None, :], model=model)
        cfgn = analytic_normals(plan, 0)
        # what the device actually experiences
        real = GeometryCache(cfg, mpos, mnrm, pos[i][None, :],
                             true_normals[i][None, :], model=model)
        h[i] = real.total_gain(cfgn)[0]
    return snr_db(cfg, h + diffuse_for(true_normals))


cases = []
for model in ("cosine", "specular"):
    cases.append((model, "face-up", run(model, up, up)))
    for act in POLAR_MODELS:
        cases.append((model, f"{act}, genie",
                      run(model, normals[act], normals[act])))
        cases.append((model, f"{act}, blind",
                      run(model, normals[act], up)))

# A tilted PD can put every arriving ray outside the field of view, leaving
# zero received power. Those drops are genuine total link loss, not small
# numbers -- keep them out of the dB statistics and report them on their own.
print(f"\n{'model':<10}{'receiver':<18}{'mean*':>8}{'median':>8}"
      f"{'5th pct':>9}{'outage':>9}{'lost':>7}")
res, res_live = {}, {}
for model, label, s in cases:
    live = s > 0.0                       # h > 0; dead links sit at -300 dB
    out = float(np.mean(s < SNR_THRESHOLD_DB))
    res[(model, label)] = s
    res_live[(model, label)] = s[live]
    print(f"{model:<10}{label:<18}{s[live].mean():8.2f}"
          f"{np.median(s[live]):8.2f}{np.percentile(s[live], 5):9.2f}"
          f"{100*out:8.1f}%{100*(1-live.mean()):6.1f}%")

print(f"\n(* mean over surviving links only. outage = P[SNR < "
      f"{SNR_THRESHOLD_DB:.0f} dB] including lost links;")
print("   lost = fraction with zero received power, every ray outside the FoV)")

# ---- decompose into the two separate effects ------------------------------
print("\nDecomposition [dB]; positive = SNR lost")
print(f"{'model':<10}{'activity':<9}{'tilt effect':>12}{'pose-blind':>12}"
      f"{'outage +':>10}")
for model in ("cosine", "specular"):
    base = res_live[(model, "face-up")]
    for act in POLAR_MODELS:
        genie = res_live[(model, f"{act}, genie")]
        blind = res_live[(model, f"{act}, blind")]
        d_out = (np.mean(res[(model, f"{act}, blind")] < SNR_THRESHOLD_DB)
                 - np.mean(res[(model, "face-up")] < SNR_THRESHOLD_DB))
        print(f"{model:<10}{act:<9}{base.mean() - genie.mean():12.2f}"
              f"{genie.mean() - blind.mean():12.2f}{100*d_out:9.1f}%")

print("\n  tilt effect = face-up -> genie. NEGATIVE means tilting HELPS: the")
print("                IRS panels sit on the walls at z = 1.5 m, so a face-up")
print("                PD is badly aimed for wall reflections. Mean SNR rises,")
print("                but see the outage column -- the spread widens sharply.")
print("  pose-blind  = genie -> blind, the cost of the controller not knowing")
print("                the device pose, i.e. what assuming a face-up PD spends.")

print("\n  NOTE the cosine-model rows: pose-blind is exactly 0.00 dB. This is")
print("  structural, not a rounding artefact. Under the base paper's model the")
print("  optimal normal is n = u_out, which depends only on mirror and receiver")
print("  POSITIONS. Device orientation cannot change the optimal steering, so")
print("  knowing the pose is worth precisely nothing. Under specular reflection")
print("  the optimum is the bisector, the LED direction enters, and pose")
print("  knowledge is worth real dB. Orientation-aware IRS control is therefore")
print("  unmodellable in the prevailing channel model -- a second consequence")
print("  of omitting the law of reflection.")

# ---- figure ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7), constrained_layout=True,
                         sharey=True)
styles = {"face-up": ("-", "k"), "sitting, genie": ("--", "C0"),
          "sitting, blind": (":", "C0"), "walking, genie": ("--", "C1"),
          "walking, blind": (":", "C1")}
for ax, model in zip(axes, ("cosine", "specular")):
    for label, (ls, c) in styles.items():
        s = np.sort(res[(model, label)])
        ax.plot(s, np.linspace(0, 1, len(s)), ls=ls, color=c, lw=1.6,
                label=label)
    ax.axvline(SNR_THRESHOLD_DB, color="grey", lw=1, ls=":")
    ax.set_xlim(40, 80)
    ax.set_xlabel("electrical SNR [dB]")
    ax.set_title(f"{model} reflection model", fontsize=10)
    ax.grid(alpha=.3)
axes[0].set_ylabel("CDF")
axes[0].legend(fontsize=8, loc="lower right")
fig.suptitle("Effect of random device orientation on IRS-assisted VLC",
             fontsize=11)
print("\nwrote:", ", ".join(save_fig(fig, "fig6_orientation")))
