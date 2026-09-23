"""
Day 3 validation + Figure 3.

Three things, in order:

A. Show that free-angle search is infeasible. All three metaheuristics are run
   over the raw 2K roll/yaw variables and compared against the analytic
   bisector. They stall ~11 dB short because a 1 cm^2 PD at 3 m subtends ~0.1
   deg and random mirrors never land in that window. This is a supporting
   result for the paper, not a bug -- report it.

B. Re-run all three over the mirror-to-target ASSIGNMENT space, where they are
   genuine competitors, and check they reach the analytic optimum. This is the
   real de-risking gate: if an optimiser cannot reach a known optimum in a
   searchable space, it is broken.

C. Measure t_eval per objective mode. That number is the Day 9 latency axis.

Run:  python3 validate_optimizers.py
"""

import platform

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotting import publication_style, save_fig

from channel import (Config, build_mirrors, upward_normals, receiver_grid,
                     noise_variance)
from fitness import (GeometryCache, Objective, AssignmentObjective,
                     analytic_vector, local_targets, pointing_footprint)
from optimizers import ALGORITHMS

SEED = 20260922
N_POP, N_ITER = 30, 60
MAX_ANGLE = np.radians(90.0)
STYLES = {"TLBO": ("-", "o"), "GA": ("--", "s"), "PSO": ("-.", "^")}

publication_style()
cfg = Config()
mpos, mnrm = build_mirrors(cfg)
user = np.array([[3.60, 1.40, cfg.rx_plane_z]])
MODEL = "specular"   # switch to "cosine" to see the base paper's landscape
cache = GeometryCache(cfg, mpos, mnrm, user, upward_normals(1), model=MODEL)

print(f"host       : {platform.machine()}, numpy {np.__version__}")
print(f"model      : {MODEL}\nmirrors K  : {len(mpos)}   population {N_POP}   iterations {N_ITER}")

h_los = cache.h_los[0]
snr_los = float(10 * np.log10((cfg.responsivity * cfg.p_led * h_los) ** 2
                              / noise_variance(cfg, cfg.p_led * h_los)))
f_star = Objective(cache, "single", max_angle=MAX_ANGLE)(
    analytic_vector(cache, 0))
print(f"\nuser {user[0][:2]}   LoS-only {snr_los:.2f} dB   "
      f"analytic optimum {f_star:.2f} dB  ({f_star - snr_los:+.2f} dB)")

# ==========================================================================
# A. free-angle search -- expected to fail
# ==========================================================================
print("\nA. free-angle search (dim = 2K = %d)" % (2 * len(mpos)))
print(f"{'alg':<6}{'best dB':>9}{'gap':>8}{'evals':>8}  verdict")
free = {}
for name, fn in ALGORITHMS.items():
    obj = Objective(cache, "single", max_angle=MAX_ANGLE)
    r = fn(obj, (-MAX_ANGLE, MAX_ANGLE), n_pop=N_POP, n_iter=N_ITER, seed=SEED)
    free[name] = r
    gap = f_star - r.best_f
    print(f"{name:<6}{r.best_f:9.2f}{gap:8.2f}{r.n_evals:8d}  "
          f"{'PASS' if gap <= 0.5 else 'infeasible as expected'}")

# ==========================================================================
# B. assignment search -- the fair baseline
# ==========================================================================
spot = pointing_footprint(cfg, float(np.median(cache.d2)))
room_cells = int(np.ceil(cfg.room[0] / spot)) * int(np.ceil(cfg.room[1] / spot))
print(f"\n   pointing footprint at median mirror distance: {100*spot:.1f} cm")
print(f"   tiling the whole room would need ~{room_cells:,} candidate targets "
      f"-- rules out room-wide target search")

# candidates = the tracking uncertainty region around the user, at footprint
# resolution. This is what a real controller searches.
targets = local_targets(user[0], radius=2 * spot, n=5)
print(f"\nB. assignment search (dim = K = {len(mpos)}, "
      f"{len(targets)} local targets, spacing {100*spot:.1f} cm)")

probe = AssignmentObjective(cache, targets, "single", max_angle=MAX_ANGLE)
per_target = [probe(probe.all_to_target(g)) for g in range(probe.G)]
best_g = int(np.argmax(per_target))
f_best_cell = per_target[best_g]
print(f"   best single target = #{best_g}, {f_best_cell:.2f} dB   "
      f"(analytic bisector {f_star:.2f} dB)")

# Equal EVALUATION budget, not equal iterations -- TLBO spends 2 x n_pop per
# iteration, so matching iterations would hand it a 2x compute advantage.
EVAL_BUDGET = 45_000
print(f"   budget: {EVAL_BUDGET:,} fitness evaluations for every algorithm\n")
print(f"{'alg':<6}{'best dB':>9}{'gap':>8}{'evals':>9}{'t_eval ms':>11}"
      f"{'tau_comp':>10}  verdict")
assign = {}
for name, fn in ALGORITHMS.items():
    obj = AssignmentObjective(cache, targets, "single", max_angle=MAX_ANGLE)
    per_iter = 2 * N_POP if name == "TLBO" else N_POP
    r = fn(obj, (0, probe.G - 1), n_pop=N_POP,
           n_iter=EVAL_BUDGET // per_iter, seed=SEED)
    assign[name] = r
    gap = f_best_cell - r.best_f
    ok = ("converges" if gap <= 1.0 else
          "partial" if gap <= 2.5 else "stalls (premature convergence)")
    print(f"{name:<6}{r.best_f:9.2f}{gap:8.2f}{r.n_evals:9,d}"
          f"{1e3*r.t_eval:11.3f}{r.tau_compute():9.2f}s  {ok}")

t_eval = float(np.mean([r.t_eval for r in assign.values()]))
print(f"\nt_eval (single user) = {1e3*t_eval:.3f} ms   <- Day 9 latency input")

# ---- the number the paper is built on -------------------------------------
best = max(assign.values(), key=lambda r: r.best_f)
tau_meta = best.tau_compute()
print(f"\n   best metaheuristic ({best.name}) needs {tau_meta:.2f} s "
      f"to come within {f_best_cell - best.best_f:.2f} dB of the optimum.")
for v in (0.5, 1.0, 1.5):
    print(f"   a user at {v:.1f} m/s travels {v*tau_meta:5.1f} m in that time "
          f"(room is {cfg.room[0]:.0f} m across)")
print(f"   analytic bisector reaches the SAME optimum in O(K) = {len(mpos)} "
      f"operations, no search at all.")

# ==========================================================================
# C. Figure 3
# ==========================================================================
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7), constrained_layout=True)
for name, r in assign.items():
    ls, mk = STYLES[name]
    axes[0].plot(np.arange(1, len(r.history) + 1), r.history,
                 ls=ls, marker=mk, markevery=8, ms=4, label=name)
    axes[1].plot(np.array(r.history_evals) * t_eval * 1e3, r.history,
                 ls=ls, marker=mk, markevery=8, ms=4, label=name)
for name, r in free.items():
    ls, _ = STYLES[name]
    axes[0].plot(np.arange(1, len(r.history) + 1), r.history,
                 ls=ls, color="0.65", lw=1)
axes[0].plot([], [], color="0.65", lw=1, label="free-angle (infeasible)")

for ax, xl in ((axes[0], "iteration"),
               (axes[1], r"optimiser compute $\tau_{\rm compute}$ [ms]")):
    ax.axhline(f_best_cell, color="k", lw=1.2, ls=":", label="analytic optimum")
    ax.axhline(snr_los, color="grey", lw=1.0, ls=":", label="LoS only")
    ax.set_xlabel(xl)
    ax.set_ylabel("electrical SNR [dB]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
axes[1].set_xscale("log")
axes[0].set_title("Convergence per iteration", fontsize=10)
axes[1].set_title(r"Convergence per unit compute", fontsize=10)
save_fig(fig, "fig3_convergence")
print("\nwrote fig3_convergence.pdf / .png")

# ==========================================================================
# room-wide cost
# ==========================================================================
_, _, pts = receiver_grid(cfg, 11)
room_cache = GeometryCache(cfg, mpos, mnrm, pts, upward_normals(len(pts)), model=MODEL)
room_obj = AssignmentObjective(room_cache, targets, "paper", max_angle=MAX_ANGLE)
room_obj.evaluate_many(
    np.random.default_rng(0).uniform(0, room_obj.G - 1, size=(20, room_obj.dim)))
print(f"\nt_eval (room-wide, {len(pts)} points) = {1e3*room_obj.t_eval:.2f} ms")
print(f"   a {N_POP}x{N_ITER} TLBO run costs "
      f"{2*N_POP*N_ITER*room_obj.t_eval:.1f} s of controller time")
