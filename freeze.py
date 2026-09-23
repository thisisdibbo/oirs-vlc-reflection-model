"""
Day 15 freeze: the frozen experimental configuration, stated in one place.

Every experiment script imports its seed count, Monte Carlo size and diffuse
setting from here, so that "what settings produced the submitted figures" has
exactly one answer. Editing a constant below changes every figure; editing a
constant inside a day script does not, and should not be done.

    Development run (minutes):   LIFI_FAST=1 python3 day5_mobility.py
    Frozen run    (submission):             python3 day5_mobility.py

WHICH EXPERIMENTS GET DIFFUSE REFLECTIONS, AND WHY
--------------------------------------------------
Not all of them, deliberately.

The reference system states on its p.6 that it models shot and thermal noise
only, and argues that the IRS's directional reflections suppress diffuse
multipath. Three scripts exist to REPRODUCE that system on its own terms:

    compare_models.py       reproduction of its Fig. 3 bands
    validate_optimizers.py  its TLBO/GA/PSO comparison
    sweep_panel.py          sensitivity to the panel size it does not state

Those keep diffuse OFF. Adding a channel component the original excludes would
make the reproduction a comparison against a different system, and the transmit
power was calibrated to its Fig. 3a under its own assumptions.

The extension experiments -- Days 4, 5, 6, 8 and 9 -- are claims about physical
behaviour, not about the original's numbers, so they run with diffuse ON. This
is the conservative direction: a floor of scattered light partially covers a
mis-aimed specular path, so including it SHRINKS the staleness penalty and
therefore works against this paper's own argument (Day 7 measured the size of
that effect: 17.15 dB -> 15.68 dB).

Day 7 is the experiment that measures the difference, so it sweeps diffuse on
and off by design and ignores DIFFUSE below.
"""

import os

# Set LIFI_FAST=1 in the environment for a quick development run.
FAST = bool(os.environ.get("LIFI_FAST"))

# ---- randomness ----------------------------------------------------------
SEED0 = 20260922

# ---- sample sizes --------------------------------------------------------
N_SEEDS       = 6 if FAST else 50    # independent mobility traces per cell
N_SEEDS_SUB   = 4 if FAST else 25    # secondary sweeps (was N_SEEDS // 2)
N_SEEDS_SWEEP = 3 if FAST else 12    # inner loop of the element-count sweep
N_DROPS       = 2000 if FAST else 10000   # Day 4 Monte Carlo drops

# ---- channel -------------------------------------------------------------
DIFFUSE = True          # first-order diffuse bounce in the extension experiments

# ---- measured controller timings -----------------------------------------
# Both are produced by validate_optimizers.py on the frozen run and pinned here
# so that Days 8 and 9 charge the metaheuristic the same latency the optimiser
# section reports. Re-measure and update BOTH if the hardware changes; the
# ratio to the closed form is machine-dependent, which is why the paper reports
# it as an order of magnitude rather than a precise factor.
T_EVAL_MS  = 0.169      # one fitness evaluation, single user
TAU_META_S = 7.69       # GA convergence: 45,000 evaluations at T_EVAL_MS

# ---- figures -------------------------------------------------------------
FIG_DPI = 600           # raster fallback; the PDF vector copy is what ships


def diffuse_kw(cfg):
    """
    Keyword arguments enabling the first-order diffuse bounce, for splatting
    into a TraceEvaluator call:

        TraceEvaluator(cfg, mpos, mnrm, tr, model=m, **freeze.diffuse_kw(cfg))

    Returns an empty dict when DIFFUSE is False, so the same call site works
    either way. Patch construction is ~1 ms, so this is not worth caching.
    """
    if not DIFFUSE:
        return {}
    from diffuse import build_patches
    return dict(diffuse=True, patches=build_patches(cfg))


def banner(script):
    """One line at the top of every run, so a log says what produced it."""
    mode = "FAST (development -- do NOT use for the paper)" if FAST else "FROZEN"
    return (f"[{script}]  {mode}  "
            f"N_SEEDS={N_SEEDS}  N_DROPS={N_DROPS}  diffuse={DIFFUSE}")
