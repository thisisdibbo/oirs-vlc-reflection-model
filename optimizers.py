"""
TLBO, GA and PSO for mirror-array IRS configuration.

All three share one signature and return the same Result record, so they can be
swapped in a sweep and charged identically in the control-latency model.

Evaluations per iteration differ, and that difference is a finding, not a
detail:

    TLBO : 2 * n_pop   (teacher phase + learner phase)
    GA   : 1 * n_pop
    PSO  : 1 * n_pop

TLBO reaches a given fitness in fewer ITERATIONS than GA or PSO, which is how
the base paper reports it -- but it pays twice per iteration. Plot convergence
against evaluations and wall-clock, not against iteration index, or the
comparison flatters TLBO. Result.history_evals exists for exactly this.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Result:
    name: str
    best_x: np.ndarray
    best_f: float
    history: list = field(default_factory=list)        # best-so-far per iteration
    history_evals: list = field(default_factory=list)  # cumulative evals per iteration
    n_evals: int = 0
    t_eval: float = float("nan")                       # mean s per fitness call
    wall_time: float = 0.0
    evals_per_iter: int = 0

    def tau_compute(self, t_eval=None, iters=None):
        """
        Optimiser convergence time in seconds -- plan equation 6.

        tau_compute = N_iter * N_pop * t_eval, using the measured per-evaluation
        cost. Pass `iters` to price early stopping at that iteration.
        """
        te = self.t_eval if t_eval is None else t_eval
        n = self.n_evals if iters is None else min(
            self.n_evals, self.evals_per_iter * int(iters))
        return n * te

    def iters_to_reach(self, target):
        """First iteration index whose best-so-far reaches `target`, else None."""
        for i, v in enumerate(self.history):
            if v >= target:
                return i
        return None


def _init_pop(rng, n_pop, lo, hi, x0=None):
    X = rng.uniform(lo, hi, size=(n_pop, len(lo)))
    if x0 is not None:
        X[0] = np.clip(x0, lo, hi)          # optional warm start
    return X


def _bounds(bounds, dim):
    lo, hi = bounds
    return np.full(dim, lo, dtype=float), np.full(dim, hi, dtype=float)


# --------------------------------------------------------------------------
# TLBO -- Rao et al. Teaching-Learning-Based Optimisation (no tuning parameters)
# --------------------------------------------------------------------------

def tlbo(objective, bounds, n_pop=30, n_iter=100, seed=0, x0=None):
    import time
    rng = np.random.default_rng(seed)
    dim = objective.dim
    lo, hi = _bounds(bounds, dim)
    t_start = time.perf_counter()

    X = _init_pop(rng, n_pop, lo, hi, x0)
    f = objective.evaluate_many(X)
    hist, hist_e = [], []

    for _ in range(n_iter):
        # ---- teacher phase ----
        teacher = X[np.argmax(f)]
        mean = X.mean(axis=0)
        tf = rng.integers(1, 3, size=(n_pop, 1))            # teaching factor 1 or 2
        r = rng.random((n_pop, dim))
        Xn = np.clip(X + r * (teacher[None, :] - tf * mean[None, :]), lo, hi)
        fn = objective.evaluate_many(Xn)
        better = fn > f
        X[better], f[better] = Xn[better], fn[better]

        # ---- learner phase ----
        partner = (np.arange(n_pop) + rng.integers(1, n_pop, size=n_pop)) % n_pop
        toward = (f > f[partner])[:, None]
        step = np.where(toward, X - X[partner], X[partner] - X)
        r = rng.random((n_pop, dim))
        Xn = np.clip(X + r * step, lo, hi)
        fn = objective.evaluate_many(Xn)
        better = fn > f
        X[better], f[better] = Xn[better], fn[better]

        hist.append(float(f.max()))
        hist_e.append(objective.n_evals)

    i = int(np.argmax(f))
    return Result("TLBO", X[i].copy(), float(f[i]), hist, hist_e,
                  objective.n_evals, objective.t_eval,
                  time.perf_counter() - t_start, 2 * n_pop)


# --------------------------------------------------------------------------
# GA -- real coded, tournament selection, BLX-alpha crossover, elitism
# --------------------------------------------------------------------------

def ga(objective, bounds, n_pop=30, n_iter=100, seed=0, x0=None,
       p_cross=0.9, alpha=0.5, p_mut=None, sigma_frac=0.1, n_elite=2):
    import time
    rng = np.random.default_rng(seed)
    dim = objective.dim
    lo, hi = _bounds(bounds, dim)
    if p_mut is None:
        p_mut = 1.0 / dim
    sigma = sigma_frac * (hi - lo)
    t_start = time.perf_counter()

    X = _init_pop(rng, n_pop, lo, hi, x0)
    f = objective.evaluate_many(X)
    hist, hist_e = [], []

    def tournament(k=3):
        idx = rng.integers(0, n_pop, size=(n_pop, k))
        return idx[np.arange(n_pop), np.argmax(f[idx], axis=1)]

    for _ in range(n_iter):
        elite = np.argsort(f)[-n_elite:]
        parents = X[tournament()]

        # BLX-alpha on shuffled parent pairs
        mates = parents[rng.permutation(n_pop)]
        cmin = np.minimum(parents, mates)
        cmax = np.maximum(parents, mates)
        span = cmax - cmin
        child = rng.uniform(cmin - alpha * span, cmax + alpha * span)
        do_cross = rng.random((n_pop, 1)) < p_cross
        child = np.where(do_cross, child, parents)

        # Gaussian mutation
        mut = rng.random((n_pop, dim)) < p_mut
        child = np.where(mut, child + rng.normal(0, sigma, (n_pop, dim)), child)
        child = np.clip(child, lo, hi)

        child[:n_elite] = X[elite]                 # carry elites unchanged
        fc = objective.evaluate_many(child)
        X, f = child, fc

        hist.append(float(f.max()))
        hist_e.append(objective.n_evals)

    i = int(np.argmax(f))
    return Result("GA", X[i].copy(), float(f[i]), hist, hist_e,
                  objective.n_evals, objective.t_eval,
                  time.perf_counter() - t_start, n_pop)


# --------------------------------------------------------------------------
# PSO -- inertia weight linearly decreased, velocity clamped
# --------------------------------------------------------------------------

def pso(objective, bounds, n_pop=30, n_iter=100, seed=0, x0=None,
        w_start=0.9, w_end=0.4, c1=1.49445, c2=1.49445, v_frac=0.2):
    import time
    rng = np.random.default_rng(seed)
    dim = objective.dim
    lo, hi = _bounds(bounds, dim)
    v_max = v_frac * (hi - lo)
    t_start = time.perf_counter()

    X = _init_pop(rng, n_pop, lo, hi, x0)
    V = rng.uniform(-v_max, v_max, size=(n_pop, dim))
    f = objective.evaluate_many(X)

    p_best, p_best_f = X.copy(), f.copy()
    g = int(np.argmax(f))
    g_best, g_best_f = X[g].copy(), float(f[g])
    hist, hist_e = [], []

    for it in range(n_iter):
        w = w_start - (w_start - w_end) * it / max(n_iter - 1, 1)
        r1 = rng.random((n_pop, dim))
        r2 = rng.random((n_pop, dim))
        V = np.clip(w * V
                    + c1 * r1 * (p_best - X)
                    + c2 * r2 * (g_best[None, :] - X), -v_max, v_max)
        X = np.clip(X + V, lo, hi)
        f = objective.evaluate_many(X)

        imp = f > p_best_f
        p_best[imp], p_best_f[imp] = X[imp], f[imp]
        g = int(np.argmax(p_best_f))
        if p_best_f[g] > g_best_f:
            g_best, g_best_f = p_best[g].copy(), float(p_best_f[g])

        hist.append(g_best_f)
        hist_e.append(objective.n_evals)

    return Result("PSO", g_best, g_best_f, hist, hist_e,
                  objective.n_evals, objective.t_eval,
                  time.perf_counter() - t_start, n_pop)


ALGORITHMS = {"TLBO": tlbo, "GA": ga, "PSO": pso}
