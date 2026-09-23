"""
The proposed controller: closed-form predictive bisector steering.

Two ideas, both cheap.

1. THE OPTIMUM IS ANALYTIC. For specular reflection the gain-maximising mirror
   normal is the bisector of the incident and reflected rays,

       n_k = unit( u_in,k + u_out,k )

   so there is nothing to search. Cost is O(K): one normalise per mirror. The
   metaheuristics in the base paper spend N_iter x N_pop x K evaluations looking
   for a quantity this expression returns directly.

2. SPEND THE SAVED BUDGET ON PREDICTING THE POSE, NOT ON SEARCHING. A controller
   still has to live with actuator latency -- the base paper states 10-50 ms for
   motorised mirrors (their p.8). Because the bisector is free, the whole latency
   budget can be used to aim at where the user WILL be:

       p_hat(t + tau) = p(t) + v_hat(t) tau

   and steer to p_hat. Cost is still O(K).

Honesty about the predictor
---------------------------
A simulator knows velocity exactly; a real controller estimates it. `sigma_pos`
and `sigma_vel` inject Gaussian position and velocity estimation error, and both
are swept. With zero error the predictive scheme is a genie bound; the swept
results are what should be quoted.

Orientation is held rather than extrapolated. The measured pose process has a
coherence time of order 0.5 s against a 10-50 ms actuator delay, so holding is
close to optimal and adds no parameters. Extrapolating tilt is future work.
"""

import time

import numpy as np

from channel import _unit


def bisector_normals(mirror_pos, u_in_sel, rx_pos):
    """
    Closed-form optimum for one receiver pose. O(K).

    mirror_pos (K, 3), u_in_sel (K, 3) unit mirror->LED for the chosen LED,
    rx_pos (3,). Returns (K, 3).
    """
    u_out = _unit(np.asarray(rx_pos, dtype=float)[None, :] - mirror_pos)
    return _unit(u_in_sel + u_out)


def predict_pose(pos, vel, tau, room, rng=None, sigma_pos=0.0, sigma_vel=0.0,
                 margin=0.05):
    """
    Constant-velocity extrapolation with optional estimation error.

    pos (T, 3), vel (T, 3) -> predicted (T, 3) for a horizon of `tau` seconds.
    """
    pos = np.asarray(pos, dtype=float)
    vel = np.asarray(vel, dtype=float)
    if rng is not None and (sigma_pos or sigma_vel):
        pos = pos + rng.normal(0.0, sigma_pos, pos.shape) * [1, 1, 0]
        vel = vel + rng.normal(0.0, sigma_vel, vel.shape) * [1, 1, 0]
    out = pos + vel * tau
    lx, ly, _ = room
    out[:, 0] = np.clip(out[:, 0], margin, lx - margin)
    out[:, 1] = np.clip(out[:, 1], margin, ly - margin)
    out[:, 2] = pos[:, 2]
    return out


def predictive_configs(mirror_pos, u_in_sel, pred_pos):
    """
    Mirror normals a predictive controller commands, for a whole trace. O(K T).

    Only the bisector is needed, so this avoids rebuilding the full (K, L, T)
    geometry tensor for the predicted poses -- that was the bottleneck.

    mirror_pos (K, 3), u_in_sel (K, 3), pred_pos (T, 3) -> (K, T, 3).

    `u_in_sel` is the per-mirror LED assignment taken from the true geometry.
    A real controller would derive it from its own pose estimate, but the
    assignment depends mainly on the fixed LED-to-mirror leg and is stable
    against the few centimetres of prediction error considered here.
    """
    mp = np.asarray(mirror_pos, dtype=float)
    u_out = _unit(np.asarray(pred_pos, dtype=float)[None, :, :] - mp[:, None, :])
    return _unit(u_in_sel[:, None, :] + u_out)


def predicted_trace(trace, tau, room, rng=None, sigma_pos=0.0, sigma_vel=0.0):
    """As predict_pose, but returning a Trace (kept for convenience)."""
    from dataclasses import replace
    return replace(trace, pos=predict_pose(trace.pos, trace.vel, tau, room,
                                           rng, sigma_pos, sigma_vel))


def time_bisector(mirror_pos, u_in_sel, rx_pos, repeats=2000):
    """
    Measured wall-clock for one closed-form configuration update.

    This is the number to put beside the metaheuristics' tau_compute. Report the
    CPU model with it.
    """
    mp = np.asarray(mirror_pos, dtype=float)
    t0 = time.perf_counter()
    for _ in range(repeats):
        bisector_normals(mp, u_in_sel, rx_pos)
    return (time.perf_counter() - t0) / repeats


def complexity_table(K, n_pop, n_iter):
    """
    Operation counts per configuration update, for the paper's complexity table.

    The metaheuristics evaluate a fitness function that is itself O(K L) per
    call, so their per-update cost carries the population and iteration factors
    on top of the same per-mirror work the closed form does once.
    """
    return {
        "closed-form bisector": {"order": "O(K)", "fitness_evals": 0,
                                 "per_update_K_ops": K},
        "TLBO": {"order": "O(N_iter N_pop K)",
                 "fitness_evals": 2 * n_pop * n_iter,
                 "per_update_K_ops": 2 * n_pop * n_iter * K},
        "GA": {"order": "O(N_iter N_pop K)",
               "fitness_evals": n_pop * n_iter,
               "per_update_K_ops": n_pop * n_iter * K},
        "PSO": {"order": "O(N_iter N_pop K)",
                "fitness_evals": n_pop * n_iter,
                "per_update_K_ops": n_pop * n_iter * K},
    }
