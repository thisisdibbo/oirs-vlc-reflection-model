"""
Time-correlated user mobility and device pose.

Two processes, both needed before any latency claim can be made: a stale mirror
configuration only costs something if the channel has actually moved on.

Position -- random waypoint
    Standard indoor mobility model. The user picks a destination uniformly in
    the room, walks to it at constant speed, optionally pauses, repeats.

Orientation -- Ornstein-Uhlenbeck driving the measured marginals
    The MARGINAL distributions are the measured ones from Soltani et al.
    (see orientation.py) and are not altered here. What this module adds is
    TEMPORAL correlation, because independent draws per timestep would make the
    pose unpredictable by construction and would rig the comparison against any
    predictive scheme.

    Method: a standard OU process z(t) with unit stationary variance and
    correlation exp(-dt / tau_c); the walking (Gaussian) marginal is obtained
    directly as mu + sigma*z, and the sitting (Laplace) marginal by mapping
    u = Phi(z) through the Laplace quantile function. Azimuth is a wrapped
    random walk, whose stationary marginal is uniform as measured.

    tau_c, the orientation coherence time, is MY modelling parameter, not a
    measured one. It is swept rather than fixed -- the ratio of control latency
    to coherence time is the quantity the latency study is about. Check the
    autocorrelation figure in arXiv:1805.07999 for a measured anchor and cite
    it if you use one.
"""

from dataclasses import dataclass
from math import erf

import numpy as np

from orientation import POLAR_MODELS, pd_normals

DEFAULT_TAU_C = 0.5          # orientation coherence time [s] -- swept, not measured


@dataclass
class Trace:
    t: np.ndarray            # (T,)   time [s]
    pos: np.ndarray          # (T, 3) receiver position
    vel: np.ndarray          # (T, 3) receiver velocity [m/s]
    normals: np.ndarray      # (T, 3) unit PD normals
    theta: np.ndarray        # (T,)   elevation [rad]
    alpha: np.ndarray        # (T,)   azimuth [rad]
    speed: float
    activity: str
    tau_c: float

    def __len__(self):
        return len(self.t)


def _phi(z):
    """Standard normal CDF, via math.erf (no scipy dependency)."""
    return 0.5 * (1.0 + np.vectorize(erf)(z / np.sqrt(2.0)))


def _laplace_quantile(u, mu, b):
    """Inverse CDF of Laplace(mu, b)."""
    u = np.clip(u, 1e-12, 1 - 1e-12)
    return np.where(u < 0.5,
                    mu + b * np.log(2.0 * u),
                    mu - b * np.log(2.0 * (1.0 - u)))


def _ou(n, dt, tau_c, rng):
    """OU process, unit stationary variance, correlation exp(-dt/tau_c)."""
    rho = np.exp(-dt / tau_c)
    z = np.empty(n)
    z[0] = rng.normal()
    innov = rng.normal(size=n) * np.sqrt(1.0 - rho ** 2)
    for i in range(1, n):
        z[i] = rho * z[i - 1] + innov[i]
    return z


def correlated_orientation(n, dt, activity, tau_c, rng,
                           azimuth_rate_rad_s=0.6):
    """
    Time series of (theta, alpha) whose marginals match orientation.py.

    azimuth_rate sets how fast the device heading wanders; its stationary
    marginal is uniform regardless of the rate.
    """
    kind, mu, sigma = POLAR_MODELS[activity]
    z = _ou(n, dt, tau_c, rng)

    if kind == "gaussian":
        theta_deg = mu + sigma * z
    else:
        b = sigma / np.sqrt(2.0)
        theta_deg = _laplace_quantile(_phi(z), mu, b)
    # measured support is [0, 90] deg; clipping touches <1e-5 of samples here
    theta = np.radians(np.clip(theta_deg, 0.0, 90.0))

    steps = rng.normal(0.0, azimuth_rate_rad_s * np.sqrt(dt), size=n)
    alpha = np.mod(rng.uniform(0, 2 * np.pi) + np.cumsum(steps), 2 * np.pi)
    return theta, alpha


def _in_footprint(xy, footprints, clearance=0.0):
    """True if a plan position falls inside any obstacle footprint."""
    for f in footprints:
        if (f[0] - clearance <= xy[0] <= f[1] + clearance
                and f[2] - clearance <= xy[1] <= f[3] + clearance):
            return True
    return False


def random_waypoint(n, dt, speed, room, rng, z_height, margin=0.3,
                    pause_prob=0.0, pause_steps=25, footprints=(),
                    clearance=0.15):
    """
    Random-waypoint positions and velocities, avoiding obstacle footprints.

    `footprints` is a sequence of [xmin, xmax, ymin, ymax].

    People walk around furniture. A walker that strolls through the sofa has
    every link occluded by construction, and excluding those samples afterwards
    biases the result -- the excluded positions are specific locations, not a
    random subset. With the base paper's Table 3 furniture that was 7.4% of
    timesteps, concentrated in the middle of the room.

    Targets inside a footprint are rejected, and a step that would enter one
    triggers a new target, so the walker deflects around obstacles.

    Returns pos (n, 3) and vel (n, 3). Velocity is zero while paused; it is
    what a predictive controller extrapolates from.
    """
    lx, ly, _ = room
    lo = np.array([margin, margin])
    hi = np.array([lx - margin, ly - margin])

    def draw_free():
        q = rng.uniform(lo, hi)
        for _ in range(200):
            if not _in_footprint(q, footprints, clearance):
                return q
            q = rng.uniform(lo, hi)
        return q                       # give up gracefully rather than hang

    pos = np.empty((n, 3))
    vel = np.zeros((n, 3))
    p = draw_free()
    target = draw_free()
    paused = 0

    for i in range(n):
        pos[i] = (p[0], p[1], z_height)
        if paused > 0:
            paused -= 1
            continue
        d = target - p
        dist = np.linalg.norm(d)
        step = speed * dt
        if dist <= step:
            p = target.copy()
            target = draw_free()
            if rng.random() < pause_prob:
                paused = pause_steps
        else:
            direction = d / dist
            nxt = p + direction * step
            if _in_footprint(nxt, footprints, clearance):
                target = draw_free()           # deflect around the obstacle
                continue
            p = nxt
            vel[i, :2] = direction * speed
    return pos, vel


def make_trace(cfg, duration, dt, speed, activity, rng,
               tau_c=DEFAULT_TAU_C, pause_prob=0.0, avoid_furniture=True):
    """
    Build a full Trace for one user.

    avoid_furniture=True routes the walker around cfg.blockers footprints.
    Set False only to reproduce the earlier, biased behaviour.
    """
    n = int(round(duration / dt))
    t = np.arange(n) * dt
    fp = ([[b[1], b[2], b[3], b[4]] for b in cfg.blockers]
          if avoid_furniture else ())
    pos, vel = random_waypoint(n, dt, speed, cfg.room, rng, cfg.rx_plane_z,
                               pause_prob=pause_prob, footprints=fp)
    theta, alpha = correlated_orientation(n, dt, activity, tau_c, rng)
    return Trace(t=t, pos=pos, vel=vel, normals=pd_normals(theta, alpha),
                 theta=theta, alpha=alpha, speed=speed, activity=activity,
                 tau_c=tau_c)


def predict(trace, i, horizon_steps):
    """
    Constant-velocity / persistence prediction of the pose `horizon_steps`
    ahead of index i. This is the cheap predictor the O(K) steering rule uses;
    upgrade to a Kalman filter later and report both.
    """
    dt = trace.t[1] - trace.t[0]
    tau = horizon_steps * dt
    pos_hat = trace.pos[i] + trace.vel[i] * tau
    lx, ly, _ = (trace.pos[:, 0].max(), trace.pos[:, 1].max(), 0)
    pos_hat = np.array([np.clip(pos_hat[0], 0.05, 4.95),
                        np.clip(pos_hat[1], 0.05, 4.95),
                        trace.pos[i, 2]])
    return pos_hat, trace.normals[i]          # pose held, position extrapolated
