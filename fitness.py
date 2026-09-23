"""
Cached evaluation of both IRS reflection models, plus the base paper's
fitness function.

The geometry that does not depend on mirror orientation -- distances,
irradiance angles, the Lambertian factor -- is computed once. Each evaluation
then touches only the orientation-dependent term, which differs between the
two models:

  cosine   : cos(alpha) = n_hat . u_out                      (their Eq. 1 + 8)
  specular : cos(theta) * W(reflected ray -> PD)             (law of reflection)

W is a raised-cosine window over the angular half-width the PD subtends at the
mirror, floored at `accept_floor_deg` for mirror surface and actuator
imperfection. At perfect bisector alignment W = 1.
"""

import time

import numpy as np

from channel import Config, _unit, noise_variance, snr_linear, snr_db, los_gain
from mirrors import angles_to_normals, local_frames, normals_to_angles, pack, repair, unpack

MODELS = ("cosine", "specular")


class GeometryCache:
    def __init__(self, cfg: Config, mirror_pos, wall_normals, rx_pos, rx_normals,
                 model="specular", accept_floor_deg=0.5):
        if model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        self.cfg, self.model = cfg, model
        self.mirror_pos = np.asarray(mirror_pos, dtype=float)
        self.wall_normals = _unit(np.asarray(wall_normals, dtype=float))
        self.frames = local_frames(self.wall_normals)

        rx_pos = np.atleast_2d(np.asarray(rx_pos, dtype=float))
        rx_normals = _unit(np.atleast_2d(np.asarray(rx_normals, dtype=float)))
        self.rx_pos, self.rx_normals = rx_pos, rx_normals

        led = cfg.led_positions
        led_n = _unit(cfg.led_normal)
        self.K, self.L, self.R = len(self.mirror_pos), len(led), len(rx_pos)

        v_in = led[None, :, :] - self.mirror_pos[:, None, :]
        self.d1 = np.linalg.norm(v_in, axis=-1)
        self.u_in = v_in / self.d1[..., None]
        cos_phi = np.einsum("kld,d->kl", -self.u_in, led_n)

        v_out = rx_pos[None, :, :] - self.mirror_pos[:, None, :]
        self.d2 = np.linalg.norm(v_out, axis=-1)
        self.u_out = v_out / self.d2[..., None]
        cos_psi = np.einsum("krd,rd->kr", -self.u_out, rx_normals)

        inside = cos_psi >= np.cos(np.radians(cfg.fov_deg))
        cos_phi_v = np.where(cos_phi > 0, cos_phi, 0.0)
        cos_psi_v = np.where((cos_psi > 0) & inside, cos_psi, 0.0)

        dsum = self.d1[:, :, None] + self.d2[:, None, :]
        self.base = ((cfg.m + 1) * cfg.a_pd * cfg.mirror_rho
                     / (2 * np.pi * dsum ** 2)
                     * np.power(cos_phi_v, cfg.m)[:, :, None]
                     * cfg.t_filter * cfg.g_concentrator
                     * cos_psi_v[:, None, :])

        r_pd = np.sqrt(cfg.a_pd / np.pi)
        alpha = np.maximum(np.arctan(r_pd / self.d2), np.radians(accept_floor_deg))
        self.one_minus_cos_acc = np.maximum(1.0 - np.cos(alpha), 1e-12)

        self.h_los = los_gain(cfg, rx_pos, rx_normals).sum(axis=1)

    # ------------------------------------------------------------------
    def irs_gain(self, normals):
        n = np.asarray(normals, dtype=float)

        if self.model == "cosine":
            # Base paper Eq. (8): orientation enters only as n_hat . u_out.
            # The incoming ray direction is absent -- no law of reflection.
            cos_alpha = np.clip(np.einsum("krd,kd->kr", self.u_out, n), 0.0, None)
            return np.einsum("klr->r", self.base * cos_alpha[:, None, :])

        # specular
        cos_theta = np.einsum("kld,kd->kl", self.u_in, n)
        refl = 2.0 * cos_theta[:, :, None] * n[:, None, :] - self.u_in
        cos_delta = np.einsum("kld,krd->klr", refl, self.u_out)
        t = (1.0 - cos_delta) / self.one_minus_cos_acc[:, None, :]
        window = np.where(t < 1.0,
                          0.5 * (1.0 + np.cos(np.pi * np.sqrt(np.clip(t, 0, 1)))),
                          0.0)
        amp = self.base * np.clip(cos_theta, 0.0, None)[:, :, None]
        return np.einsum("klr->r", amp * window)

    def total_gain(self, normals):
        return self.h_los + self.irs_gain(normals)

    def snr_db(self, normals, ambient=None):
        return snr_db(self.cfg, self.total_gain(normals), ambient)

    def snr_linear(self, normals, ambient=None):
        return snr_linear(self.cfg, self.total_gain(normals), ambient)


# ----------------------------------------------------------------------
# analytic optima -- one per model, both closed form, neither needs search
# ----------------------------------------------------------------------

def analytic_normals(cache: GeometryCache, rx_index=0):
    """
    Optimal mirror normals for `cache.model`, in closed form.

      cosine   : n_hat = u_out           -- aim the normal at the receiver.
                 Maximises cos(alpha) = 1. Independent of LED position, which
                 is precisely the physics the model omits.
      specular : n_hat = bisector(u_in, u_out).  Depends on both legs, as
                 actual reflection must.
    """
    rx = cache.rx_pos[rx_index]
    u_out = _unit(rx[None, :] - cache.mirror_pos)
    if cache.model == "cosine":
        return u_out
    best = np.argmax(cache.base[:, :, rx_index], axis=1)
    u_in = cache.u_in[np.arange(cache.K), best]
    return _unit(u_in + u_out)


def analytic_vector(cache: GeometryCache, rx_index=0, max_angle=np.radians(90.0)):
    """The same optimum expressed as a decision vector the optimisers search."""
    n_hat = analytic_normals(cache, rx_index)
    omega, gamma = normals_to_angles(n_hat, cache.frames)
    return pack(*repair(omega, gamma, max_angle=max_angle))


# ----------------------------------------------------------------------

def pointing_footprint(cfg: Config, distance, accept_floor_deg=0.5):
    """Lateral spot on the receiver plane still collected by the PD."""
    r_pd = np.sqrt(cfg.a_pd / np.pi)
    alpha = max(np.arctan(r_pd / distance), np.radians(accept_floor_deg))
    return 2.0 * distance * np.tan(alpha)


def local_targets(center, radius, n):
    c = np.asarray(center, dtype=float).reshape(3)
    off = np.linspace(-radius, radius, n)
    dx, dy = np.meshgrid(off, off, indexing="xy")
    pts = np.tile(c, (dx.size, 1))
    pts[:, 0] += dx.ravel()
    pts[:, 1] += dy.ravel()
    return pts


class Objective:
    """
    Fitness wrapper that decodes, evaluates and COUNTS.

    mode="paper" is the base paper's Eq. (15):

        f(x) = w1 * SNR(x) - w2 * CoV(x),   SNR LINEAR, w1 = 0.7, w2 = 0.3

    Verified against their reported optimum: 0.7 * 10^(69.82/10) = 6.717e6,
    matching the stated best fitness of 6.717536e6 exactly. Using dB here is
    the easy mistake -- it changes the optimiser's behaviour completely.

    mode="single" returns SNR in dB at one receiver, for validation against
    the analytic optimum.
    """

    def __init__(self, cache: GeometryCache, mode="single",
                 max_angle=np.radians(90.0)):
        if mode not in ("single", "paper"):
            raise ValueError("mode must be 'single' or 'paper'")
        self.cache, self.mode = cache, mode
        self.cfg = cache.cfg
        self.max_angle = max_angle
        self.dim = 2 * cache.K
        self.n_evals = 0
        self.total_time = 0.0

    @property
    def t_eval(self):
        return self.total_time / self.n_evals if self.n_evals else float("nan")

    def normals_from(self, x):
        omega, gamma = repair(*unpack(x), max_angle=self.max_angle)
        return angles_to_normals(omega, gamma, self.cache.frames)

    def __call__(self, x):
        t0 = time.perf_counter()
        n = self.normals_from(x)
        if self.mode == "single":
            val = float(self.cache.snr_db(n)[0])
        else:
            h = self.cache.total_gain(n)
            lin = self.cache.snr_linear(n)
            c = float(np.std(h) / np.mean(h))
            val = float(self.cfg.w_snr * np.mean(lin) - self.cfg.w_cov * c)
        self.total_time += time.perf_counter() - t0
        self.n_evals += 1
        return val

    def evaluate_many(self, X):
        return np.array([self(x) for x in X])

    def reset_counters(self):
        self.n_evals = 0
        self.total_time = 0.0


class AssignmentObjective(Objective):
    """
    Search over mirror-to-target assignment instead of free angles.

    Needed only for the specular model: with a ~0.1 deg detector window, a
    random mirror orientation reflects into the PD with probability ~1e-5, so
    free-angle population search has nothing to climb. Under the cosine model
    the landscape is smooth and free-angle search works -- which is exactly why
    the base paper's TLBO converges.
    """

    def __init__(self, cache: GeometryCache, targets, mode="single",
                 max_angle=np.radians(90.0)):
        super().__init__(cache, mode=mode, max_angle=max_angle)
        self.targets = np.atleast_2d(np.asarray(targets, dtype=float))
        self.G = len(self.targets)
        self.dim = cache.K

        om = np.empty((cache.K, self.G))
        ga = np.empty((cache.K, self.G))
        for g, tgt in enumerate(self.targets):
            u_out = _unit(tgt[None, :] - cache.mirror_pos)
            if cache.model == "cosine":
                n_hat = u_out
            else:
                best = np.argmax(cache.base.max(axis=2), axis=1)
                n_hat = _unit(cache.u_in[np.arange(cache.K), best] + u_out)
            o, g_ = normals_to_angles(n_hat, cache.frames)
            om[:, g], ga[:, g] = repair(o, g_, max_angle=max_angle)
        self.OM, self.GA = om, ga

    def indices_from(self, x):
        return np.clip(np.round(np.asarray(x, dtype=float)), 0, self.G - 1).astype(int)

    def normals_from(self, x):
        idx = self.indices_from(x)
        k = np.arange(self.cache.K)
        return angles_to_normals(self.OM[k, idx], self.GA[k, idx], self.cache.frames)

    def angles_from(self, x):
        idx = self.indices_from(x)
        k = np.arange(self.cache.K)
        return self.OM[k, idx].copy(), self.GA[k, idx].copy()

    def all_to_target(self, g):
        return np.full(self.dim, float(g))
