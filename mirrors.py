"""
Mirror orientation parametrisation for the optical IRS.

Each mirror element carries two angles, roll (omega) and yaw (gamma), defined
as the direction cosines of the mirror normal along the two in-wall axes:

    n_hat = sin(omega) * e1 + sin(gamma) * e2 + sqrt(1 - sin^2 w - sin^2 g) * e3

where (e1, e2, e3) is the mirror's local frame and e3 is the inward wall
normal.  Two properties make this the right choice here:

  * it is exactly invertible -- omega = asin(n . e1), gamma = asin(n . e2) --
    so the analytic bisector optimum can be written in the SAME coordinates the
    optimisers search in.  That is what makes the TLBO-vs-bisector validation
    and the slew-time model possible.
  * the third component is non-negative by construction, so every feasible
    angle pair yields a mirror that faces into the room.

Feasibility requires sin^2(omega) + sin^2(gamma) <= 1; `repair` projects any
violating pair back onto that disc.
"""

import numpy as np

from channel import _unit


def local_frames(wall_normals):
    """
    Build an orthonormal frame per mirror from its inward wall normal.

    e3 = inward wall normal, e2 = vertical in-wall axis (world +z for side
    walls), e1 = e2 x e3 completes a right-handed frame.

    wall_normals (K, 3) -> e1, e2, e3 each (K, 3)
    """
    e3 = _unit(np.asarray(wall_normals, dtype=float))
    up = np.tile(np.array([0.0, 0.0, 1.0]), (e3.shape[0], 1))

    # degenerate only for floor/ceiling-mounted mirrors; fall back to +x
    par = np.abs(np.einsum("kd,kd->k", e3, up)) > 0.99
    up[par] = np.array([1.0, 0.0, 0.0])

    e2 = _unit(up - np.einsum("kd,kd->k", up, e3)[:, None] * e3)
    e1 = np.cross(e2, e3)
    return e1, e2, e3


def repair(omega, gamma, max_angle=np.radians(75.0)):
    """Clip to the box, then project onto the feasible disc sin^2 w + sin^2 g <= 1."""
    omega = np.clip(omega, -max_angle, max_angle)
    gamma = np.clip(gamma, -max_angle, max_angle)
    s = np.sin(omega) ** 2 + np.sin(gamma) ** 2
    bad = s > 1.0
    if np.any(bad):
        scale = np.sqrt(0.999 / s[bad])
        omega[bad] = np.arcsin(np.clip(np.sin(omega[bad]) * scale, -1, 1))
        gamma[bad] = np.arcsin(np.clip(np.sin(gamma[bad]) * scale, -1, 1))
    return omega, gamma


def angles_to_normals(omega, gamma, frames):
    """(K,) roll + (K,) yaw -> (K, 3) unit mirror normals."""
    e1, e2, e3 = frames
    a = np.sin(omega)
    b = np.sin(gamma)
    c = np.sqrt(np.clip(1.0 - a ** 2 - b ** 2, 0.0, None))
    return a[:, None] * e1 + b[:, None] * e2 + c[:, None] * e3


def normals_to_angles(normals, frames):
    """(K, 3) unit normals -> (omega, gamma). Exact inverse of angles_to_normals."""
    e1, e2, _ = frames
    n = _unit(np.asarray(normals, dtype=float))
    omega = np.arcsin(np.clip(np.einsum("kd,kd->k", n, e1), -1, 1))
    gamma = np.arcsin(np.clip(np.einsum("kd,kd->k", n, e2), -1, 1))
    return omega, gamma


def pack(omega, gamma):
    """(K,) + (K,) -> flat decision vector of length 2K."""
    return np.concatenate([omega, gamma])


def unpack(x):
    """Flat decision vector -> (omega, gamma)."""
    k = len(x) // 2
    return np.asarray(x[:k], dtype=float).copy(), np.asarray(x[k:], dtype=float).copy()


def quantize(omega, gamma, bits, max_angle=np.radians(75.0)):
    """
    Round both angles onto a b-bit uniform grid over [-max_angle, max_angle].

    This is the knob for Backup Idea A (required actuator precision) and for
    the quantised-steering realism check in the main paper.
    """
    levels = 2 ** int(bits)
    step = 2 * max_angle / (levels - 1)
    q = lambda v: np.round((v + max_angle) / step) * step - max_angle
    return q(np.clip(omega, -max_angle, max_angle)), q(np.clip(gamma, -max_angle, max_angle))


def slew_time(omega_from, gamma_from, omega_to, gamma_to, rate_deg_s):
    """
    Seconds for the slowest mirror to reach its new pose.

    tau_slew = max_k |delta Omega_k| / Omega_dot_max   -- the plan's equation 6.
    """
    d = np.maximum(np.abs(omega_to - omega_from), np.abs(gamma_to - gamma_from))
    return float(np.degrees(np.max(d)) / rate_deg_s)


def pointing_error(omega, gamma, sigma_deg, rng):
    """Add i.i.d. Gaussian pointing error to every mirror (Backup Idea A)."""
    s = np.radians(sigma_deg)
    return repair(omega + rng.normal(0, s, omega.shape),
                  gamma + rng.normal(0, s, gamma.shape))
