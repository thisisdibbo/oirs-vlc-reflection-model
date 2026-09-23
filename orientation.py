"""
Random device orientation for the receiver.

Measured model, NOT invented here. Parameters from:

  M. D. Soltani, A. A. Purwita, Z. Zeng, H. Haas & M. Safari, "Modeling the
  Random Orientation of Mobile Devices: Measurement, Analysis and LiFi Use
  Case", IEEE Trans. Wireless Commun. 18(3), 2019. arXiv:1805.07999

  Numerical values as tabulated in M. D. Soltani et al., "Measurements-Based
  Channel Models for Indoor LiFi Systems", arXiv:2001.09596.

  Elevation angle theta (tilt of the PD normal away from vertical):
    sitting  (stationary) : truncated Laplace,  mu = 41.39 deg, sigma = 7.68 deg
    walking  (mobile)     : truncated Gaussian, mu = 29.67 deg, sigma = 7.78 deg
  Azimuth alpha: uniform on [0, 2*pi) for both.

Reviewers in this area check these numbers. Do not substitute your own.

Why it matters here
-------------------
The base paper assumes a face-up photodiode throughout. A sitting user's device
is tilted ~41 degrees on average, and the PD field of view is 60 degrees
(their Table 3), so a large fraction of arriving rays land near or outside the
FoV edge. Two steering regimes are therefore distinguished:

  "genie"  -- the IRS controller knows the true device pose and steers to it.
  "blind"  -- the controller assumes the device faces up, as the base paper
              implicitly does, while the device is actually tilted.

The gap between them is the cost of ignoring orientation, and it is separate
from (and additive to) the control-latency cost.
"""

import numpy as np

# activity -> (distribution, mean_deg, std_deg)
POLAR_MODELS = {
    "sitting": ("laplace", 41.39, 7.68),
    "walking": ("gaussian", 29.67, 7.78),
}

THETA_LIMITS_DEG = (0.0, 90.0)   # physical truncation of the elevation angle


def sample_polar(activity, n, rng, limits_deg=THETA_LIMITS_DEG):
    """
    Draw n elevation angles [rad] for the given activity.

    Truncation is by rejection, which is exact (no CDF inversion error) and
    needs no scipy. Both distributions are narrow relative to the [0, 90]
    window, so acceptance is effectively 100% and this is not slow.
    """
    if activity not in POLAR_MODELS:
        raise ValueError(f"activity must be one of {sorted(POLAR_MODELS)}")
    kind, mu, sigma = POLAR_MODELS[activity]
    lo, hi = limits_deg

    out = np.empty(0)
    while out.size < n:
        need = int((n - out.size) * 1.3) + 16
        if kind == "laplace":
            # Laplace scale b relates to std as sigma = b * sqrt(2)
            draw = rng.laplace(mu, sigma / np.sqrt(2.0), size=need)
        else:
            draw = rng.normal(mu, sigma, size=need)
        out = np.concatenate([out, draw[(draw >= lo) & (draw <= hi)]])
    return np.radians(out[:n])


def sample_orientation(activity, n, rng):
    """Return (theta, alpha) in radians: elevation and azimuth."""
    theta = sample_polar(activity, n, rng)
    alpha = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return theta, alpha


def pd_normals(theta, alpha):
    """
    Unit PD normals from elevation and azimuth.

    theta = 0 is face-up (the base paper's assumption throughout).
    """
    theta = np.atleast_1d(theta)
    alpha = np.atleast_1d(alpha)
    return np.stack([np.sin(theta) * np.cos(alpha),
                     np.sin(theta) * np.sin(alpha),
                     np.cos(theta)], axis=-1)


def sample_pd_normals(activity, n, rng):
    """Convenience: draw n random PD normals for the given activity."""
    return pd_normals(*sample_orientation(activity, n, rng))


def describe():
    """One-line summary per activity, for printing into a results log."""
    lines = []
    for act, (kind, mu, sd) in POLAR_MODELS.items():
        lines.append(f"{act:<8} truncated {kind:<8} "
                     f"mu = {mu:5.2f} deg, sigma = {sd:4.2f} deg, "
                     f"azimuth uniform [0, 2pi)")
    return "\n".join(lines)
