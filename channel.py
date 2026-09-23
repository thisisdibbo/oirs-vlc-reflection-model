"""
Indoor VLC channel with a mirror-array optical IRS -- two reflection models.

Base paper (all parameters below traced to it):
  Dixit, Kumar, Sharan, Pandey, Kumar & Singh, "Optimizing intelligent
  reflecting surface assisted visible light communication networks under
  blockage and practical constraints using TLBO for IoT applications",
  Scientific Reports 15:27400 (2025). DOI 10.1038/s41598-025-12520-7

Values marked [T3] come from their Table 3, [F1] from Fig. 1, [p7] from the
text on page 7. Values marked [ASSUMED] are not stated anywhere in the paper
and are set from Ghassemlooy, Popoola & Rajbhandari, "Optical Wireless
Communications: System and Channel Modelling with MATLAB" (CRC Press).

THE TWO REFLECTION MODELS
-------------------------
"cosine"   -- the base paper's Eq. (1) with Eq. (8). Mirror orientation enters
              the gain ONLY through cos(alpha_lm) = n_hat . u_out, the angle
              between the mirror normal and the direction to the receiver. The
              incoming ray direction never appears: there is no law of
              reflection. A mirror is a cosine-lobe re-radiator, brightest when
              aimed at the user regardless of where the light came from.
              Optimum: n_hat = u_out. Closed form. Landscape smooth.

"specular" -- actual specular reflection. The reflected ray 2(u_in.n)n - u_in
              must reach the PD, within the angular window the finite detector
              subtends. Optimum: n_hat = bisector(u_in, u_out). Closed form.
              Landscape has no usable gradient for population search.

Both optima are analytic. Neither needs a metaheuristic.
"""

from dataclasses import dataclass, field
import numpy as np

Q_E = 1.602176634e-19
K_B = 1.380649e-23


@dataclass
class Config:
    # ---- room and geometry ------------------------------------------------
    room: tuple = (5.0, 5.0, 3.0)                      # [T3]
    rx_plane_z: float = 0.80                           # [F1] H1 = 0.8 m

    led_positions: np.ndarray = field(default_factory=lambda: np.array([
        [1.25, 1.25, 3.0],      # S1 [T3]
        [1.25, 3.75, 3.0],      # S2 [T3]
        [3.75, 1.25, 3.0],      # S3 [T3]
        [3.75, 3.75, 3.0],      # S4 [T3]
    ]))
    led_normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))
    # [CALIBRATED] P_t^Avg is defined in their Table 2 but its value appears
    # nowhere in the paper. Set so the conventional peak SNR matches the ~51 dB
    # maximum of their Fig. 3a. Their corner minimum (43 dB) is still ~5 dB
    # above ours: our LoS-only map spans 12 dB across the full room against
    # their ~10 dB, so no single P_t reproduces both ends. Report this.
    # The cosine-vs-specular overstatement is insensitive to this choice.
    p_led: float = 1.75
    half_angle_deg: float = 60.0                       # [T3] phi_1/2

    # ---- receiver ---------------------------------------------------------
    a_pd: float = 1e-4                                 # [T3] 1 cm^2
    fov_deg: float = 60.0                              # [T3] phi_c
    responsivity: float = 0.54                         # [ASSUMED]
    n_concentrator: float = 1.5                        # [T3] eta
    t_filter: float = 1.0                              # [ASSUMED] G_F never stated

    # ---- IRS panels -------------------------------------------------------
    # [T3] panel CENTRES; each carries a 10 x 10 element grid.
    # Panel physical extent is NOT stated in the paper -- sweep `panel_size`.
    panel_centres: np.ndarray = field(default_factory=lambda: np.array([
        [2.5, 5.0, 1.5],        # R1, wall y = 5
        [0.0, 2.5, 1.5],        # R2, wall x = 0
        [2.5, 0.0, 1.5],        # R3, wall y = 0
        [5.0, 2.5, 1.5],        # R4, wall x = 5
    ]))
    mirrors_per_panel: int = 10                        # [T3] 10 x 10 per panel
    panel_size: tuple = (1.0, 1.0)                     # [ASSUMED] width, height [m]
    mirror_rho: float = 0.9                            # [T3] delta

    # ---- electrical front end ---------------------------------------------
    bandwidth: float = 50e6                            # [T3] B = 50 MHz
    # p.6: the paper models shot + thermal only and treats ambient light as a
    # separate +0.49 dB aside. Default OFF to reproduce them; switch on for P4.
    ambient_default: bool = False
    i_bg: float = 740e-6                               # [ASSUMED]
    i2: float = 0.562                                  # [ASSUMED]
    i3: float = 0.0868                                 # [ASSUMED]
    temp_k: float = 295.0                              # [ASSUMED]
    g_ol: float = 10.0                                 # [ASSUMED]
    c_pd_per_cm2: float = 112e-12                      # [ASSUMED]
    gamma_fet: float = 1.5                             # [ASSUMED]
    gm_fet: float = 30e-3                              # [ASSUMED]

    # ---- optimiser search space, [p7] -------------------------------------
    # "x in [0,180]^l x [0,90]^l, first l components polar (yaw) in [0,90],
    #  next l components azimuth (roll) in [0,180]"
    polar_bounds_deg: tuple = (0.0, 90.0)
    azimuth_bounds_deg: tuple = (0.0, 180.0)

    # ---- their fitness, Eq. (15), [T3] weights ----------------------------
    # f(x) = w1 * SNR(x) - w2 * CoV(x), with SNR in LINEAR units.
    # Verified: 0.7 * 10^(69.82/10) = 6.717e6 = their reported best fitness.
    w_snr: float = 0.7
    w_cov: float = 0.3

    # ---- furniture / blockage, [T3] [xmin, xmax, ymin, ymax, h, rho] ------
    blockers: tuple = (
        ("table", 2.0, 3.0, 2.0, 3.0, 0.75, 0.5),
        ("sofa", 0.5, 1.5, 0.5, 2.0, 0.85, 0.3),
        ("human", 3.0, 3.5, 3.0, 3.5, 1.70, 0.2),
    )

    # ---- actuator limits, stated on p.8 of the base paper -----------------
    # "response times of conventional motorized mirror steering units are on
    #  the order of 10-50 ms ... These values determine realistic bounds for
    #  real-time IRS adaptation." They state it, then never use it.
    mirror_response_ms: tuple = (10.0, 50.0)
    mirror_switch_power_w: tuple = (0.1, 0.5)

    wall_rho: float = 0.8                              # [ASSUMED], for P4

    @property
    def m(self) -> float:
        """Lambertian emission order p, their Eq. (2)."""
        return -np.log(2.0) / np.log(np.cos(np.radians(self.half_angle_deg)))

    @property
    def g_concentrator(self) -> float:
        """Their Eq. (3)."""
        return self.n_concentrator ** 2 / np.sin(np.radians(self.fov_deg)) ** 2

    @property
    def c_pd(self) -> float:
        return self.c_pd_per_cm2 / 1e-4


# --------------------------------------------------------------------------

def _unit(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


def receiver_grid(cfg: Config, n: int = 25, margin: float = 0.25):
    lx, ly, _ = cfg.room
    xs = np.linspace(margin, lx - margin, n)
    ys = np.linspace(margin, ly - margin, n)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, cfg.rx_plane_z)], axis=-1)
    return X, Y, pts


def upward_normals(n_rx: int):
    v = np.zeros((n_rx, 3))
    v[:, 2] = 1.0
    return v


# --------------------------------------------------------------------------

def los_gain(cfg: Config, rx_pos, rx_normal):
    """Their Eq. (6) with Eq. (7). Returns (R, L)."""
    rx_pos = np.atleast_2d(rx_pos)
    rx_normal = _unit(np.atleast_2d(rx_normal))
    tx = cfg.led_positions

    d_vec = tx[None, :, :] - rx_pos[:, None, :]
    d = np.linalg.norm(d_vec, axis=-1)
    u = d_vec / d[..., None]

    cos_phi = np.einsum("rld,d->rl", -u, _unit(cfg.led_normal))
    cos_psi = np.einsum("rld,rd->rl", u, rx_normal)

    valid = (cos_phi > 0) & (cos_psi > 0) & (cos_psi >= np.cos(np.radians(cfg.fov_deg)))
    h = ((cfg.m + 1) * cfg.a_pd / (2 * np.pi * d ** 2)
         * np.power(np.clip(cos_phi, 0, None), cfg.m)
         * cfg.t_filter * cfg.g_concentrator
         * np.clip(cos_psi, 0, None))
    return np.where(valid, h, 0.0)


def build_mirrors(cfg: Config):
    """
    Element centres and inward wall normals for the four IRS panels.

    Panels are COMPACT, centred on the [T3] coordinates -- not spread across
    the walls. Their extent is unstated, so it comes from cfg.panel_size.
    """
    lx, ly, _ = cfg.room
    n = cfg.mirrors_per_panel
    w, h = cfg.panel_size
    du = np.linspace(-w / 2, w / 2, n)
    dv = np.linspace(-h / 2, h / 2, n)
    U, V = np.meshgrid(du, dv, indexing="xy")

    pos, nrm = [], []
    for c in cfg.panel_centres:
        if np.isclose(c[1], 0.0):                       # wall y = 0
            p = np.stack([c[0] + U.ravel(), np.zeros(U.size), c[2] + V.ravel()], -1)
            v = np.array([0.0, 1.0, 0.0])
        elif np.isclose(c[1], ly):                      # wall y = Ly
            p = np.stack([c[0] + U.ravel(), np.full(U.size, ly), c[2] + V.ravel()], -1)
            v = np.array([0.0, -1.0, 0.0])
        elif np.isclose(c[0], 0.0):                     # wall x = 0
            p = np.stack([np.zeros(U.size), c[1] + U.ravel(), c[2] + V.ravel()], -1)
            v = np.array([1.0, 0.0, 0.0])
        else:                                           # wall x = Lx
            p = np.stack([np.full(U.size, lx), c[1] + U.ravel(), c[2] + V.ravel()], -1)
            v = np.array([-1.0, 0.0, 0.0])
        pos.append(p)
        nrm.append(np.tile(v, (p.shape[0], 1)))
    return np.concatenate(pos), np.concatenate(nrm)


# --------------------------------------------------------------------------

def noise_variance(cfg: Config, p_received, ambient=None):
    """Their Eq. (12): sigma^2 = sigma_S^2 + sigma_T^2."""
    if ambient is None:
        ambient = cfg.ambient_default
    p = np.asarray(p_received, dtype=float)
    b = cfg.bandwidth

    shot = 2 * Q_E * cfg.responsivity * p * b
    if ambient:
        shot = shot + 2 * Q_E * cfg.i_bg * cfg.i2 * b

    ca = cfg.c_pd * cfg.a_pd
    th = ((8 * np.pi * K_B * cfg.temp_k / cfg.g_ol) * ca * cfg.i2 * b ** 2
          + (16 * np.pi ** 2 * K_B * cfg.temp_k * cfg.gamma_fet / cfg.gm_fet)
          * ca ** 2 * cfg.i3 * b ** 3)
    return shot + th


def snr_linear(cfg: Config, h_total, ambient=None):
    h = np.asarray(h_total, dtype=float)
    p_rx = cfg.p_led * h
    sig = (cfg.responsivity * p_rx) ** 2
    return np.where(sig > 0, sig / noise_variance(cfg, p_rx, ambient), 1e-30)


def snr_db(cfg: Config, h_total, ambient=None):
    with np.errstate(divide="ignore"):
        return 10 * np.log10(snr_linear(cfg, h_total, ambient))


def ber_ook(snr_lin):
    from math import erfc
    s = np.asarray(snr_lin, dtype=float)
    return 0.5 * np.vectorize(erfc)(np.sqrt(np.clip(s, 0, None)) / np.sqrt(2))


def cov(values):
    v = np.asarray(values, dtype=float)
    return float(np.std(v) / np.mean(v))
