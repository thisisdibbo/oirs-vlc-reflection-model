"""
Vectorised evaluation of a mobility trace.

The per-timestep GeometryCache used on Day 5 rebuilt the whole geometry 1500
times per run, which made multi-seed averaging impractical. Everything here is
built once as (K, L, T) tensors and evaluated with a handful of einsums, which
is ~100x faster and makes 20-seed runs routine.

The key algebraic shortcut for the specular model avoids ever forming the
(K, L, T, 3) reflected-ray tensor:

    refl . u_out = 2 (u_in . n)(n . u_out) - (u_in . u_out)

so only (K, L, T) arrays are needed.

Results are identical to the per-step path; `selftest.py` checks that.
"""

import numpy as np

from channel import Config, _unit, los_gain, snr_db


class TraceEvaluator:
    """
    Pre-computes trace geometry and the analytic optimum at every timestep.

    gains(lag) returns the total channel gain at times t = lag .. T-1 when the
    configuration computed at t - lag is applied at t.
    """

    def __init__(self, cfg: Config, mirror_pos, wall_normals, trace,
                 model="specular", accept_floor_deg=0.5, masks=None,
                 diffuse=False, patches=None, dtype=np.float64):
        """
        masks: optional dict from blockage.link_masks. Blocked links are zeroed
        in `base` and in the LoS term BEFORE the analytic optimum is computed,
        so a blockage-aware controller sees the occlusion. Under the cosine
        model the optimum is n = u_out regardless, so it cannot react to
        blockage either -- another steering decision that model cannot express.
        """
        self.cfg, self.model = cfg, model
        led = cfg.led_positions
        led_n = _unit(cfg.led_normal)

        mp = np.asarray(mirror_pos, dtype=dtype)
        pos = np.asarray(trace.pos, dtype=dtype)
        nrm = _unit(np.asarray(trace.normals, dtype=dtype))
        K, L, T = len(mp), len(led), len(pos)
        self.K, self.L, self.T = K, L, T

        # --- LED -> mirror (fixed over the trace) ---
        v_in = led[None, :, :] - mp[:, None, :]
        d1 = np.linalg.norm(v_in, axis=-1)
        self.u_in = (v_in / d1[..., None]).astype(dtype)          # (K,L,3)
        cos_phi = np.einsum("kld,d->kl", -self.u_in, led_n)

        # --- mirror -> receiver (varies with t) ---
        v_out = pos[None, :, :] - mp[:, None, :]
        d2 = np.linalg.norm(v_out, axis=-1)
        self.u_out = (v_out / d2[..., None]).astype(dtype)        # (K,T,3)
        cos_psi = np.einsum("ktd,td->kt", -self.u_out, nrm)

        inside = cos_psi >= np.cos(np.radians(cfg.fov_deg))
        cos_phi_v = np.where(cos_phi > 0, cos_phi, 0.0)
        cos_psi_v = np.where((cos_psi > 0) & inside, cos_psi, 0.0)

        dsum = d1[:, :, None] + d2[:, None, :]                    # (K,L,T)
        self.base = ((cfg.m + 1) * cfg.a_pd * cfg.mirror_rho
                     / (2 * np.pi * dsum ** 2)
                     * np.power(cos_phi_v, cfg.m)[:, :, None]
                     * cfg.t_filter * cfg.g_concentrator
                     * cos_psi_v[:, None, :]).astype(dtype)

        r_pd = np.sqrt(cfg.a_pd / np.pi)
        alpha = np.maximum(np.arctan(r_pd / d2), np.radians(accept_floor_deg))
        self.omc = np.maximum(1.0 - np.cos(alpha), 1e-12).astype(dtype)

        # --- apply occlusion before anything is optimised ---
        if masks is not None:
            lm = masks["led_mirror"]                       # (K,L) or (K,L,T)
            if lm.ndim == 2:
                lm = lm[:, :, None]
            self.base *= lm.astype(dtype)
            self.base *= masks["mirror_rx"][:, None, :].astype(dtype)

        los = los_gain(cfg, pos, nrm)                      # (T, L)
        if masks is not None:
            los = los * masks["led_rx"].T.astype(dtype)
        self.h_los = los.sum(axis=1)

        # First-order diffuse reflections do not depend on mirror orientation,
        # so they join the LoS term. They form the floor that remains when a
        # steered configuration goes stale.
        self.h_diffuse = np.zeros_like(self.h_los)
        if diffuse:
            from diffuse import diffuse_gain
            self.h_diffuse = diffuse_gain(cfg, pos, nrm, patches=patches)
            self.h_los = self.h_los + self.h_diffuse

        if model == "specular":
            self.C = np.einsum("kld,ktd->klt", self.u_in, self.u_out)
            best = np.argmax(self.base, axis=1)                   # (K,T)
            u_in_sel = np.take_along_axis(
                self.u_in[:, :, None, :], best[:, None, :, None], axis=1)[:, 0]
            self.N = _unit(u_in_sel + self.u_out)                 # (K,T,3)
        else:
            self.Bsum = self.base.sum(axis=1)                     # (K,T)
            self.N = self.u_out.copy()

    # ------------------------------------------------------------------
    def gains(self, lag=0, N=None):
        """
        Total channel gain for t in [lag, T).

        By default the configuration applied at t is this trace's own analytic
        optimum from t - lag. Pass `N` (K, T, 3) to apply an EXTERNALLY computed
        configuration instead -- that is how a predictive controller is scored:
        configs derived from a predicted pose, evaluated against the true one.
        """
        lag = int(lag)
        src = self.N if N is None else np.asarray(N, dtype=self.N.dtype)
        sl = slice(lag, self.T)
        nsl = slice(0, self.T - lag) if N is None else sl
        Nl = src[:, nsl, :]                                       # (K,T',3)
        uo = self.u_out[:, sl, :]

        if self.model == "cosine":
            cos_a = np.clip(np.einsum("ktd,ktd->kt", uo, Nl), 0.0, None)
            irs = np.einsum("kt->t", self.Bsum[:, sl] * cos_a)
        else:
            cos_th = np.einsum("kld,ktd->klt", self.u_in, Nl)     # (K,L,T')
            A = np.einsum("ktd,ktd->kt", uo, Nl)                  # (K,T')
            cos_d = 2.0 * cos_th * A[:, None, :] - self.C[:, :, sl]
            u = (1.0 - cos_d) / self.omc[:, None, sl]
            win = np.where(u < 1.0,
                           0.5 * (1.0 + np.cos(np.pi * np.sqrt(np.clip(u, 0, 1)))),
                           0.0)
            irs = np.einsum("klt->t", self.base[:, :, sl]
                            * np.clip(cos_th, 0.0, None) * win)
        return self.h_los[sl] + irs

    def snr_db(self, lag=0, N=None):
        return snr_db(self.cfg, self.gains(lag, N=N))
