"""
Diagnostic for the pose-knowledge result in Section 5.4 / Figure 6.

Finding: the 2.24-2.58 dB specular pose gain is produced by a degenerate
np.argmax in fitness.py:122, not by any modelled pose dependence.

    best = np.argmax(cache.base[:, :, rx_index], axis=1)

cache.base[k, l, r] carries cos_psi_v[k, r] as a factor. That factor is
constant across the LED index l, so for a mirror inside the field of view it
cannot change which l wins -- the LED choice is pose-independent there. For a
mirror OUTSIDE the field of view the factor is exactly 0, the whole slice is
zero, and np.argmax returns index 0: the controller silently falls back to
LED 0 instead of the geometrically best LED.

Under the face-up assumption roughly 357 of 400 mirrors are outside the FoV per
drop, so the fallback fires for most of the array. When the true tilted pose
puts those mirrors back inside the FoV they contribute with a bisector built
from the wrong incident leg. That, and only that, is the measured gap.

Run with the repo path as the first argument:
    python3 check_pose.py /path/to/oirs-vlc-reflection-model [n_drops]
"""

import sys
import numpy as np

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1200

from channel import Config, build_mirrors, snr_db, upward_normals, _unit
from diffuse import build_patches, diffuse_gain
from fitness import GeometryCache, analytic_normals
from orientation import sample_pd_normals

SEED = 20260922
cfg = Config()
mpos, mnrm = build_mirrors(cfg)
rng = np.random.default_rng(SEED)

lx, ly, _ = cfg.room
pos = np.stack([rng.uniform(0.25, lx - 0.25, N),
                rng.uniform(0.25, ly - 0.25, N),
                np.full(N, cfg.rx_plane_z)], axis=-1)
up = upward_normals(N)
patches = build_patches(cfg)
acts = {a: sample_pd_normals(a, N, rng) for a in ("sitting", "walking")}
h_diff = {a: diffuse_gain(cfg, pos, v, patches=patches) for a, v in acts.items()}


def normals_posefree(cache, rx_index=0):
    """analytic_normals with cos_psi removed from the LED argmax."""
    rx = cache.rx_pos[rx_index]
    u_out = _unit(rx[None, :] - cache.mirror_pos)
    if cache.model == "cosine":
        return u_out
    led_n = _unit(cache.cfg.led_normal)
    cos_phi = np.einsum("kld,d->kl", -cache.u_in, led_n)
    cos_phi_v = np.where(cos_phi > 0, cos_phi, 0.0)
    dsum = cache.d1 + cache.d2[:, rx_index][:, None]
    best = np.argmax(np.power(cos_phi_v, cache.cfg.m) / dsum ** 2, axis=1)
    return _unit(cache.u_in[np.arange(cache.K), best] + u_out)


def live_mean(s):
    return s[s > 0.0].mean()


print(f"N = {N} drops/case, seed {SEED}, K = {len(mpos)}, "
      f"L = {len(cfg.led_positions)}, FoV = {cfg.fov_deg:.0f} deg\n")
print(f"{'model':<10}{'activity':<9}{'published':>12}{'pose-free':>12}"
      f"{'LED flips':>11}{'FoV-caused':>12}")

for model in ("cosine", "specular"):
    for act, tilt in acts.items():
        g = np.empty(N); b = np.empty(N)
        gp = np.empty(N); bp = np.empty(N)
        flips = caused = 0
        for i in range(N):
            pt = GeometryCache(cfg, mpos, mnrm, pos[i][None, :],
                               tilt[i][None, :], model=model)
            pb = GeometryCache(cfg, mpos, mnrm, pos[i][None, :],
                               up[i][None, :], model=model)
            g[i] = pt.total_gain(analytic_normals(pt, 0))[0]
            b[i] = pt.total_gain(analytic_normals(pb, 0))[0]
            gp[i] = pt.total_gain(normals_posefree(pt, 0))[0]
            bp[i] = pt.total_gain(normals_posefree(pb, 0))[0]
            d = np.argmax(pt.base[:, :, 0], axis=1) != np.argmax(pb.base[:, :, 0], axis=1)
            dead_t = pt.base[:, :, 0].sum(axis=1) <= 0.0
            dead_b = pb.base[:, :, 0].sum(axis=1) <= 0.0
            flips += int(d.sum())
            caused += int((d & (dead_t != dead_b)).sum())
        hd = h_diff[act]
        pub = live_mean(snr_db(cfg, g + hd)) - live_mean(snr_db(cfg, b + hd))
        fix = live_mean(snr_db(cfg, gp + hd)) - live_mean(snr_db(cfg, bp + hd))
        pct = 100.0 * caused / flips if flips else 0.0
        print(f"{model:<10}{act:<9}{pub:11.2f} dB{fix:11.2f} dB"
              f"{flips:11d}{pct:11.1f}%")

print("\nThe pose-free column is the gap once the degenerate argmax is removed.")
print("The genie arm is unchanged by the substitution; only the blind arm moves,")
print("because it was being charged for an arbitrary array index, not for")
print("lacking pose knowledge.")
