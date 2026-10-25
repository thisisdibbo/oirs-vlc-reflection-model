import numpy as np

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from channel import Config, build_mirrors, upward_normals, _unit
from fitness import GeometryCache
from mirrors import angles_to_normals, normals_to_angles

cfg = Config()
mpos, mnrm = build_mirrors(cfg)
rng = np.random.default_rng(20260922)
lx, ly, _ = cfg.room
N_USERS = 12
users = np.stack([rng.uniform(0.25, lx - 0.25, N_USERS),
                  rng.uniform(0.25, ly - 0.25, N_USERS),
                  np.full(N_USERS, cfg.rx_plane_z)], axis=-1)


def per_mirror_gain(cache, n, best, r=0):
    k = np.arange(cache.K)
    u_in = cache.u_in[k, best]
    b = cache.base[k, best, r]
    cos_theta = np.einsum("kd,kd->k", u_in, n)
    refl = 2.0 * cos_theta[:, None] * n - u_in
    cos_delta = np.einsum("kd,kd->k", refl, cache.u_out[:, r, :])
    t = (1.0 - cos_delta) / cache.one_minus_cos_acc[:, r]
    w = np.where(t < 1.0, 0.5 * (1.0 + np.cos(np.pi * np.sqrt(np.clip(t, 0, 1)))), 0.0)
    return b * np.clip(cos_theta, 0.0, None) * w


def grid_gain(cache, om, ga, best, r=0):
    n = angles_to_normals(om, ga, cache.frames)
    return per_mirror_gain(cache, n, best, r)


print(f"{'alpha_min':>10}{'bisector':>14}{'refined':>14}{'gap [dB]':>12}"
      f"{'predicted':>12}")

for acc in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0):
    tot_bis = tot_ref = 0.0
    pred_max = 0.0
    for u in users:
        cache = GeometryCache(cfg, mpos, mnrm, u[None, :], upward_normals(1),
                              model="specular", accept_floor_deg=acc)
        best = np.argmax(cache.led_score[:, :, 0], axis=1)
        k = np.arange(cache.K)
        u_in = cache.u_in[k, best]
        n0 = _unit(u_in + cache.u_out[:, 0, :])
        om0, ga0 = normals_to_angles(n0, cache.frames)

        g_bis = per_mirror_gain(cache, n0, best)
        live = g_bis > 0.0

        best_g = g_bis.copy()
        span = np.radians(3.0 * max(acc, 0.5))
        for _ in range(4):
            offs = np.linspace(-span, span, 25)
            for do in offs:
                for dg in offs:
                    g = grid_gain(cache, om0 + do, ga0 + dg, best)
                    better = g > best_g
                    best_g = np.where(better, g, best_g)
            span /= 6.0

        tot_bis += g_bis.sum()
        tot_ref += best_g.sum()

        cos_th = np.einsum("kd,kd->k", u_in, n0)
        tan_th = np.sqrt(np.clip(1 - cos_th ** 2, 0, 1)) / np.clip(cos_th, 1e-9, None)
        a = np.radians(acc)
        pred_max = max(pred_max, float(np.max(a ** 2 * tan_th[live] ** 2 / (4 * np.pi ** 2))))

    gap_db = 10.0 * np.log10(tot_ref / tot_bis)
    print(f"{acc:9.2f} deg{tot_bis:14.4e}{tot_ref:14.4e}{gap_db:12.5f}"
          f"{10 * np.log10(1 + pred_max):12.5f}")

print("\npredicted = 10log10(1 + alpha^2 tan^2(theta_i) / 4pi^2), worst mirror")
