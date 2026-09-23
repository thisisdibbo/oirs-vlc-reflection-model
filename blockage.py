"""
Occlusion by furniture and people.

Geometry straight from the base paper's Table 3, which gives each obstacle as
[x_min, x_max, y_min, y_max, height, reflectivity]:

    table  [2.0, 3.0, 2.0, 3.0, 0.75, 0.5]
    sofa   [0.5, 1.5, 0.5, 2.0, 0.85, 0.3]
    human  [3.0, 3.5, 3.0, 3.5, 1.70, 0.2]

They are axis-aligned boxes standing on the floor, and in the base paper all
three are STATIC. The extension here is a human who walks.

Three link families can be occluded and all three are tested:
    LED  -> mirror   (blocked light never reaches the IRS at all)
    mirror -> PD     (the reflected beam is intercepted)
    LED  -> PD       (the direct path, which is what the base paper blocks)

Reflectivity from Table 3 is carried but not yet used: these surfaces also
scatter, and treating an obstacle as a pure absorber is pessimistic. Noted as
a limitation rather than silently ignored.
"""

import numpy as np


def boxes_from_config(cfg, exclude=()):
    """
    (N, 6) array of [xmin, xmax, ymin, ymax, zmin, zmax] from cfg.blockers.

    `exclude` drops obstacles by name, e.g. exclude=("human",) when the human
    is going to be re-added as a mobile blocker.
    """
    out = []
    for name, x0, x1, y0, y1, h, _rho in cfg.blockers:
        if name in exclude:
            continue
        out.append([x0, x1, y0, y1, 0.0, h])
    return np.array(out, dtype=float).reshape(-1, 6)


def walking_box(n, dt, speed, room, rng, footprint=0.5, height=1.70,
                margin=0.4):
    """
    A person-sized box following a random waypoint. Returns (n, 6).

    footprint is the side length of the box in plan; height matches the Table 3
    human (1.70 m).
    """
    lx, ly, _ = room
    lo = np.array([margin, margin])
    hi = np.array([lx - margin, ly - margin])
    p = rng.uniform(lo, hi)
    target = rng.uniform(lo, hi)
    half = footprint / 2.0

    out = np.empty((n, 6))
    for i in range(n):
        out[i] = [p[0] - half, p[0] + half, p[1] - half, p[1] + half, 0.0, height]
        d = target - p
        dist = np.linalg.norm(d)
        step = speed * dt
        if dist <= step:
            p, target = target.copy(), rng.uniform(lo, hi)
        else:
            p = p + (d / dist) * step
    return out


def inside_any(pos, boxes):
    """
    True where a point lies inside an obstacle's volume.

    A random-waypoint user will otherwise walk straight through the sofa, and
    every link from a receiver inside a box starts occluded, which is not a
    blockage result but a mobility-model artefact. Timesteps flagged here are
    excluded and the excluded fraction is reported.
    """
    p = np.atleast_2d(np.asarray(pos, dtype=float))
    out = np.zeros(len(p), dtype=bool)
    for b in boxes:
        out |= ((p[:, 0] >= b[0]) & (p[:, 0] <= b[1])
                & (p[:, 1] >= b[2]) & (p[:, 1] <= b[3])
                & (p[:, 2] >= b[4]) & (p[:, 2] <= b[5]))
    return out


def segments_blocked(P, Q, boxes):
    """
    Slab test, vectorised.

    P, Q broadcast to a common shape (..., 3); boxes is (N, 6). Returns a
    boolean array of shape (...) that is True where the segment P->Q enters at
    least one box.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    P, Q = np.broadcast_arrays(P, Q)
    d = Q - P
    blocked = np.zeros(P.shape[:-1], dtype=bool)
    if boxes.size == 0:
        return blocked

    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d                                    # inf where d == 0
    for b in boxes:
        lo = np.array([b[0], b[2], b[4]])
        hi = np.array([b[1], b[3], b[5]])
        t1 = (lo - P) * inv
        t2 = (hi - P) * inv
        # an axis with d == 0 gives nan; replace so it never constrains
        tmin_ax = np.where(np.isnan(np.minimum(t1, t2)), -np.inf,
                           np.minimum(t1, t2))
        tmax_ax = np.where(np.isnan(np.maximum(t1, t2)), np.inf,
                           np.maximum(t1, t2))
        # a ray parallel to a slab and outside it can never hit
        parallel_out = (d == 0) & ((P < lo) | (P > hi))
        tmin = tmin_ax.max(axis=-1)
        tmax = tmax_ax.min(axis=-1)
        hit = (tmax >= np.maximum(tmin, 0.0)) & (tmin <= 1.0) \
            & ~parallel_out.any(axis=-1)
        blocked |= hit
    return blocked


def link_masks(cfg, mirror_pos, rx_pos, boxes_static, boxes_t=None):
    """
    Visibility masks for one trace.

    boxes_static : (N, 6) obstacles that never move
    boxes_t      : (T, 6) one moving obstacle per timestep, or None

    Returns dict with
        led_mirror : (K, L)   or (K, L, T) if a mobile blocker is present
        mirror_rx  : (K, T)
        led_rx     : (L, T)
    """
    mp = np.asarray(mirror_pos, dtype=float)
    rx = np.asarray(rx_pos, dtype=float)
    led = cfg.led_positions
    K, L, T = len(mp), len(led), len(rx)

    out = {}

    # --- LED -> mirror ---
    P = np.broadcast_to(led[None, :, :], (K, L, 3))
    Q = np.broadcast_to(mp[:, None, :], (K, L, 3))
    stat = segments_blocked(P, Q, boxes_static)
    if boxes_t is None:
        out["led_mirror"] = ~stat
    else:
        m = np.repeat(stat[:, :, None], T, axis=2)
        for t in range(T):
            m[:, :, t] |= segments_blocked(P, Q, boxes_t[t:t + 1])
        out["led_mirror"] = ~m

    # --- mirror -> PD ---
    P = np.broadcast_to(mp[:, None, :], (K, T, 3))
    Q = np.broadcast_to(rx[None, :, :], (K, T, 3))
    m = segments_blocked(P, Q, boxes_static)
    if boxes_t is not None:
        for t in range(T):
            m[:, t] |= segments_blocked(mp, np.broadcast_to(rx[t], (K, 3)),
                                        boxes_t[t:t + 1])
    out["mirror_rx"] = ~m

    # --- LED -> PD (the direct path) ---
    P = np.broadcast_to(led[:, None, :], (L, T, 3))
    Q = np.broadcast_to(rx[None, :, :], (L, T, 3))
    m = segments_blocked(P, Q, boxes_static)
    if boxes_t is not None:
        for t in range(T):
            m[:, t] |= segments_blocked(led, np.broadcast_to(rx[t], (L, 3)),
                                        boxes_t[t:t + 1])
    out["led_rx"] = ~m
    return out
