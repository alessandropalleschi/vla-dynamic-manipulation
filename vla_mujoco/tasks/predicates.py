"""Generic spatial predicates over entities (used for success checking and as training labels)."""

from __future__ import annotations

import numpy as np


def touching(env, a: str, b: str) -> bool:
    d = env.data
    ga, gb = env.boxes[a]["geom"], env.boxes[b]["geom"]
    for i in range(d.ncon):
        c = d.contact[i]
        if {int(c.geom1), int(c.geom2)} == {ga, gb} and c.dist < 1e-3:
            return True
    return False


def on(env, a: str, b: str) -> tuple[bool, float]:
    """Is ``a`` resting on top of ``b``? Returns (truth, xy centre offset in metres)."""
    pa, _ = env.entity_pose(a)
    pb, _ = env.entity_pose(b)
    off = float(np.linalg.norm(pa[:2] - pb[:2]))
    above = pa[2] - pb[2] > 0.8 * (env.ent[a].hz + env.ent[b].hz)
    return bool(touching(env, a, b) and above), off


def disturbed(env, name: str) -> bool:
    """Has a bystander been knocked over / moved noticeably from where it started?"""
    e = env.ent[name]
    p, R = env.entity_pose(name)
    p0 = env.init_pose[name]
    if R[2, 2] < 0.9:  # tipped over
        return True
    if e.location == "belt":
        u, _ = env.belt_coords(p[:2])
        u0, _ = env.belt_coords(p0[:2])
        return bool(abs(u - u0) > 0.05 or p[2] < p0[2] - 0.02)
    return bool(np.linalg.norm(p[:2] - p0[:2]) > 0.03 or abs(p[2] - p0[2]) > 0.02)


def support_height(env, xy, exclude: str | None = None) -> float:
    """Height of the highest surface under ``xy``: table, belt or the top of any entity footprint covering xy.

    Analytic for the box/cylinder primitives (a ray-cast would be the generalisation for meshes)."""
    u, _ = env.belt_coords(xy)
    z = env.belt_top if abs(u) < 0.15 else env.table_z
    for name, e in env.ent.items():
        if name == exclude:
            continue
        p, R = env.entity_pose(name)
        d = np.asarray(xy) - p[:2]
        if e.shape == "cylinder":
            inside = float(np.hypot(*d)) <= e.dims[0]
        else:
            lx, ly = R[:2, 0] @ d, R[:2, 1] @ d
            inside = abs(lx) <= e.dims[0] and abs(ly) <= e.dims[1]
        if inside:
            z = max(z, float(p[2] + e.hz))
    return float(z)
