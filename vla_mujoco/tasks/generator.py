"""Sample a scene layout (entities on the belt and table), assign task roles and generate language.

Task kinds
    stack          pick a table object, put it on a belt object
    belt_to_table  pick a moving belt object, put it on a table object
    belt_to_belt   pick one belt object, put it on another belt object
    tower          pick two table objects; put the first on a belt object, then the second on top of the first

Roles are chosen so that the task is reachable in time (arrival windows depend on the belt speed) and the
instruction uniquely identifies every referenced object; otherwise the layout is resampled.
"""

from __future__ import annotations

import math

import numpy as np

from ..lang import describe, make_instruction
from ..sim.entities import PALETTE, Entity, sample_entity
from ..sim.scene import BELT_TOP, N_OBJ
from .tasks import Step, Task

KINDS = ("stack", "belt_to_table", "belt_to_belt", "tower")
CATEGORY_P = {"cube": 0.25, "block": 0.25, "cylinder": 0.25, "slab": 0.125, "disc": 0.125}
T_PICK = 6.0  # s: typical time until a belt object is picked
T_SUPPORT = 12.0  # s: a belt object used as support must reach the workspace within this time


def _sample_cat(rng):
    return str(rng.choice(list(CATEGORY_P), p=list(CATEGORY_P.values())))


def _make(rng, cfg, idx, color, location) -> Entity:
    if not cfg.randomize_shape:  # nominal scene: identical 10 cm cubes
        e = sample_entity(rng, idx, "cube", color, location)
        e.dims = (0.05, 0.05, 0.05)
        return e
    return sample_entity(rng, idx, _sample_cat(rng), color, location)


def _colors(rng, n):
    names = list(PALETTE)
    dup = rng.random() < 0.4  # sometimes repeat colours: language must then use other attributes
    return [str(c) for c in (rng.choice(names, n, replace=True) if dup else rng.choice(names, n, replace=False))]


def _table_layout(env, rng, ents: list[Entity]) -> bool:
    """Non-overlapping positions beside the belt, clear of the (rotated) rail."""
    placed: list[Entity] = []
    for e in ents:
        for _ in range(60):
            y = float(rng.uniform(-0.42, 0.42))
            rail_x = env.belt_to_world(-0.178, 0.0)[0] + y * math.tan(env.belt_yaw)
            xmax = min(0.31, rail_x - max(e.dims[:2]) - 0.04)
            if xmax < 0.24:
                continue
            x = float(rng.uniform(0.24, xmax))
            if all(abs(y - o.table_y) >= e.footprint + o.footprint + 0.06 for o in placed):
                e.table_y = y
                e.pos = np.array([x, y, env.table_z + e.hz + 0.0005])
                e.yaw = float(rng.uniform(-0.5, 0.5))
                placed.append(e)
                break
        else:
            return False
    return True


def _belt_layout(env, rng, ents: list[Entity]) -> None:
    """Items in arrival order (index 0 = most downstream) with random gaps; first item compensated for speed."""
    v = env.belt_speed
    s = float(rng.uniform(-0.55, -0.35) - 4.0 * max(0.0, v - 0.05))
    prev = None
    for e in ents:
        if prev is not None:
            s -= prev.footprint + e.footprint + 0.10 + float(rng.uniform(0.0, 0.14))
        u = float(rng.uniform(-0.03, 0.03))
        e.belt_s = s
        xy = env.belt_to_world(u, s)
        e.pos = np.array([xy[0], xy[1], BELT_TOP + e.hz + 0.0005])
        e.yaw = float(env.belt_yaw + rng.uniform(-0.15, 0.15))
        prev = e


def _eligible(env, e: Entity, horizon: float, lo: float, hi: float) -> bool:
    return lo <= e.belt_s + env.belt_speed * horizon <= hi


def _build(env, rng, kind, cfg):
    n_b = max(0, cfg.n_distractors)
    n_t = max(0, cfg.n_table_distractors)
    need_b = {"stack": 1, "belt_to_table": 1, "belt_to_belt": 2, "tower": 1}[kind]
    need_t = {"stack": 1, "belt_to_table": 1, "belt_to_belt": 0, "tower": 2}[kind]
    nb, nt = need_b + n_b, need_t + n_t
    while nb + nt > N_OBJ:  # respect the pool size, drop bystanders first
        if n_t > 0 and nt >= nb:
            nt -= 1
        else:
            nb -= 1
    nb = max(nb, need_b)
    nt = max(nt, need_t)
    colors = _colors(rng, nb + nt)
    belt = [_make(rng, cfg, i, colors[i], "belt") for i in range(nb)]
    table = [_make(rng, cfg, nb + i, colors[nb + i], "table") for i in range(nt)]
    _belt_layout(env, rng, belt)
    belt = [e for e in belt if e.belt_s > -1.05]  # items past the upstream end of the belt would fall off
    if len(belt) < need_b:
        return None
    if not _table_layout(env, rng, table):
        return None

    def support_ok(e):  # arrives in time and is not gone before we get there
        return _eligible(env, e, T_SUPPORT, -0.2, 10.0) and _eligible(env, e, 9.0, -10.0, 0.45)

    def pick_ok(e):
        return _eligible(env, e, T_PICK, -0.25, 0.35)

    if kind == "stack":
        ys = [e for e in belt if support_ok(e)]
        if not ys:
            return None
        x, y = table[rng.integers(len(table))], ys[rng.integers(len(ys))]
        order = [(x, y)]
    elif kind == "belt_to_table":
        xs = [e for e in belt if pick_ok(e)]
        if not xs:
            return None
        x, y = xs[rng.integers(len(xs))], table[rng.integers(len(table))]
        order = [(x, y)]
    elif kind == "belt_to_belt":
        xs = [e for e in belt if pick_ok(e)]
        if not xs:
            return None
        x = xs[rng.integers(len(xs))]
        ys = [e for e in belt if e is not x and support_ok(e)]
        if not ys:
            return None
        y = ys[rng.integers(len(ys))]
        order = [(x, y)]
    else:  # tower
        # two sequential placements: the support must arrive in time AND stay in reach for ~17 s
        ys = [e for e in belt if _eligible(env, e, 8.0, -0.25, 10.0) and _eligible(env, e, 17.0, -10.0, 0.40)]
        if not ys or len(table) < 2:
            return None
        y = ys[rng.integers(len(ys))]
        i = rng.permutation(len(table))[:2]
        x, z = table[i[0]], table[i[1]]
        order = [(x, y), (z, x)]
    return belt + table, order


def generate(env, rng, cfg):
    """Returns (entities, task). Retries until the layout is feasible and every phrase is unambiguous."""
    kinds = list(KINDS) if cfg.task == "random" else [cfg.task]
    for attempt in range(80):
        kind = kinds[rng.integers(len(kinds))] if attempt < 60 else "stack"
        built = _build(env, rng, kind, cfg)
        if built is None:
            continue
        ents, order = built
        if kind == "tower":
            (x, y), (z, _) = order
            roles = [x, y, z]
        else:
            ((x, y),) = order
            roles = [x, y]
        pick_roles = {0, 2} if kind == "tower" else {0}
        nps = [describe(e, ents, rng, role="pick" if i in pick_roles else "place") for i, e in enumerate(roles)]
        if any(n is None for n in nps):
            continue
        instr, sub = make_instruction("tower" if kind == "tower" else "stack", nps, rng)
        steps = [Step(a.name, b.name, text) for (a, b), text in zip(order, sub, strict=True)]
        ents.sort(key=lambda e: e.idx)
        return ents, Task(kind=kind, steps=steps, instruction=instr)
    raise RuntimeError("could not generate a feasible, unambiguous task layout")
