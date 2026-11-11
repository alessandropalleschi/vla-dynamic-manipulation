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


def _window(env, role: str) -> tuple[float, float]:
    """Initial belt coordinates s at which an object can still fulfil ``role`` (depends on the belt speed)."""
    v = env.belt_speed
    lo, hi = {
        "support": (-0.2 - T_SUPPORT * v, 0.45 - 9.0 * v),  # arrives in time and is not gone before we get there
        "tower_support": (-0.25 - 8.0 * v, 0.40 - 17.0 * v),  # two sequential placements: stays in reach ~17 s
        "pick": (-0.25 - T_PICK * v, 0.35 - T_PICK * v),
    }[role]
    return max(lo, -1.0), min(hi, 0.15)


def _belt_layout(env, rng, ents: list[Entity], windows: dict[str, tuple[float, float]]) -> bool:
    """Random, non-overlapping belt positions. Role objects are placed first inside their feasible window and
    bystanders anywhere on either side of them, so an object's *rank* on the belt does not reveal its role
    (otherwise the instruction would be redundant and a policy could ignore the language)."""
    placed: list[Entity] = []
    order = sorted(ents, key=lambda e: e.name not in windows)  # roles first
    for e in order:
        lo, hi = windows.get(e.name, (-1.0, 0.1))
        if lo > hi:
            return False
        for _ in range(120):
            s = float(rng.uniform(lo, hi))
            if all(abs(s - o.belt_s) >= e.footprint + o.footprint + 0.10 for o in placed):
                break
        else:
            return False
        u = float(rng.uniform(-0.03, 0.03))
        e.belt_s = s
        xy = env.belt_to_world(u, s)
        e.pos = np.array([xy[0], xy[1], BELT_TOP + e.hz + 0.0005])
        e.yaw = float(env.belt_yaw + rng.uniform(-0.15, 0.15))
        placed.append(e)
    return True


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

    # roles first (uniformly at random), then positions that make them feasible
    bi = rng.permutation(len(belt))
    ti = rng.permutation(len(table)) if table else []
    if kind == "stack":
        x, y = table[ti[0]], belt[bi[0]]
        windows, order = {y.name: _window(env, "support")}, [(x, y)]
    elif kind == "belt_to_table":
        x, y = belt[bi[0]], table[ti[0]]
        windows, order = {x.name: _window(env, "pick")}, [(x, y)]
    elif kind == "belt_to_belt":
        x, y = belt[bi[0]], belt[bi[1]]
        windows, order = {x.name: _window(env, "pick"), y.name: _window(env, "support")}, [(x, y)]
    else:  # tower: x onto belt object y, then z onto x
        y = belt[bi[0]]
        x, z = table[ti[0]], table[ti[1]]
        windows, order = {y.name: _window(env, "tower_support")}, [(x, y), (z, x)]
    if not _belt_layout(env, rng, belt, windows) or not _table_layout(env, rng, table):
        return None
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
