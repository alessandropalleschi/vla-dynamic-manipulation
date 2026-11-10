"""Referring-expression generation: minimal *unique* descriptions of entities in the current scene.

Attributes (any subset may be used, in a per-call random order): category noun, colour, relative size,
location (conveyor / table) and ordering (arrival order along the belt, left/right on the table).
Colour is deliberately optional, so a policy cannot rely on it: "the second cube on the conveyor",
"the larger cylinder", "the red one on the table" are all produced depending on what disambiguates.
"""

from __future__ import annotations

import numpy as np

from ..sim.entities import Entity

ORD = ["first", "second", "third", "fourth", "fifth"]


def _ordinal(target: Entity, cands: list[Entity]):
    """Ordering word for ``target`` among ``cands`` that share its location, or None if not expressible."""
    same = [e for e in cands if e.location == target.location]
    if len(same) < 2:
        return None
    if target.location == "belt":  # arrival order: most downstream (largest s) first
        order = sorted(same, key=lambda e: -e.belt_s)
        i = order.index(target)
        if i == len(order) - 1 and len(order) > 2:
            return "last"
        return ORD[i] if i < len(ORD) else None
    order = sorted(same, key=lambda e: -e.table_y)  # +y is the robot's left
    if order[0] is target:
        return "leftmost"
    if order[-1] is target:
        return "rightmost"
    return None


def _size_word(target: Entity, cands: list[Entity]):
    same = [e for e in cands if e.category == target.category]
    if len(same) < 2:
        return None
    vols = sorted(e.volume for e in same)
    v = target.volume
    if v == vols[-1] and v > 1.2 * vols[-2]:
        return "larger" if len(same) == 2 else "largest"
    if v == vols[0] and vols[1] > 1.2 * v:
        return "smaller" if len(same) == 2 else "smallest"
    return None


def describe(
    target: Entity,
    ents: list[Entity],
    rng: np.random.Generator,
    over_specify: float = 0.25,
    role: str = "place",
):
    """Noun phrase that uniquely identifies ``target`` among ``ents`` (None if impossible).

    Semantics are conjunctive and unambiguous: category / colour / location filter the scene first; a size word
    ("larger") or an ordering word ("second", "leftmost") is then evaluated *among the objects those filters
    match* (an ordering word implies the location filter).
    """
    cands = list(ents)
    used: dict = {}
    # phase 1: filters, in random order (colour is often skipped unless needed; category is the usual head noun)
    filters = ["color", "location"]
    rng.shuffle(filters)
    if rng.random() < 0.85:
        filters.insert(0, "category")
    else:
        filters.append("category")
    for a in filters:
        if len(cands) == 1:
            break
        new = [e for e in cands if getattr(e, "category" if a == "category" else a) == getattr(target, a)]
        if 0 < len(new) < len(cands):
            used[a] = getattr(target, a)
            cands = new
    # phase 2: relative attributes among the remaining candidates (need the noun to read naturally)
    if len(cands) > 1:
        rel = ["size", "ordinal"]
        rng.shuffle(rel)
        for a in rel:
            if len(cands) == 1:
                break
            if a == "size":
                val = _size_word(target, cands)
                if val and len({e.category for e in cands}) == 1:
                    used["size"] = val
                    used.setdefault("category", target.category)
                    cands = [target]
            else:
                val = _ordinal(target, cands)
                if val:
                    used["ordinal"] = val
                    used["location"] = target.location
                    cands = [target]
    if len(cands) != 1:
        return None
    if "location" not in used and rng.random() < over_specify:
        used["location"] = target.location  # harmless over-specification
    words = [w for w in (used.get("ordinal"), used.get("size"), used.get("color")) if w]
    phrase = "the " + " ".join(words + [used.get("category", "object")])
    if "location" in used:
        # the object to pick up is "from" a place, the support is "on" a place (keeps "on"/"onto" unambiguous)
        prep = "from" if role == "pick" else "on"
        phrase += f" {prep} the " + {"belt": "conveyor", "table": "table"}[used["location"]]
    return phrase


STACK_TEMPLATES = [
    "put {a} onto {b}",
    "stack {a} on top of {b}",
    "place {a} onto {b}",
    "pick up {a} and put it onto {b}",
    "move {a} on top of {b}",
    "take {a} and stack it onto {b}",
]


def make_instruction(kind: str, nps: list[str], rng: np.random.Generator) -> tuple[str, list[str]]:
    """Full-task instruction and one subgoal sentence per step. ``nps`` are noun phrases in step order.

    ``stack``: nps = [a (pick), b (support)]            ->  a onto b
    ``tower``: nps = [a (pick), b (support), c (pick)]  ->  a onto b, then c on top of it
    """
    t = STACK_TEMPLATES[rng.integers(len(STACK_TEMPLATES))]
    if kind == "tower":
        a, b, c = nps
        subgoals = [f"put {a} onto {b}", f"put {c} on top of the object you just placed"]
        joiner = rng.choice(
            [
                ", then put {c} on top of it",
                " and then put {c} on top of it",
                ". After that, put {c} on top of it",
            ]
        )
        return t.format(a=a, b=b) + joiner.format(c=c), subgoals
    a, b = nps
    return t.format(a=a, b=b), [f"put {a} onto {b}"]
