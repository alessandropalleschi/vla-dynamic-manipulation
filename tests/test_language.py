"""Referring expressions and instruction generation (pure python, no simulator)."""

import numpy as np

from vla_mujoco.lang import describe, make_instruction
from vla_mujoco.sim.entities import sample_entity


def scene(seed=0):
    rng = np.random.default_rng(seed)
    spec = [("cube", "red", "belt"), ("cube", "red", "belt"), ("cube", "blue", "table"), ("cylinder", "red", "table")]
    ents = []
    for i, (c, col, loc) in enumerate(spec):
        e = sample_entity(rng, i, c, col, loc)
        e.belt_s, e.table_y = -0.4 - 0.4 * i, 0.3 - 0.4 * i
        ents.append(e)
    return ents


def resolve(phrase, ents):
    """Independent parser of the generated phrase: returns the entities it denotes (conjunctive semantics)."""
    w = phrase.replace("from the", "on the").split()
    loc = "belt" if "conveyor" in w else "table" if "table" in w else None
    cands = [e for e in ents if (loc is None or e.location == loc)]
    cats = {e.category for e in ents}
    cat = next((x for x in w if x in cats), None)
    if cat:
        cands = [e for e in cands if e.category == cat]
    col = next((x for x in w if x in {e.color for e in ents}), None)
    if col:
        cands = [e for e in cands if e.color == col]
    for word in (
        "first",
        "second",
        "third",
        "last",
        "leftmost",
        "rightmost",
        "larger",
        "smaller",
        "largest",
        "smallest",
    ):
        if word in w:
            if word in ("larger", "largest"):
                cands = [max(cands, key=lambda e: e.volume)]
            elif word in ("smaller", "smallest"):
                cands = [min(cands, key=lambda e: e.volume)]
            elif word in ("leftmost", "rightmost"):
                cands = sorted(cands, key=lambda e: -e.table_y)
                cands = [cands[0] if word == "leftmost" else cands[-1]]
            else:
                order = sorted(cands, key=lambda e: -e.belt_s)
                idx = {"first": 0, "second": 1, "third": 2, "last": -1}[word]
                cands = [order[idx]]
    return cands


def test_every_description_is_unique_and_resolves_to_its_target():
    ents = scene()
    for seed in range(60):
        rng = np.random.default_rng(seed)
        for t in ents:
            for role in ("pick", "place"):
                p = describe(t, ents, rng, role=role)
                assert p is not None
                got = resolve(p, ents)
                assert got == [t], (p, [e.name for e in got], t.name)


def test_colour_is_not_always_used():
    ents = scene()
    phrases = {describe(ents[2], ents, np.random.default_rng(s)) for s in range(40)}
    assert any("blue" in p for p in phrases) and any("blue" not in p for p in phrases)


def test_pick_and_place_use_different_prepositions():
    ents = scene()
    pick = {describe(ents[0], ents, np.random.default_rng(s), role="pick") for s in range(30)}
    place = {describe(ents[0], ents, np.random.default_rng(s), role="place") for s in range(30)}
    assert any("from the conveyor" in p for p in pick) and not any("on the conveyor" in p for p in pick)
    assert any("on the conveyor" in p for p in place) and not any("from the conveyor" in p for p in place)


def test_instruction_variety_and_subgoals():
    rng = np.random.default_rng(0)
    texts = {make_instruction("stack", ["the cube", "the cylinder"], rng)[0] for _ in range(40)}
    assert len(texts) >= 4
    instr, sub = make_instruction("tower", ["a", "b", "c"], rng)
    assert len(sub) == 2 and instr.endswith("c on top of it")
