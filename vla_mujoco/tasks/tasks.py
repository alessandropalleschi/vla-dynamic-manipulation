"""Task = ordered steps over entity ids + generated language; success is computed from predicates."""

from __future__ import annotations

from dataclasses import dataclass, field

from . import predicates as P


@dataclass
class Step:
    obj: str  # entity to pick up
    onto: str  # entity it must end up resting on
    text: str  # subgoal sentence for this step


@dataclass
class Task:
    kind: str  # "stack" | "belt_to_table" | "belt_to_belt" | "tower"
    steps: list[Step]
    instruction: str
    involved: set = field(default_factory=set)

    def __post_init__(self):
        self.involved = {n for s in self.steps for n in (s.obj, s.onto)}

    def status(self, env) -> dict:
        """Predicate truth values for the whole task (all steps must hold simultaneously at the end)."""
        ok, offs = [], []
        for s in self.steps:
            t, off = P.on(env, s.obj, s.onto)
            ok.append(t and off <= env.cfg.place_tol_m)
            offs.append(off)
        bystanders = [n for n in env.ent if n not in self.involved]
        collateral = [n for n in bystanders if P.disturbed(env, n)]
        return {
            "step_ok": ok,
            "offsets": offs,
            "all_on": all(ok),
            "collateral": collateral,
            "n_done": sum(ok),
        }

    def current_step(self, env) -> int:
        """Index of the first step whose predicate does not hold yet (len(steps) when done)."""
        for i, s in enumerate(self.steps):
            t, off = P.on(env, s.obj, s.onto)
            if not (t and off <= env.cfg.place_tol_m):
                return i
        return len(self.steps)
