"""Every generated instruction must be fully covered by the policy tokenizer (needs torch)."""

import pytest

pytest.importorskip("torch")

from vla_mujoco.sim.env import DynamicStackEnv, EnvConfig  # noqa: E402
from vla_mujoco.vla.policy import VOCAB  # noqa: E402


def test_instruction_words_are_in_vocab():
    env = DynamicStackEnv(EnvConfig(seed=0, render=False))
    nouns = set()
    for s in range(40):
        env.reset(seed=s)
        words = env.instruction.split()
        assert all(w in VOCAB for w in words), env.instruction
        nouns.update(words)
    assert {"cylinder", "cube"} <= nouns
