"""Hashed tokenizer: deterministic, no out-of-vocabulary words, order-sensitive (needs torch)."""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from vla_mujoco.sim.env import DynamicStackEnv, EnvConfig  # noqa: E402
from vla_mujoco.vla.text import MAX_TOKENS, VOCAB_SIZE, hash_tokens  # noqa: E402


def test_tokenizer_handles_any_generated_instruction():
    env = DynamicStackEnv(EnvConfig(seed=0, render=False, task="random"))
    for s in range(30):
        env.reset(seed=s)
        tok = hash_tokens(env.instruction)
        n = len(env.instruction.replace(".", "").replace(",", "").split())
        assert tok.shape == (MAX_TOKENS,)
        assert int((tok > 0).sum()) == min(n, MAX_TOKENS), env.instruction  # nothing dropped / unknown
        assert int(tok.max()) < VOCAB_SIZE


def test_tokenizer_is_deterministic_and_order_sensitive():
    a, b = "put the red cube onto the blue block", "put the blue block onto the red cube"
    assert torch.equal(hash_tokens(a), hash_tokens(a))
    assert not torch.equal(hash_tokens(a), hash_tokens(b))
