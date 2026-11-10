"""Chunked execution: the belief updates every step, the action head re-plans every `exec_horizon` steps."""

import numpy as np
import pytest

pytest.importorskip("torch")

from vla_mujoco.sim.env import DynamicStackEnv, EnvConfig  # noqa: E402
from vla_mujoco.vla.policy import PolicyRunner, VLAPolicy  # noqa: E402


@pytest.mark.parametrize("arch", ["gru", "stack", "mlp"])
def test_runner_replans_every_k_steps(arch):
    env = DynamicStackEnv(EnvConfig(seed=0, img_size=64))
    obs, _ = env.reset(seed=0)
    runner = PolicyRunner(VLAPolicy(arch=arch, chunk=8), exec_horizon=4)
    for _ in range(12):
        a = runner.act(obs)
        assert a.shape == (5,) and np.all(np.abs(a[:4]) <= 1) and 0 <= a[4] <= 1
        obs, *_ = env.step(a)
    assert runner.replans == 3
