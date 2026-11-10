"""Fast checks for the MuJoCo stack (no rendering needed). Run: pytest tests/test_mujoco_env.py"""
import numpy as np

from vla_mujoco.env import DynamicStackEnv, EnvConfig
from vla_mujoco.expert import StackExpert


def make(seed=0, **kw):
    return DynamicStackEnv(EnvConfig(seed=seed, render=False, **kw))


def run_expert(env, seed):
    obs, info = env.reset(seed=seed)
    ex = StackExpert(env)
    for _ in range(int(env.cfg.max_time_s * env.cfg.control_hz)):
        obs, _, term, trunc, info = env.step(ex.act(info))
        if term or trunc:
            break
    return info


def test_ik_reaches_pose():
    env = make()
    q, ok = env.solve_ik(np.array([0.5, 0.2, 0.2]), 0.3, env.home_q)
    assert ok
    p, R = env._fk(q)
    assert np.allclose(p, [0.5, 0.2, 0.2], atol=1e-3)
    assert np.allclose(R[:, 2], [0, 0, -1], atol=1e-2)       # tool points down


def test_belt_carries_box_at_commanded_speed():
    env = make(belt_speed_range=(0.05, 0.05))
    env.reset(seed=1)
    y0 = env.box_pose("tgt")[0][1]
    for _ in range(40):
        env.step(np.zeros(5))
    dy = env.box_pose("tgt")[0][1] - y0
    assert abs(dy - 0.05 * 40 * env.dt) < 5e-3


def test_noop_policy_does_not_succeed():
    env = make()
    env.reset(seed=2)
    for _ in range(int(env.cfg.max_time_s * env.cfg.control_hz)):
        _, _, term, trunc, info = env.step(np.zeros(5))
    assert not info["success"]


def test_suction_needs_seal_and_dwell():
    env = make()
    env.reset(seed=3)
    env.step(np.array([0, 0, 0, 0, 1.0]))                    # vacuum on in free air: nothing held
    assert env.held is None


def test_expert_stacks_across_seeds():
    wins = sum(run_expert(make(seed=s), s)["success"] for s in range(8))
    assert wins >= 7


def test_expert_handles_speed_change_and_occlusion():
    env = make(speed_change=True, occlusion_s=1.0)
    assert run_expert(env, 5)["success"]


def test_reset_is_deterministic():
    a, b = make(), make()
    ia, ib = a.reset(seed=11)[1], b.reset(seed=11)[1]
    assert np.allclose(ia["priv"]["tgt_pos"], ib["priv"]["tgt_pos"])
    assert a.instruction == b.instruction
