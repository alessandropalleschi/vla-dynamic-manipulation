"""Fast checks for the MuJoCo stack (no rendering needed). Run: pytest tests/test_mujoco_env.py"""

import numpy as np

from vla_mujoco.expert import StackExpert
from vla_mujoco.sim.env import DynamicStackEnv, EnvConfig


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
    assert np.allclose(R[:, 2], [0, 0, -1], atol=1e-2)  # tool points down


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
    env.step(np.array([0, 0, 0, 0, 1.0]))  # vacuum on in free air: nothing held
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


def test_conveyor_randomisation_keeps_geometry_consistent():
    env = make(seed=4)
    yaws = set()
    for s in range(6):
        env.reset(seed=s)
        yaws.add(round(env.belt_yaw, 3))
        m, d = env.model, env.data
        # rails stay parallel to the belt slab (guards against stale geom_sameframe shortcuts)
        belt_y = d.geom_xmat[m.geom("belt_top").id].reshape(3, 3)[:, 1]
        for rail in ("rail_l", "rail_r", "belt_frame"):
            assert np.allclose(d.geom_xmat[m.geom(rail).id].reshape(3, 3)[:, 1], belt_y, atol=1e-6)
        assert np.allclose(env._info()["priv"]["belt_dir"], belt_y[:2], atol=1e-6)
    assert len(yaws) > 1


def test_randomisation_changes_views_and_can_be_disabled():
    a = make(seed=1)
    pos_a = [a.reset(seed=s) and a.model.cam_pos[a.model.camera("front").id].copy() for s in range(3)]
    assert not np.allclose(pos_a[0], pos_a[1])
    off = make(seed=1, randomize_conveyor=False, randomize_camera=False, randomize_appearance=False)
    p0 = off.model.cam_pos[off.model.camera("front").id].copy()
    off.reset(seed=7)
    assert np.allclose(off.model.cam_pos[off.model.camera("front").id], p0)
    assert off.belt_yaw == 0.0 and off.half["src"] == 0.05


def test_expert_stacks_with_out_of_distribution_randomisation():
    wins = sum(run_expert(make(seed=s, dr_scale=1.6), s)["success"] for s in range(6))
    assert wins >= 5


def test_box_resize_keeps_inertia_consistent():
    env = make(seed=2)
    env.reset(seed=2)
    m = env.model
    b = env.boxes["src"]
    h = env.half["src"]
    assert np.isclose(m.geom_size[b["geom"]][0], h)
    assert np.isclose(m.body_inertia[b["body"]][0], m.body_mass[b["body"]] * (2 * h) ** 2 / 6)
