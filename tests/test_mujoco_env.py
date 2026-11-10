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
    off = make(seed=1, **EnvConfig.dr_flags(False))
    p0 = off.model.cam_pos[off.model.camera("front").id].copy()
    off.reset(seed=7)
    assert np.allclose(off.model.cam_pos[off.model.camera("front").id], p0)
    assert off.belt_yaw == 0.0


def test_expert_stacks_with_out_of_distribution_randomisation():
    wins = sum(run_expert(make(seed=s, dr_scale=1.6), s)["success"] for s in range(6))
    assert wins >= 5


def test_shape_change_keeps_inertia_and_bounds_consistent():
    env = make(seed=2)
    m = env.model
    seen = set()
    for seed in range(12):
        env.reset(seed=seed)
        for n in ("src", "tgt", "dist"):
            b = env.boxes[n]
            seen.add(env.kind[n])
            assert np.isclose(m.geom_size[b["geom"]][1 if env.shape[n] == "cylinder" else 2], env.half[n])
            assert np.all(m.body_inertia[b["body"]] > 0)
            assert m.geom_rbound[b["geom"]] >= env.half[n]
    assert {"cube", "cylinder"} <= seen


def test_expert_stacks_every_shape_combination():
    combos, wins = set(), 0
    for s in range(30):
        env = make(seed=s)
        info = run_expert(env, s)
        combos.add((env.kind["src"], env.kind["tgt"]))
        wins += info["success"]
    assert len(combos) >= 6  # cube/cuboid/cylinder pairs are all being exercised
    assert wins >= 29


def test_table_height_clutter_robot_start_and_sensor_randomisation():
    env = make(seed=3)
    tz, starts, clutter, params = set(), set(), set(), set()
    for s in range(8):
        env.reset(seed=s)
        tz.add(round(env.table_z, 3))
        starts.add(tuple(env.ee_cmd.round(3)))
        clutter.add(env.n_clutter)
        params.add(round(env.img_params["noise"], 4))
        assert -0.05 <= env.table_z <= 0.0
        # the block rests on the (lowered) table
        assert abs(env.box_pose("src")[0][2] - (env.table_z + env.half["src"])) < 2e-3
    assert len(tz) > 1 and len(starts) > 1 and len(clutter) > 1 and len(params) > 1


def test_all_randomisation_can_be_disabled():
    off = make(seed=1, randomize_physics=False, **EnvConfig.dr_flags(False))
    off.reset(seed=5)
    assert off.kind["src"] == "cube" and off.half["src"] == 0.05 and off.table_z == 0.0
    assert off.n_clutter == 0 and off.belt_yaw == 0.0
    assert off.img_params == {"noise": 0.0, "blur": 0.0, "gain": 1.0, "bias": 0.0}
    assert np.allclose(off.ee_cmd, off.home_tcp, atol=1e-3)


def test_sensor_model_changes_pixels_but_keeps_dtype():
    env = DynamicStackEnv(EnvConfig(seed=0, img_size=32, cameras=("front",)))
    env.img_params = {"noise": 0.03, "blur": 0.5, "gain": 1.1, "bias": 0.02}
    raw = env.render("front")
    out = env._sensor(raw)
    assert out.dtype == np.uint8 and out.shape == raw.shape and not np.array_equal(out, raw)


def test_action_is_a_velocity_independent_of_control_rate():
    """The same normalised action moves the commanded pose the same distance per *second* at any rate."""
    dist = {}
    for hz in (10, 40):
        env = make(seed=0, control_hz=hz, **EnvConfig.dr_flags(False))
        env.reset(seed=0)
        p0 = env.ee_cmd.copy()
        n = int(hz * 0.5)
        for _ in range(n):
            env.step(np.array([1.0, 0, 0, 0, 0]))
        dist[hz] = (env.ee_cmd - p0)[0] / (
            n * env.dt
        )  # m/s (dt is rounded to a whole number of physics steps)
    assert np.isclose(dist[10], 0.3, rtol=1e-3) and np.isclose(dist[40], 0.3, rtol=1e-3)


def test_expert_succeeds_at_other_control_rates_and_with_latency():
    for hz in (10, 40):
        assert run_expert(make(seed=3, control_hz=hz), 3)["success"]
    env = make(seed=4, randomize_latency=True, dr_scale=2.0)
    assert run_expert(env, 4)["success"]


def test_dr_flags_cover_every_randomisation_switch():
    import dataclasses

    all_flags = {f.name for f in dataclasses.fields(EnvConfig) if f.name.startswith("randomize_")}
    # colour naming and physics are semantic/physical switches, deliberately outside dr_flags
    assert set(EnvConfig.dr_flags(True)) == all_flags - {"randomize_colors", "randomize_physics"}


def test_command_latency_delays_motion_and_is_predictable():
    env = make(seed=0, **EnvConfig.dr_flags(False))
    env.reset(seed=0)
    env.act_delay_steps = 3
    from collections import deque

    env._act_queue = deque([np.zeros(5)] * 3)
    p0 = env.ee_cmd.copy()
    for _ in range(3):
        env.step(np.array([1.0, 0, 0, 0, 0]))
        assert np.allclose(env.ee_cmd, p0)  # nothing applied yet
    assert env.ee_cmd_future[0] > p0[0] + 1e-4  # ...but three actions are in flight
    env.step(np.array([0.0, 0, 0, 0, 0]))
    assert env.ee_cmd[0] > p0[0]
