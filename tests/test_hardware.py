"""Flange adapter plate, OAK-D Pro wrist camera and the beyond-the-belt top camera (geometry, clearance, plumbing)."""

import sys

import mujoco
import numpy as np
import pytest

from vla_mujoco.expert import StackExpert
from vla_mujoco.sim import scene
from vla_mujoco.sim.env import DynamicStackEnv, EnvConfig


def stl_triangles(path):
    """Binary STL -> (n, 3, 3) vertices and (n, 3) normals (no extra dependencies)."""
    raw = np.fromfile(path, dtype=np.uint8)
    n = int(np.frombuffer(raw[80:84], "<u4")[0])
    rec = np.frombuffer(raw[84 : 84 + 50 * n], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return rec["v"].reshape(-1, 3, 3).astype(float), rec["n"].astype(float)


def model(wrist="oak", overhead="front", **kw):
    """Nominal (not randomised) hardware, forwarded once."""
    env = DynamicStackEnv(
        EnvConfig(seed=0, render=False, wrist_mount=wrist, overhead_view=overhead, **EnvConfig.dr_flags(False), **kw)
    )
    mujoco.mj_forward(env.model, env.data)
    return env


def geom_distance(env, a, b, dmax=0.3):
    m = env.model
    return mujoco.mj_geomDistance(m, env.data, m.geom(a).id, m.geom(b).id, dmax, None)


def test_options_build_validate_and_default_to_the_old_hardware():
    for wm in scene.WRIST_MOUNTS:
        for ov in scene.OVERHEAD_VIEWS:
            m = mujoco.MjModel.from_xml_string(scene.build_mjcf(32, 32, wm, ov))
            assert {"front", "overhead", "wrist"} <= {m.camera(i).name for i in range(m.ncam)}
            assert ("oak_wrist" in [m.body(i).name for i in range(m.nbody)]) == (wm == "oak")
    with pytest.raises(ValueError):
        scene.build_mjcf(32, 32, "bogus", "top")
    m = mujoco.MjModel.from_xml_string(
        scene.build_mjcf(32, 32)
    )  # defaults reproduce the previous nominal poses exactly
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert np.allclose(m.cam_pos[m.camera("wrist").id], [0.0, 0.075, 0.12]) and m.cam_fovy[m.camera("wrist").id] == 85
    ov = m.camera("overhead").id
    assert np.allclose(d.cam_xpos[ov], [scene.BELT_X - 0.1, 0.0, 1.25])
    assert np.allclose(-d.cam_xmat[ov].reshape(3, 3)[:, 2], [0, 0, -1])
    assert EnvConfig().wrist_mount == "legacy" and EnvConfig().overhead_view == "top"


def test_adapter_fills_the_flange_gap_exactly():
    tris, _ = stl_triangles(scene.ASSET_DIR / "link6.stl")
    assert abs(tris[..., 2].max()) < 1e-4  # link6's flange face is the plane z = 0 ...
    assert scene.ADAPTER_T == scene.TOOL_MOUNT_Z == 0.015  # ... and the VGC10 body starts 15 mm beyond it
    for name, wing in (("adapter_disc.stl", False), ("adapter_plate.stl", True)):
        v, _ = stl_triangles(scene.ADAPTER_DIR / name)
        flat = v.reshape(-1, 3)
        assert abs(flat[:, 2].min()) < 1e-6  # flush with the flange
        disc = flat[np.hypot(flat[:, 0], flat[:, 1]) < scene.ADAPTER_R - 1e-4]
        assert abs(disc[:, 2].max() - scene.ADAPTER_T) < 1e-6  # flush with the VGC10
        assert (flat[:, 1].max() > scene.WING_TIP_Y - 0.01) == wing  # only the camera plate has a wing
        if wing:
            assert (
                abs(flat[:, 1].max() - scene.WING_TIP_Y) < 1e-3 and np.abs(flat[:, 0]).max() <= scene.WING_HALF_W + 1e-6
            )
        else:
            assert np.hypot(flat[:, 0], flat[:, 1]).max() < scene.ADAPTER_R + 1e-6


def test_wing_face_is_the_camera_tilt_plane():
    pose = scene.oak_pose()
    d, q = pose["d"], pose["plane_point"]
    v, n = stl_triangles(scene.ADAPTER_DIR / "adapter_plate.stl")
    on_face = (n @ d) > 0.999  # triangles whose normal is the viewing direction
    assert on_face.sum() >= 2
    verts = v[on_face].reshape(-1, 3)
    assert np.abs((verts - q) @ d).max() < 2e-5  # every vertex lies on d.(X - Q) = 0
    area = (
        0.5
        * np.linalg.norm(
            np.cross(v[on_face][:, 1] - v[on_face][:, 0], v[on_face][:, 2] - v[on_face][:, 0]), axis=1
        ).sum()
    )
    assert area > 3e-3  # a real mounting face (camera back is 97 x 29.5 mm = 2.9e-3 m^2, with margin)
    tilt = np.degrees(np.arccos(d[2]))
    assert 30 < tilt < 60  # the plate's face angle is the camera tilt from the tool axis


def test_wing_collision_shape_is_the_convex_wedge_under_the_camera_face():
    env = model()
    m = env.model
    wing = m.geom("tool_adapter_wing_col")
    assert m.geom_contype[wing.id] == 1 and m.geom_conaffinity[wing.id] == 2  # collides with objects like the tool body
    v, _ = stl_triangles(scene.ADAPTER_DIR / "adapter_wing.stl")
    flat = v.reshape(-1, 3)
    assert (
        abs(flat[:, 2].min()) < 1e-6 and flat[:, 1].min() > scene.ADAPTER_R
    )  # starts beyond the disc, flush with the flange
    pose = scene.oak_pose()
    below = (flat - pose["plane_point"]) @ pose["d"]
    assert (
        below.max() < 2e-5
    )  # nothing of the wedge rises above the camera face plane (so the camera never intersects it)


def test_camera_looks_at_the_cup_tips_and_sits_on_the_wing():
    env = model()
    m, d = env.model, env.data
    cam = m.camera("wrist").id
    axis = -d.cam_xmat[cam].reshape(3, 3)[:, 2]
    tcp = d.site_xpos[env.site]
    to_tcp = (tcp - d.cam_xpos[cam]) / np.linalg.norm(tcp - d.cam_xpos[cam])
    assert np.degrees(np.arccos(np.clip(axis @ to_tcp, -1, 1))) < 4.0  # optical axis -> cup tips
    uv, visible = env.project(tcp, "wrist")
    assert visible and np.all(np.abs(uv - 0.5) < 0.15)
    gap = geom_distance(env, "oak_wrist_col", "tool_adapter_wing_col")
    assert 0.0005 < gap < 0.0025  # resting on the wing face with a ~1 mm clearance
    boxes = [
        i
        for i in range(m.ngeom)
        if m.geom_bodyid[i] == m.body("tool").id and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
    ]
    body = next(i for i in boxes if np.allclose(m.geom_size[i], [0.0505, 0.05, 0.05]))  # the VGC10 body proxy
    assert mujoco.mj_geomDistance(m, d, m.geom("oak_wrist_col").id, body, 0.3, None) > 0.02  # clear of the VGC10 body
    assert d.geom_xpos[m.geom("oak_wrist_col").id][2] != 0  # sanity: the camera box exists in the world


def test_overhead_front_view_looks_back_at_the_robot_from_beyond_the_belt():
    env = model()
    m, d = env.model, env.data
    c = m.camera("overhead").id
    axis = -d.cam_xmat[c].reshape(3, 3)[:, 2]
    assert d.cam_xpos[c][0] > scene.BELT_X + scene.BELT_HALF_W + 0.012 + 0.1  # beyond the belt and its rail
    assert axis[0] < -0.3 and axis[2] < -0.8 and abs(axis[1]) < 1e-6  # toward the robot (-x), mostly downward
    assert 20 < np.degrees(np.arccos(-axis[2])) < 30  # ~25 deg from vertical
    for p in ([0.0, 0.0, 0.25], [scene.BELT_X, 0.0, scene.BELT_TOP], [0.3, 0.2, 0.05]):  # robot, belt, pick area
        _, visible = env.project(p, "overhead")
        assert visible, p


def test_lfs_stub_for_the_new_meshes_gives_the_actionable_error(tmp_path, monkeypatch):
    stub = tmp_path / "adapter_plate.stl"
    stub.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 1\n")
    (tmp_path / "adapter_disc.stl").write_bytes(b"x" * 100)
    monkeypatch.setattr(scene, "ADAPTER_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="git lfs pull"):
        scene.build_mjcf(32, 32, "oak", "top")


def test_expert_succeeds_with_the_new_hardware_and_it_never_touches_anything():
    wins, touched = 0, {}
    for seed in range(6):
        env = DynamicStackEnv(EnvConfig(seed=seed, render=False, task="random", n_distractors=2, n_table_distractors=1,
                                        wrist_mount="oak", overhead_view="front"))  # fmt: skip
        obs, info = env.reset(seed=seed)
        ex = StackExpert(env)
        m, d = env.model, env.data
        hw = {m.geom(n).id for n in ("oak_wrist_col", "tool_adapter_col", "tool_adapter_wing_col")}
        for _ in range(int(env.cfg.max_time_s * env.cfg.control_hz)):
            obs, _, te, tr, info = env.step(ex.act(info))
            for i in range(d.ncon):
                c = d.contact[i]
                if c.geom1 in hw or c.geom2 in hw:
                    touched[m.geom(c.geom2 if c.geom1 in hw else c.geom1).name] = True
            if te or tr:
                break
        wins += int(info["success"])
    assert wins >= 5 and not touched, (wins, touched)


def tiny_checkpoint(path, **env_cfg):
    import torch

    from vla_mujoco.vla.policy import VLAPolicy
    from vla_mujoco.vla.text import TextBackend, TextConfig

    be = TextBackend(TextConfig(kind="bow"))
    m = VLAPolicy(arch="mlp", text=be.cfg, patch=8, chunk=4)
    torch.save({"model": m.state_dict(), "cfg": m.cfg, "args": {}, "env_cfg": {"img_size": 32, **env_cfg}}, path)


def spy_on(module, monkeypatch):
    seen = []
    real = module.DynamicStackEnv

    class Spy(real):
        def __init__(self, cfg):
            seen.append(cfg)
            super().__init__(cfg)

    monkeypatch.setattr(module, "DynamicStackEnv", Spy)
    return seen


def test_evaluate_and_dagger_use_the_hardware_the_policy_was_trained_on(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from vla_mujoco.vla import dagger, evaluate

    ck = tmp_path / "p.pt"
    tiny_checkpoint(ck, wrist_mount="oak", overhead_view="front")
    seen = spy_on(evaluate, monkeypatch)
    monkeypatch.setattr(
        sys, "argv", ["e", "--policy", str(ck), "--episodes", "1", "--task", "stack", "--device", "cpu"]
    )
    evaluate.main()
    assert (seen[0].wrist_mount, seen[0].overhead_view) == ("oak", "front")  # not silently the legacy default
    seen.clear()
    monkeypatch.setattr(sys, "argv", ["e", "--policy", str(ck), "--episodes", "1", "--task", "stack", "--device", "cpu",
                                      "--wrist-mount", "legacy"])  # fmt: skip
    evaluate.main()
    assert (seen[0].wrist_mount, seen[0].overhead_view) == ("legacy", "front")  # an explicit flag still wins
    seen = spy_on(dagger, monkeypatch)
    stage = dict(task="stack", n_distractors=0, n_table_distractors=0, belt_speed_range=(0.02, 0.03), dr_scale=0.5)
    for beta in (0.0, 1.0):  # learner-driven and expert-only rollouts both rebuild the policy's hardware
        dagger.rollout((1, stage, str(ck), beta, 1, False))
        assert (seen[-1].wrist_mount, seen[-1].overhead_view) == ("oak", "front"), beta
