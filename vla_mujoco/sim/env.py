"""MuJoCo environment: UF850 + vacuum tool stacking a block onto a moving conveyor box.

Gymnasium-style API (no hard gymnasium dependency)::

    env = DynamicStackEnv(EnvConfig(seed=0))
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)

Observation (what a deployed VLA may use; no privileged state)::

    obs["images"]["front" | "wrist" | "overhead"]  uint8 (H, W, 3)
    obs["proprio"]      float32 (PROPRIO_DIM,)
    obs["instruction"]  str

Action (float32, 5-D, normalised to [-1, 1] except the last)::

    [vx, vy, vz, wz, vacuum]   normalised Cartesian end-effector velocities (base frame,
    +-1 = MAX_SPEED_M_S / MAX_YAW_RATE) + vacuum command in [0, 1] (> 0.5 = on).
    The commanded pose advances by ``a * max_speed * dt``, so a policy does not depend on ``control_hz``.

``info["priv"]`` carries privileged ground truth (box poses, belt velocity,
contact flags) for the expert, auxiliary losses and evaluation only.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..tasks.generator import generate
from .entities import PALETTE  # noqa: F401  (re-exported)
from .randomize import DomainRandomizer
from .scene import BELT_TOP, CUP_R, CUP_XY, N_OBJ, build_mjcf

PROPRIO_DIM = 22
# Actions are commanded *velocities* (normalised to +-1), so a policy is independent of the control rate:
# the commanded pose advances by  a * MAX_SPEED * dt  each control step.
MAX_SPEED_M_S = 0.30  # translation speed at |action| = 1
MAX_YAW_RATE = 2.0  # rad/s at |action| = 1
WORKSPACE_LO = np.array([0.15, -0.60, -0.02])  # TCP bounds (m)
WORKSPACE_HI = np.array([0.80, 0.60, 0.60])
# VGC10 timing (config/vgc10.yaml): vacuum build-up / release delays
GRIP_TIME_S = 0.35
RELEASE_TIME_S = 0.20

HOME_Q = np.array([0.0, 0.0, -1.2, 0.0, 1.2, 0.0])  # refined at construction by IK


@dataclass
class EnvConfig:
    seed: int = 0
    img_size: int = 128
    cameras: tuple = ("front", "wrist")
    control_hz: float = 20.0
    max_time_s: float = 22.0
    belt_speed_range: tuple = (0.02, 0.07)  # m/s
    speed_change: bool = False  # surprise belt-speed change mid-episode
    task: str = "stack"  # stack | belt_to_table | belt_to_belt | tower | random
    n_distractors: int = 1  # extra bystander objects on the belt (language must disambiguate)
    n_table_distractors: int = 0  # extra bystander objects on the table
    require_unmoved: bool = True  # knocking over / displacing a bystander makes the episode fail
    occlusion_s: float = 0.0  # >0: a panel hides the target from `front` for this long
    randomize_colors: bool = True
    randomize_physics: bool = True
    randomize_conveyor: bool = True  # belt x-offset + yaw
    randomize_camera: bool = True  # camera pose / fov (front, overhead, wrist)
    randomize_appearance: bool = True  # object/scene colours, size jitter, tints, lighting
    randomize_shape: bool = True  # stacked/target/distractor shape: cube | cuboid | cylinder
    randomize_texture: bool = True  # random textures on objects, table, floor; belt stripe density
    randomize_clutter: bool = True  # 0-6 visual-only distractor objects on the table
    randomize_table: bool = True  # pick-up surface height
    randomize_robot_start: bool = True  # initial TCP pose around home
    randomize_image: bool = True  # sensor noise / blur / exposure on camera images
    randomize_latency: bool = True  # command latency (0-80 ms) and camera latency (0-100 ms)
    dr_scale: float = 1.0  # multiplies all randomisation ranges (>1 for out-of-distribution tests)
    render: bool = True
    success_hold_s: float = 0.5
    place_tol_m: float = 0.012
    extra: dict = field(default_factory=dict)

    @staticmethod
    def dr_flags(on: bool) -> dict:
        """All visual/geometric domain-randomisation switches at once (physics/colour-naming stay as set)."""
        keys = (
            "conveyor",
            "camera",
            "appearance",
            "shape",
            "texture",
            "clutter",
            "table",
            "robot_start",
            "image",
            "latency",
        )
        return {f"randomize_{k}": on for k in keys}


def _yaw_rot(yaw: float) -> np.ndarray:
    """Tool-down orientation (tool z = world -z) with x-axis rotated by ``yaw``."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


class DynamicStackEnv:
    def __init__(self, cfg: EnvConfig | None = None):
        self.cfg = cfg or EnvConfig()
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(self.cfg.img_size))
        self.data = mujoco.MjData(self.model)
        self._ik = mujoco.MjData(self.model)
        m = self.model
        self.n_sub = int(round(1.0 / (self.cfg.control_hz * m.opt.timestep)))
        self.dt = self.n_sub * m.opt.timestep
        self.rng = np.random.default_rng(self.cfg.seed)

        self.arm_qadr = np.array([m.joint(f"joint{i}").qposadr[0] for i in range(1, 7)])
        self.arm_dadr = np.array([m.joint(f"joint{i}").dofadr[0] for i in range(1, 7)])
        self.q_lo = m.jnt_range[[m.joint(f"joint{i}").id for i in range(1, 7)], 0]
        self.q_hi = m.jnt_range[[m.joint(f"joint{i}").id for i in range(1, 7)], 1]
        self.site = m.site("tcp").id
        self.belt_adr = m.joint("belt_slide").qposadr[0]
        self.belt_act = m.actuator("belt_drive").id
        self.boxes = {f"obj{i}": self._box_info(f"obj{i}") for i in range(N_OBJ)}
        self.ent, self.task, self.init_pose = {}, None, {}
        self.belt_top = BELT_TOP
        self.belt_x, self.belt_yaw = 0.62, 0.0
        self.rand = DomainRandomizer(m)
        self.act_delay_steps = 0
        self.obs_delay_steps = 0
        self.belt_dir = np.array([0.0, 1.0])
        self.renderer = mujoco.Renderer(m, self.cfg.img_size, self.cfg.img_size) if self.cfg.render else None
        self.home_q = self._solve_home()
        self.home_tcp = self._fk(self.home_q)[0]
        self.reset(seed=self.cfg.seed)

    # ------------------------------------------------------------------ setup
    def _box_info(self, name):
        m = self.model
        b = m.body(name)
        return dict(
            body=b.id,
            qadr=m.joint(f"{name}_free").qposadr[0],
            dadr=m.joint(f"{name}_free").dofadr[0],
            geom=int(m.body_geomadr[b.id]),
            eq=m.equality(f"grasp_{name}").id,
        )

    def _solve_home(self):
        q0 = np.array([0.0, 0.2, -1.6, 0.0, 1.4, 0.0])
        q, ok = self.solve_ik(np.array([0.45, 0.0, 0.35]), 0.0, q0)
        assert ok, "home IK failed"
        return q

    # --------------------------------------------------------------------- IK
    def _fk(self, q):
        d = self._ik
        d.qpos[self.arm_qadr] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        return d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()

    def solve_ik(self, pos, yaw, q_init, iters=40, tol=2e-4, damping=2e-2, max_dq=0.35):
        """Damped-least-squares IK for the tool-down pose at ``pos`` with heading ``yaw``."""
        Rd = _yaw_rot(yaw)
        q = np.clip(np.asarray(q_init, float).copy(), self.q_lo, self.q_hi)
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(iters):
            p, R = self._fk(q)
            ep = pos - p
            er = 0.5 * sum(np.cross(R[:, i], Rd[:, i]) for i in range(3))
            e = np.concatenate([ep, er])
            if np.linalg.norm(ep) < tol and np.linalg.norm(er) < 10 * tol:
                return q, True
            mujoco.mj_jacSite(self.model, self._ik, jacp, jacr, self.site)
            J = np.vstack([jacp[:, self.arm_dadr], jacr[:, self.arm_dadr]])
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), e)
            n = np.abs(dq).max()
            if n > max_dq:
                dq *= max_dq / n
            q = np.clip(q + dq, self.q_lo, self.q_hi)
        p, R = self._fk(q)
        err = np.linalg.norm(pos - p)
        return q, err < 2e-3

    # ------------------------------------------------------------------ state
    @property
    def ee_cmd_future(self):
        """Commanded TCP position once all in-flight (delayed) actions have been applied."""
        pending = [np.clip(q[:3], -1, 1) for q in self._act_queue] if self.act_delay_steps else []
        return self.ee_cmd + (np.sum(pending, axis=0) if pending else 0.0) * MAX_SPEED_M_S * self.dt

    @property
    def yaw_cmd_future(self):
        pending = [np.clip(q[3], -1, 1) for q in self._act_queue] if self.act_delay_steps else []
        return self.yaw_cmd + float(np.sum(pending)) * MAX_YAW_RATE * self.dt

    def tcp_pose(self):
        d = self.data
        return d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()

    def tcp_yaw(self):
        _, R = self.tcp_pose()
        return math.atan2(R[1, 0], R[0, 0])

    def belt_coords(self, xy):
        """World xy -> belt frame (u across the belt, s along it)."""
        c, sn = math.cos(self.belt_yaw), math.sin(self.belt_yaw)
        r = np.asarray(xy, float) - np.array([self.belt_x, 0.0])
        return float(r[0] * c + r[1] * sn), float(-r[0] * sn + r[1] * c)

    def entity_pose(self, name):
        return self.box_pose(name)

    def entity_vel(self, name):
        dadr = self.boxes[name]["dadr"]
        return self.data.qvel[dadr : dadr + 3].copy()

    def box_pose(self, name):
        b = self.boxes[name]
        return self.data.xpos[b["body"]].copy(), self.data.xmat[b["body"]].reshape(3, 3).copy()

    def belt_velocity(self):
        """Signed belt speed along the belt direction (m/s)."""
        return float(self.data.qvel[self.model.joint("belt_slide").dofadr[0]])

    def belt_to_world(self, u, s):
        """Belt-frame coords (u across, s along the belt) -> world xy."""
        c, sn = math.cos(self.belt_yaw), math.sin(self.belt_yaw)
        return np.array([self.belt_x + u * c - s * sn, u * sn + s * c])

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r, m, d, cfg = self.rng, self.model, self.data, self.cfg
        mujoco.mj_resetData(m, d)
        opts = options or {}

        # --- scene randomisation
        self.belt_speed = float(opts.get("belt_speed", r.uniform(*cfg.belt_speed_range)))
        self.speed_after = self.belt_speed
        self.speed_change_t = np.inf
        if cfg.speed_change:
            self.speed_change_t = float(r.uniform(4.0, 8.0))
            self.speed_after = float(
                np.clip(
                    self.belt_speed * r.uniform(0.5, 1.6),
                    cfg.belt_speed_range[0],
                    cfg.belt_speed_range[1] * 1.3,
                )
            )
        # --- domain randomisation (conveyor pose, cameras, appearance, object sizes)
        dr, k = self.rand, cfg.dr_scale
        self.belt_x, self.belt_yaw = dr.conveyor(r, k, cfg.randomize_conveyor)
        self.belt_dir = np.array([-math.sin(self.belt_yaw), math.cos(self.belt_yaw)])
        dr.cameras(r, k, cfg.randomize_camera)
        dr.appearance(r, k, cfg.randomize_appearance)
        self.table_z = dr.table_height(r, k, cfg.randomize_table)
        ents, self.task = generate(self, r, cfg)
        self.ent = {e.name: e for e in ents}
        self.instruction = self.task.instruction
        self.init_pose = {}
        m_obj_geoms = []
        for i in range(N_OBJ):
            name = f"obj{i}"
            b = self.boxes[name]
            if name in self.ent:
                e = self.ent[name]
                rgb = dr.jitter_color(r, PALETTE[e.color], k) if cfg.randomize_appearance else PALETTE[e.color]
                m.geom_rgba[b["geom"]] = (*rgb, 1.0)
                dr.set_shape(b["geom"], b["body"], e.shape, e.dims, e.mass)
                # MuJoCo also caches contact filters per body (body_contype/conaffinity): set both levels
                m.geom_contype[b["geom"]], m.geom_conaffinity[b["geom"]] = 2, 3
                m.body_contype[b["body"]], m.body_conaffinity[b["body"]] = 2, 3
                m.body_gravcomp[b["body"]] = 0.0
                if cfg.randomize_physics:
                    m.geom_friction[b["geom"], 0] = r.uniform(0.7, 1.2)
                self._set_box(name, e.pos, e.yaw)
                self.init_pose[name] = e.pos.copy()
                m_obj_geoms.append(b["geom"])
            else:  # parked: no collisions, no gravity, far away
                m.geom_contype[b["geom"]] = m.geom_conaffinity[b["geom"]] = 0
                m.body_contype[b["body"]] = m.body_conaffinity[b["body"]] = 0
                m.body_gravcomp[b["body"]] = 1.0
                self._set_box(name, (4 + 0.4 * i, 4, -4), 0.0)
        dr.textures(r, m_obj_geoms, k, cfg.randomize_texture)
        self.n_clutter = dr.clutter(r, d, k, cfg.randomize_clutter)
        self.img_params = dr.image_params(r, k, cfg.randomize_image)
        self._img_rng = np.random.default_rng(int(r.integers(2**32)))
        self._released = {n: False for n in self.ent}

        # latency: commands take effect `act_delay_steps` later; images are `obs_delay_steps` old
        if cfg.randomize_latency:
            self.act_delay_s = float(r.uniform(0.0, 0.08) * k)
            self.obs_delay_s = float(r.uniform(0.0, 0.10) * k)
        else:
            self.act_delay_s = self.obs_delay_s = 0.0
        self.act_delay_steps = int(round(self.act_delay_s / self.dt))
        self.obs_delay_steps = int(round(self.obs_delay_s / self.dt))
        self._act_queue = deque([np.zeros(5)] * self.act_delay_steps)
        self._frames = {c: deque(maxlen=self.obs_delay_steps + 1) for c in cfg.cameras}

        # robot start pose: home, or a random TCP pose around it
        q0, yaw0 = self.home_q, 0.0
        if cfg.randomize_robot_start:
            pos = self.home_tcp + r.uniform(-0.07, 0.07, 3) * k
            pos[2] = np.clip(pos[2], 0.25, 0.45)
            yaw_try = float(r.uniform(-0.4, 0.4) * k)
            q_try, ok = self.solve_ik(pos, yaw_try, self.home_q)
            if ok:
                q0, yaw0 = q_try, yaw_try
        d.qpos[self.arm_qadr] = q0
        d.ctrl[:6] = q0
        mujoco.mj_forward(m, d)
        p, _ = self.tcp_pose()
        self.ee_cmd = p.copy()
        self.yaw_cmd = yaw0
        self.vac_cmd = 0.0
        self.held = None
        self.seal_t = 0.0
        self.release_t = None
        self.t = 0.0
        self.steps = 0
        self._stable_t = 0.0
        self._occ_active = False
        if cfg.occlusion_s > 0:
            self.occ_start = float(r.uniform(6.0, 9.0))
        else:
            self.occ_start = np.inf
        m.eq_active0[:] = 0
        d.eq_active[:] = 0
        d.ctrl[self.belt_act] = self.belt_speed
        # let the belt/boxes reach steady state before the episode starts
        for _ in range(int(0.5 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        self.t = 0.0
        return self._obs(), self._info()

    def _set_box(self, name, pos, yaw):
        b = self.boxes[name]
        d = self.data
        d.qpos[b["qadr"] : b["qadr"] + 3] = pos
        d.qpos[b["qadr"] + 3 : b["qadr"] + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        d.qvel[b["dadr"] : b["dadr"] + 6] = 0

    # ------------------------------------------------------------------- step
    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        assert a.shape[0] == 5, "action must be [vx, vy, vz, wz, vacuum] (normalised velocities)"
        m, d = self.model, self.data
        if self.act_delay_steps > 0:  # command latency: apply the action issued `act_delay_steps` ago
            self._act_queue.append(a)
            a = self._act_queue.popleft()
        self.ee_cmd = np.clip(self.ee_cmd + np.clip(a[:3], -1, 1) * MAX_SPEED_M_S * self.dt, WORKSPACE_LO, WORKSPACE_HI)
        self.yaw_cmd = float(np.clip(self.yaw_cmd + np.clip(a[3], -1, 1) * MAX_YAW_RATE * self.dt, -2.0, 2.0))
        self.vac_cmd = float(a[4] > 0.5)

        q_cur = d.qpos[self.arm_qadr].copy()
        q_des, _ = self.solve_ik(self.ee_cmd, self.yaw_cmd, q_cur, iters=12)
        d.ctrl[:6] = q_des

        # belt speed schedule
        v = self.belt_speed if self.t < self.speed_change_t else self.speed_after
        d.ctrl[self.belt_act] = v
        self._update_occluder()

        for _ in range(self.n_sub):
            mujoco.mj_step(m, d)
        self.t += self.dt
        self.steps += 1
        self._update_suction()

        terminated, truncated, success = self._check_done()
        info = self._info()
        info["success"] = success
        reward = 1.0 if success else 0.0
        return self._obs(), reward, terminated, truncated, info

    # ---------------------------------------------------------------- suction
    def _seal_candidate(self):
        """Return the box whose top face is sealed by the cups right now (else None)."""
        tcp, R = self.tcp_pose()
        tool_z = R[:, 2]
        for name, e in self.ent.items():
            pos, Rb = self.box_pose(name)
            n = Rb[:, 2]
            hz = e.hz
            hx, hy = e.dims[0], e.dims[1]
            top = pos + n * hz
            gap = float((tcp - top) @ n)
            if not (-0.004 <= gap <= 0.008 and float(tool_z @ n) < -0.97):
                continue
            # all four cup centres (+-CUP_XY) must be on the top face, >= half a cup radius from the edge
            ok = True
            for sx in (-1, 1):
                for sy in (-1, 1):
                    c = tcp + R @ np.array([sx * CUP_XY, sy * CUP_XY, 0.0]) - top
                    u, v = Rb[:, 0] @ c, Rb[:, 1] @ c
                    if e.shape == "cylinder":
                        ok &= math.hypot(u, v) <= hx - 0.5 * CUP_R
                    else:
                        ok &= abs(u) <= hx - 0.5 * CUP_R and abs(v) <= hy - 0.5 * CUP_R
            if ok:
                return name
        return None

    def _attach(self, name):
        b = self.boxes[name]
        m, d = self.model, self.data
        tool = m.body("tool").id
        # weld: relpose = pose of body2 in body1 frame
        p1, R1 = d.xpos[tool], d.xmat[tool].reshape(3, 3)
        p2, R2 = d.xpos[b["body"]], d.xmat[b["body"]].reshape(3, 3)
        rel_p = R1.T @ (p2 - p1)
        rel_R = R1.T @ R2
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, rel_R.reshape(-1))
        data = np.zeros(m.eq_data.shape[1])
        data[3:6] = rel_p
        data[6:10] = q
        data[10] = 1.0
        m.eq_data[b["eq"]] = data
        d.eq_active[b["eq"]] = 1
        self.held = name

    def _detach(self):
        if self.held is not None:
            self.data.eq_active[self.boxes[self.held]["eq"]] = 0
            self._released[self.held] = True
            self.held = None

    def _update_suction(self):
        if self.held is None:
            cand = self._seal_candidate() if self.vac_cmd > 0.5 else None
            if cand is not None:
                self.seal_t += self.dt
                if self.seal_t >= GRIP_TIME_S:
                    self._attach(cand)
                    self.seal_t = 0.0
            else:
                self.seal_t = 0.0
            self.release_t = None
        else:
            if self.vac_cmd < 0.5:  # vacuum decays over RELEASE_TIME_S
                self.release_t = (self.release_t or 0.0) + self.dt
                if self.release_t >= RELEASE_TIME_S:
                    self._detach()
                    self.release_t = None
            else:
                self.release_t = None

    # --------------------------------------------------------------- occluder
    def _update_occluder(self):
        mid = self.model.body("occluder").mocapid[0]
        on = self.cfg.occlusion_s > 0 and self.occ_start <= self.t < self.occ_start + self.cfg.occlusion_s
        if on:
            tgt, _ = self.box_pose(self.task.steps[0].onto)
            self.data.mocap_pos[mid] = [1.0, tgt[1], 0.2]
        else:
            self.data.mocap_pos[mid] = [1.0, 0.0, -2.0]
        self._occ_active = on

    # ------------------------------------------------------------------- done
    def _check_done(self):
        st = self.task.status(self)
        self._task_status = st
        ok = st["all_on"] and self.held is None and not (self.cfg.require_unmoved and st["collateral"])
        self._stable_t = self._stable_t + self.dt if ok else 0.0
        success = self._stable_t >= self.cfg.success_hold_s
        failed = bool(self.cfg.require_unmoved and st["collateral"])
        for name in self.task.involved:
            p, _ = self.entity_pose(name)
            if p[2] < -0.1 or abs(self.belt_coords(p[:2])[1]) > 1.15:  # fell off / ran off the belt
                failed = True
        for step in self.task.steps:  # released but resting on the ground instead of the target
            e = self.ent[step.obj]
            p, _ = self.entity_pose(step.obj)
            u, _ = self.belt_coords(p[:2])
            base = self.belt_top if abs(u) < 0.15 else self.table_z
            if self._released[step.obj] and self.held is None and not st["step_ok"][self.task.steps.index(step)]:
                failed |= bool(p[2] - e.hz < base + 0.01)
        truncated = self.t >= self.cfg.max_time_s
        return bool(success or failed), bool(truncated and not (success or failed)), bool(success)

    # ---------------------------------------------------------- observations
    def render(self, camera):
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def proprio(self):
        d = self.data
        tcp, _ = self.tcp_pose()
        yaw = self.tcp_yaw()
        return np.concatenate(
            [
                d.qpos[self.arm_qadr],
                d.qvel[self.arm_dadr],
                tcp,
                [math.sin(yaw), math.cos(yaw)],
                self.ee_cmd,
                [self.vac_cmd, float(self.held is not None)],
            ]
        ).astype(np.float32)

    def _sensor(self, img):
        """Camera sensor model: exposure, optional blur, per-frame pixel noise (per-episode parameters)."""
        p = self.img_params
        if p["noise"] == 0 and p["blur"] == 0 and p["gain"] == 1.0 and p["bias"] == 0.0:
            return img
        x = img.astype(np.float32) / 255.0
        if p["blur"] > 0:
            pad = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode="edge")
            mean = sum(pad[i : i + x.shape[0], j : j + x.shape[1]] for i in range(3) for j in range(3)) / 9.0
            x = (1 - p["blur"]) * x + p["blur"] * mean
        x = x * p["gain"] + p["bias"]
        if p["noise"] > 0:
            x = x + self._img_rng.normal(0.0, p["noise"], x.shape).astype(np.float32)
        return (np.clip(x, 0, 1) * 255.0 + 0.5).astype(np.uint8)

    def _obs(self):
        imgs = {}
        if self.renderer:
            for c in self.cfg.cameras:
                self._frames[c].append(self.render(c))
                imgs[c] = self._sensor(self._frames[c][0])  # oldest kept frame = obs_delay_steps old
        return {"images": imgs, "proprio": self.proprio(), "instruction": self.instruction}

    def _entity_state(self, name):
        e = self.ent[name]
        p, R = self.entity_pose(name)
        v = self.entity_vel(name)
        return {
            "pos": p,
            "vel": v[:2].copy(),
            "yaw": float(math.atan2(R[1, 0], R[0, 0])),
            "hz": e.hz,
            "extent": (e.dims[0], e.dims[1]),
            "shape": e.shape,
            "category": e.category,
            "color": e.color,
            "location": e.location,
        }

    def _info(self):
        st = self.task.status(self)
        step = self.task.current_step(self)
        cur = self.task.steps[min(step, len(self.task.steps) - 1)]
        ents = {n: self._entity_state(n) for n in self.ent}
        return {
            "t": self.t,
            "priv": {
                "entities": ents,
                "steps": [(s.obj, s.onto) for s in self.task.steps],
                "step": step,
                "subgoal": self.task.steps[min(step, len(self.task.steps) - 1)].text,
                "task_kind": self.task.kind,
                "step_ok": st["step_ok"],
                "collateral": st["collateral"],
                # legacy keys: the current step's object and support
                "src_pos": ents[cur.obj]["pos"],
                "tgt_pos": ents[cur.onto]["pos"],
                "tgt_vel": ents[cur.onto]["vel"],
                "belt_vel": self.belt_dir * self.belt_velocity(),
                "belt_dir": self.belt_dir.copy(),
                "table_z": self.table_z,
                "belt_cmd": float(self.data.ctrl[self.belt_act]),
                "held": self.held,
                "stack_supported": bool(st["all_on"]),
                "stack_offset": float(st["offsets"][-1]),
                "occluded": self._occ_active,
                "yaw_cmd": self.yaw_cmd,
            },
        }

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
