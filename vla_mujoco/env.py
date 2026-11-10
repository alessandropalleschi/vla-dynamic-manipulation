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

    [dx, dy, dz, dyaw, vacuum]   Cartesian end-effector deltas per control step
    (base frame) + vacuum command in [0, 1] (> 0.5 = on).

``info["priv"]`` carries privileged ground truth (box poses, belt velocity,
contact flags) for the expert, auxiliary losses and evaluation only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import mujoco
import numpy as np

from .scene import BELT_TOP, BELT_X, CUP_R, CUP_XY, SRC_HALF, TGT_HALF, build_mjcf

PROPRIO_DIM = 22
MAX_STEP_M = 0.015       # per control step at action = 1  (0.3 m/s at 20 Hz)
MAX_STEP_YAW = 0.10
WORKSPACE_LO = np.array([0.15, -0.60, 0.06])   # TCP bounds (m)
WORKSPACE_HI = np.array([0.80, 0.60, 0.60])
# VGC10 timing (config/vgc10.yaml): vacuum build-up / release delays
GRIP_TIME_S = 0.35
RELEASE_TIME_S = 0.20

PALETTE = {
    "red": (0.85, 0.15, 0.15), "green": (0.15, 0.70, 0.25), "blue": (0.15, 0.30, 0.85),
    "yellow": (0.90, 0.80, 0.10), "orange": (0.95, 0.50, 0.10), "purple": (0.55, 0.20, 0.75),
    "cyan": (0.10, 0.75, 0.80), "pink": (0.95, 0.45, 0.70),
}
HOME_Q = np.array([0.0, 0.0, -1.2, 0.0, 1.2, 0.0])  # refined at construction by IK


@dataclass
class EnvConfig:
    seed: int = 0
    img_size: int = 128
    cameras: tuple = ("front", "wrist")
    control_hz: float = 20.0
    max_time_s: float = 22.0
    belt_speed_range: tuple = (0.02, 0.07)      # m/s
    speed_change: bool = False                  # surprise belt-speed change mid-episode
    n_distractors: int = 1                      # extra box on the belt (language must disambiguate)
    occlusion_s: float = 0.0                    # >0: a panel hides the target from `front` for this long
    randomize_colors: bool = True
    randomize_physics: bool = True
    render: bool = True
    success_hold_s: float = 0.5
    place_tol_m: float = 0.012
    extra: dict = field(default_factory=dict)


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
        self.boxes = {n: self._box_info(n) for n in ("src", "tgt", "dist")}
        self.renderer = (mujoco.Renderer(m, self.cfg.img_size, self.cfg.img_size)
                         if self.cfg.render else None)
        self.home_q = self._solve_home()
        self.reset(seed=self.cfg.seed)

    # ------------------------------------------------------------------ setup
    def _box_info(self, name):
        m = self.model
        b = m.body(name)
        return dict(body=b.id, qadr=m.joint(f"{name}_free").qposadr[0],
                    dadr=m.joint(f"{name}_free").dofadr[0], geom=int(m.body_geomadr[b.id]),
                    eq=m.equality(f"grasp_{name}").id)

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
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
            n = np.abs(dq).max()
            if n > max_dq:
                dq *= max_dq / n
            q = np.clip(q + dq, self.q_lo, self.q_hi)
        p, R = self._fk(q)
        err = np.linalg.norm(pos - p)
        return q, err < 2e-3

    # ------------------------------------------------------------------ state
    def tcp_pose(self):
        d = self.data
        return d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()

    def tcp_yaw(self):
        _, R = self.tcp_pose()
        return math.atan2(R[1, 0], R[0, 0])

    def box_pose(self, name):
        b = self.boxes[name]
        return self.data.xpos[b["body"]].copy(), self.data.xmat[b["body"]].reshape(3, 3).copy()

    def box_half(self, name):
        return float(self.model.geom_size[self.boxes[name]["geom"]][0])

    def belt_velocity(self):
        return float(self.data.qvel[self.model.joint("belt_slide").dofadr[0]])

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
            self.speed_after = float(np.clip(self.belt_speed * r.uniform(0.5, 1.6),
                                             cfg.belt_speed_range[0], cfg.belt_speed_range[1] * 1.3))
        names = list(PALETTE)
        if cfg.randomize_colors:
            src_c, tgt_c, dist_c = r.choice(len(names), 3, replace=False)
        else:
            src_c, tgt_c, dist_c = 0, 2, 1
        self.src_color, self.tgt_color, self.dist_color = names[src_c], names[tgt_c], names[dist_c]
        for n, c in (("src", self.src_color), ("tgt", self.tgt_color), ("dist", self.dist_color)):
            m.geom_rgba[self.boxes[n]["geom"]] = (*PALETTE[c], 1.0)
        if cfg.randomize_physics:
            m.geom_friction[self.boxes["src"]["geom"], 0] = r.uniform(0.7, 1.2)
            m.geom_friction[self.boxes["tgt"]["geom"], 0] = r.uniform(0.7, 1.2)
            m.body_mass[self.boxes["src"]["body"]] = r.uniform(0.15, 0.35)
        self.instruction = (f"stack the {self.src_color} block on the {self.tgt_color} box "
                            f"moving on the conveyor")

        # source block on the table (right of the belt, reachable), random heading
        self.src_yaw = float(r.uniform(-0.5, 0.5))
        sx, sy = r.uniform(0.24, 0.31), r.uniform(0.18, 0.38)
        self._set_box("src", (sx, sy, SRC_HALF + 0.0005), self.src_yaw)
        # target + distractor on the belt, upstream (negative y)
        # start further upstream on fast belts so the target is still reachable after the pick
        ty = float(opts.get("target_y", r.uniform(-0.55, -0.35) - 4.0 * max(0.0, self.belt_speed - 0.05)))
        tx = BELT_X + r.uniform(-0.03, 0.03)
        self._set_box("tgt", (tx, ty, BELT_TOP + TGT_HALF + 0.0005), r.uniform(-0.15, 0.15))
        if cfg.n_distractors > 0:
            side = r.choice([-1.0, 1.0])
            self._set_box("dist", (BELT_X + r.uniform(-0.03, 0.03), ty + side * r.uniform(0.25, 0.32),
                                   BELT_TOP + TGT_HALF + 0.0005), r.uniform(-0.15, 0.15))
        else:
            self._set_box("dist", (0.2, -0.6, TGT_HALF + 0.0005), 0.0)

        # robot at home, commanded pose = home pose
        d.qpos[self.arm_qadr] = self.home_q
        d.ctrl[:6] = self.home_q
        mujoco.mj_forward(m, d)
        p, _ = self.tcp_pose()
        self.ee_cmd = p.copy()
        self.yaw_cmd = 0.0
        self.vac_cmd = 0.0
        self.held = None
        self.seal_t = 0.0
        self.release_t = None
        self.t = 0.0
        self.steps = 0
        self._stable_t = 0.0
        self._released_once = False
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
        d.qpos[b["qadr"]: b["qadr"] + 3] = pos
        d.qpos[b["qadr"] + 3: b["qadr"] + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        d.qvel[b["dadr"]: b["dadr"] + 6] = 0

    # ------------------------------------------------------------------- step
    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        assert a.shape[0] == 5, "action must be [dx, dy, dz, dyaw, vacuum]"
        m, d = self.model, self.data
        self.ee_cmd = np.clip(self.ee_cmd + np.clip(a[:3], -1, 1) * MAX_STEP_M, WORKSPACE_LO, WORKSPACE_HI)
        self.yaw_cmd = float(np.clip(self.yaw_cmd + np.clip(a[3], -1, 1) * MAX_STEP_YAW, -2.0, 2.0))
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
        for name in ("src", "tgt", "dist"):
            pos, Rb = self.box_pose(name)
            n = Rb[:, 2]
            half = self.box_half(name)
            top = pos + n * half
            gap = float((tcp - top) @ n)
            if not (-0.004 <= gap <= 0.008 and float(tool_z @ n) < -0.97):
                continue
            # all four cup centres (VGC10: +-CUP_XY) must lie on the top face, with >= half a cup radius of margin
            ok = True
            for sx in (-1, 1):
                for sy in (-1, 1):
                    c = tcp + R @ np.array([sx * CUP_XY, sy * CUP_XY, 0.0]) - top
                    u, v = Rb[:, 0] @ c, Rb[:, 1] @ c
                    ok &= abs(u) <= half - 0.5 * CUP_R and abs(v) <= half - 0.5 * CUP_R
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
            self.held = None
            self._released_once = True

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
            if self.vac_cmd < 0.5:                       # vacuum decays over RELEASE_TIME_S
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
            tgt, _ = self.box_pose("tgt")
            self.data.mocap_pos[mid] = [1.0, tgt[1], 0.2]
        else:
            self.data.mocap_pos[mid] = [1.0, 0.0, -2.0]
        self._occ_active = on

    # ------------------------------------------------------------------- done
    def stack_status(self):
        """(supported, xy_offset): is `src` resting on top of `tgt`?"""
        d = self.data
        sb, tb = self.boxes["src"]["geom"], self.boxes["tgt"]["geom"]
        touching = False
        for i in range(d.ncon):
            c = d.contact[i]
            if {int(c.geom1), int(c.geom2)} == {sb, tb} and c.dist < 1e-3:
                touching = True
                break
        sp, _ = self.box_pose("src")
        tp, _ = self.box_pose("tgt")
        off = float(np.linalg.norm(sp[:2] - tp[:2]))
        above = sp[2] - tp[2] > 0.5 * (SRC_HALF + TGT_HALF)
        return bool(touching and above), off

    def _check_done(self):
        supported, off = self.stack_status()
        ok = supported and off <= self.cfg.place_tol_m and self.held is None
        self._stable_t = self._stable_t + self.dt if ok else 0.0
        success = self._stable_t >= self.cfg.success_hold_s
        sp, _ = self.box_pose("src")
        tp, _ = self.box_pose("tgt")
        failed = (sp[2] < -0.05 or tp[2] < BELT_TOP - 0.02 or abs(tp[1]) > 1.1
                  or (self._released_once and self.held is None and not supported
                      and sp[2] < SRC_HALF + 0.01))
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
        return np.concatenate([
            d.qpos[self.arm_qadr], d.qvel[self.arm_dadr], tcp, [math.sin(yaw), math.cos(yaw)],
            self.ee_cmd, [self.vac_cmd, float(self.held is not None)],
        ]).astype(np.float32)

    def _obs(self):
        imgs = {c: self.render(c) for c in self.cfg.cameras} if self.renderer else {}
        return {"images": imgs, "proprio": self.proprio(), "instruction": self.instruction}

    def _info(self):
        tp, _ = self.box_pose("tgt")
        sp, _ = self.box_pose("src")
        dp, _ = self.box_pose("dist")
        supported, off = self.stack_status()
        return {"t": self.t, "priv": {
            "src_pos": sp, "tgt_pos": tp, "dist_pos": dp, "src_yaw": self.src_yaw,
            "belt_vel": np.array([0.0, self.belt_velocity()]),
            "belt_cmd": float(self.data.ctrl[self.belt_act]),
            "held": self.held, "stack_supported": supported, "stack_offset": off,
            "occluded": self._occ_active, "yaw_cmd": self.yaw_cmd,
        }}

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
