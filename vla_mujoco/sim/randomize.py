"""Per-episode domain randomisation applied directly to the compiled MuJoCo model.

Nominal values are captured once, so every ``apply`` samples around the nominal scene (no drift).
All magnitudes scale with ``scale`` (1.0 = training range; use >1 for out-of-distribution tests).

Groups (toggled independently through ``EnvConfig``):

* **conveyor**   - belt centre x, belt yaw, belt colour/rail colour
* **camera**     - front/overhead/wrist position, orientation, field of view
* **appearance** - object colour jitter, object size, table/floor/robot tint, lighting
"""

from __future__ import annotations

import colorsys
import math

import mujoco
import numpy as np

from .scene import BELT_X


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def _quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def _small_rot_quat(rng, deg: float) -> np.ndarray:
    """Random small rotation (axis uniform, angle ~ U(-deg, deg))."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-9
    ang = math.radians(rng.uniform(-deg, deg))
    return np.concatenate([[math.cos(ang / 2)], math.sin(ang / 2) * axis])


def _hsv(rng, h=(0, 1), s=(0, 1), v=(0, 1)):
    return np.array(colorsys.hsv_to_rgb(rng.uniform(*h), rng.uniform(*s), rng.uniform(*v)))


class DomainRandomizer:
    def __init__(self, model: mujoco.MjModel):
        self.m = m = model
        self.geom = {n: m.geom(n).id for n in ("table", "belt_frame", "rail_l", "rail_r", "belt_top")}
        self.belt_body = m.body("belt").id
        self.focus = m.body("focus").id
        self.cams = {n: m.camera(n).id for n in ("front", "overhead", "wrist")}
        self.mat_floor, self.mat_belt = m.material("floor").id, m.material("belt").id
        self.arm_geoms = [i for i in range(m.ngeom) if m.geom_contype[i] == 1 and m.geom_conaffinity[i] == 2]
        # nominal snapshots
        self.nom = dict(
            belt_pos=m.body_pos[self.belt_body].copy(),
            geom_pos={k: m.geom_pos[i].copy() for k, i in self.geom.items()},
            geom_rgba={k: m.geom_rgba[i].copy() for k, i in self.geom.items()},
            arm_rgba=m.geom_rgba[self.arm_geoms].copy(),
            cam_pos=m.cam_pos.copy(),
            cam_quat=m.cam_quat.copy(),
            cam_fovy=m.cam_fovy.copy(),
            focus_pos=m.body_pos[self.focus].copy(),
            mat_rgba=m.mat_rgba.copy(),
            light_pos=m.light_pos.copy(),
            light_diffuse=m.light_diffuse.copy(),
            head_amb=m.vis.headlight.ambient.copy(),
            head_dif=m.vis.headlight.diffuse.copy(),
        )

    # ----------------------------------------------------------------- conveyor
    def conveyor(self, rng, scale=1.0, enabled=True):
        """Returns (belt_x, belt_yaw)."""
        m, nom = self.m, self.nom
        bx = BELT_X + (rng.uniform(-0.03, 0.03) * scale if enabled else 0.0)
        yaw = rng.uniform(-0.10, 0.10) * scale if enabled else 0.0
        R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        q = _yaw_quat(yaw)
        c = np.array([bx, 0.0])
        m.body_pos[self.belt_body] = [bx, 0.0, nom["belt_pos"][2]]
        m.body_quat[self.belt_body] = q
        for k in ("belt_frame", "rail_l", "rail_r"):
            i = self.geom[k]
            local = nom["geom_pos"][k][:2] - np.array([BELT_X, 0.0])
            m.geom_pos[i, :2] = c + R @ local
            m.geom_pos[i, 2] = nom["geom_pos"][k][2]
            m.geom_quat[i] = q
            # MuJoCo skips the local rotation for geoms compiled with the body's orientation
            m.geom_sameframe[i] = mujoco.mjtSameFrame.mjSAMEFRAME_NONE
        return float(bx), float(yaw)

    # ------------------------------------------------------------------- camera
    def cameras(self, rng, scale=1.0, enabled=True):
        m, nom = self.m, self.nom
        m.cam_pos[:] = nom["cam_pos"]
        m.cam_quat[:] = nom["cam_quat"]
        m.cam_fovy[:] = nom["cam_fovy"]
        m.body_pos[self.focus] = nom["focus_pos"]
        if not enabled:
            return
        f, o, w = self.cams["front"], self.cams["overhead"], self.cams["wrist"]
        m.cam_pos[f] = nom["cam_pos"][f] + rng.normal(0, 1, 3) * np.array([0.08, 0.07, 0.06]) * scale
        m.body_pos[self.focus] = nom["focus_pos"] + rng.normal(0, 0.03, 3) * scale
        m.cam_fovy[f] += rng.uniform(-6, 6) * scale
        m.cam_pos[o] = nom["cam_pos"][o] + rng.normal(0, 1, 3) * np.array([0.08, 0.08, 0.08]) * scale
        m.cam_quat[o] = _quat_mul(nom["cam_quat"][o], _small_rot_quat(rng, 4 * scale))
        m.cam_fovy[o] += rng.uniform(-6, 6) * scale
        m.cam_pos[w] = nom["cam_pos"][w] + rng.normal(0, 0.008, 3) * scale
        m.cam_quat[w] = _quat_mul(nom["cam_quat"][w], _small_rot_quat(rng, 3 * scale))
        m.cam_fovy[w] += rng.uniform(-5, 5) * scale

    # --------------------------------------------------------------- appearance
    @staticmethod
    def jitter_color(rng, rgb, scale=1.0):
        c = np.asarray(rgb, float) * rng.uniform(1 - 0.15 * scale, 1 + 0.15 * scale)
        return np.clip(c + rng.normal(0, 0.04 * scale, 3), 0.02, 1.0)

    def appearance(self, rng, scale=1.0, enabled=True):
        """Tint scene surfaces and lighting. Object colours/sizes are handled by the env."""
        m, nom = self.m, self.nom
        for k, i in self.geom.items():
            m.geom_rgba[i] = nom["geom_rgba"][k]
        m.geom_rgba[self.arm_geoms] = nom["arm_rgba"]
        m.mat_rgba[:] = nom["mat_rgba"]
        m.light_pos[:] = nom["light_pos"]
        m.light_diffuse[:] = nom["light_diffuse"]
        m.vis.headlight.ambient[:] = nom["head_amb"]
        m.vis.headlight.diffuse[:] = nom["head_dif"]
        if not enabled:
            return
        m.geom_rgba[self.geom["table"], :3] = _hsv(rng, s=(0.05, 0.55), v=(0.4, 0.9))
        m.mat_rgba[self.mat_floor, :3] = _hsv(rng, s=(0.0, 0.25), v=(0.55, 1.0))
        m.mat_rgba[self.mat_belt, :3] = np.clip(
            rng.uniform(0.6, 1.5, 3) * 0.5 + 0.5 * rng.uniform(0.7, 1.3), 0.3, 1.5
        )
        rail = _hsv(rng, s=(0.4, 1.0), v=(0.5, 1.0))
        m.geom_rgba[self.geom["rail_l"], :3] = rail
        m.geom_rgba[self.geom["rail_r"], :3] = rail
        tint = _hsv(rng, s=(0.0, 0.18), v=(0.75, 1.0))
        m.geom_rgba[self.arm_geoms, :3] = np.clip(nom["arm_rgba"][:, :3] * tint, 0, 1)
        # lighting
        m.light_pos[:, :] = nom["light_pos"] + rng.normal(0, 0.3 * scale, nom["light_pos"].shape)
        m.light_diffuse[:] = nom["light_diffuse"] * rng.uniform(0.6, 1.3, (m.nlight, 1))
        m.vis.headlight.ambient[:] = rng.uniform(0.25, 0.6, 3) * 0.5 + rng.uniform(0.15, 0.3)
        m.vis.headlight.diffuse[:] = rng.uniform(0.3, 0.7)

    # ------------------------------------------------------------------ objects
    def resize_box(self, geom_id: int, body_id: int, half: float):
        """Resize a cube at runtime, keeping inertia/bounds consistent."""
        m = self.m
        m.geom_size[geom_id] = half
        m.geom_rbound[geom_id] = half * math.sqrt(3)
        m.geom_aabb[geom_id] = [0, 0, 0, half, half, half]
        s = 2 * half
        m.body_inertia[body_id] = m.body_mass[body_id] * s * s / 6.0
