"""Scripted privileged expert: executes the task's steps with generic Pick / Place skills.

Every step is "put entity A onto entity B", where either object may be static (table) or moving (belt).
Phases per step::

    approach -> descend (grasp) -> lift -> transfer -> track (descend onto the support) -> release -> retreat

* *approach / transfer* solve a constant-velocity intercept to a hover point above the (possibly moving) object;
* *descend / track* are closed loop on the live pose with the object's velocity as feed-forward;
* hover height is derived from the tallest object in the scene and the height of the held object;
* the place height comes from the support query (highest surface under the target), so layers work;
* release is timed to absorb the VGC10 vacuum-release delay;
* a dropped or never-sealed object sends the expert back to *approach* (re-grasp).

It reads ground truth from ``info["priv"]`` (and the env for planning queries) and emits the same 5-D normalised
velocity action a learned policy produces, so demonstrations are directly usable for behaviour cloning.
"""

from __future__ import annotations

import numpy as np

from .sim.env import MAX_SPEED_M_S, MAX_YAW_RATE, RELEASE_TIME_S, DynamicStackEnv
from .tasks.predicates import support_height

HOVER_Z = 0.36
PLACE_GAP = 0.004  # object bottom above the support at release (m)
GRASP_TIMEOUT_S = 6.0


def _wrap_quarter(yaw):
    """Wrap yaw to (-pi/4, pi/4]: boxes and cylinders are symmetric under 90 deg rotations."""
    return (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4


class StackExpert:
    def __init__(self, env: DynamicStackEnv, seed: int = 0, noise: float = 0.0):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.noise = noise  # std of Gaussian action noise (DART-style) in action units
        self.reset()

    def reset(self):
        self.phase = "approach"
        self.phase_t = 0.0
        self.step_idx = 0
        self.regrasps = 0

    # ----------------------------------------------------------------- helpers
    def _move(self, goal, speed, yaw_goal=None, yaw_speed=1.2):
        """Action that moves the commanded pose toward ``goal`` at <= ``speed`` m/s."""
        env = self.env
        delta = goal - env.ee_cmd_future  # plan from the pose after in-flight (delayed) actions land
        step = speed * env.dt
        n = np.linalg.norm(delta)
        if n > step:
            delta = delta * step / n
        a = delta / (MAX_SPEED_M_S * env.dt)
        dyaw = 0.0
        if yaw_goal is not None:
            e = (yaw_goal - env.yaw_cmd_future + np.pi) % (2 * np.pi) - np.pi
            dyaw = np.clip(e, -yaw_speed * env.dt, yaw_speed * env.dt) / (MAX_YAW_RATE * env.dt)
        return a, dyaw

    def _enter(self, phase):
        self.phase, self.phase_t = phase, 0.0

    @staticmethod
    def _hover_z(ents, held_h):
        """TCP height that clears every object (and the held one hanging below the TCP)."""
        top = max((e["pos"][2] + e["hz"] for e in ents.values()), default=0.0)
        return float(max(HOVER_Z, top + held_h + 0.06))

    def _intercept(self, pos, vel, z):
        """Hover point above a (possibly moving) object that the TCP reaches when the object does."""
        cmd = self.env.ee_cmd_future
        t_arr = 0.0
        goal = np.array([pos[0], pos[1], z])
        for _ in range(6):
            goal = np.array([pos[0] + vel[0] * t_arr, pos[1] + vel[1] * t_arr, z])
            t_arr = np.linalg.norm(goal - cmd) / 0.35
        return goal

    def _track_xy(self, pos, vel):
        """Closed-loop xy goal with feed-forward; commands land act_delay_s later while the object moves."""
        look = 2.0 * self.env.dt + self.env.act_delay_s
        return np.asarray(pos[:2]) + np.asarray(vel[:2]) * look

    # --------------------------------------------------------------------- act
    def act(self, info) -> np.ndarray:
        env, P = self.env, info["priv"]
        self.phase_t += env.dt
        ents = P["entities"]
        steps = P["steps"]
        vac = 0.0
        cmd = env.ee_cmd_future

        if self.phase == "approach" and P["step"] != self.step_idx:
            self.step_idx = P["step"]
        if self.step_idx >= len(steps):  # all predicates hold: back off and wait for the stability check
            a, dyaw = self._move(np.array([cmd[0], cmd[1], self._hover_z(ents, 0.0)]), 0.25, None)
            return self._finish(a, dyaw, 0.0)

        obj, onto = steps[self.step_idx]
        o, t = ents[obj], ents[onto]
        o_pos, o_vel, o_hz = o["pos"], np.append(o["vel"], 0.0), o["hz"]
        t_pos, t_vel = t["pos"], np.append(t["vel"], 0.0)
        yaw_o = _wrap_quarter(o["yaw"])
        yaw_t = _wrap_quarter(t["yaw"])
        held_h = 2 * o_hz  # the object's bottom hangs this far below the TCP
        o_top = np.array([o_pos[0], o_pos[1], o_pos[2] + o_hz])
        others = {k: v for k, v in ents.items() if k != obj}

        # lost the object while carrying it: go back and grasp again
        if self.phase in ("lift", "transfer", "track") and P["held"] != obj:
            self.regrasps += 1
            self._enter("approach")

        if self.phase == "approach":
            goal = self._intercept(o_pos, o_vel, self._hover_z(others, 0.0))
            a, dyaw = self._move(goal, 0.30, yaw_o)
            near_xy = np.linalg.norm((cmd - goal)[:2]) < 8e-3
            if near_xy and abs(cmd[2] - goal[2]) < 4e-3 and abs(yaw_o - env.yaw_cmd_future) < 0.03:
                self._enter("descend")
        elif self.phase == "descend":
            xy = self._track_xy(o_pos, o_vel)
            err_xy = np.linalg.norm(xy - cmd[:2])
            z_goal = o_top[2] + 0.0005
            vz = -0.10 if err_xy < 0.010 else 0.0
            dz = max(z_goal - cmd[2], vz * env.dt)
            a, dyaw = self._move(np.array([xy[0], xy[1], cmd[2] + dz]), 0.6, yaw_o)
            gap = cmd[2] - o_top[2]
            vac = 1.0 if gap < 0.012 else 0.0
            if P["held"] == obj:
                self._enter("lift")
            elif self.phase_t > GRASP_TIMEOUT_S:  # never sealed (e.g. object slipped away): retry
                self.regrasps += 1
                self._enter("approach")
        elif self.phase == "lift":
            vac = 1.0
            z = self._hover_z(others, held_h)
            a, dyaw = self._move(np.array([cmd[0], cmd[1], z]), 0.35, yaw_t)
            if abs(cmd[2] - z) < 3e-3:
                self._enter("transfer")
        elif self.phase == "transfer":
            vac = 1.0
            goal = self._intercept(t_pos, t_vel, self._hover_z(others, held_h))
            a, dyaw = self._move(goal, 0.35, yaw_t)
            if np.linalg.norm((cmd - goal)[:2]) < 6e-3:
                self._enter("track")
        elif self.phase == "track":
            vac = 1.0
            place_z = support_height(env, t_pos[:2], exclude=obj) + held_h + PLACE_GAP
            xy = self._track_xy(t_pos, t_vel)
            err_xy = np.linalg.norm(xy - cmd[:2])
            descend_ok = err_xy < 0.010
            vz = -0.10 if descend_ok else 0.0
            dz = max(place_z - cmd[2], vz * env.dt)
            if cmd[2] <= place_z + 1e-4:
                dz = place_z - cmd[2]
            a, dyaw = self._move(np.array([xy[0], xy[1], cmd[2] + dz]), 0.6, yaw_t)
            gap = cmd[2] - place_z
            # command release RELEASE_TIME_S early so the object lets go right at place_z
            if descend_ok and gap < abs(vz) * RELEASE_TIME_S + 0.002:
                self._enter("release")
        elif self.phase == "release":
            place_z = support_height(env, t_pos[:2], exclude=obj) + held_h + PLACE_GAP
            xy = self._track_xy(t_pos, t_vel)
            a, dyaw = self._move(np.array([xy[0], xy[1], max(place_z, cmd[2])]), 0.6, yaw_t)
            if P["held"] is None and self.phase_t > 0.1:
                self._enter("retreat")
        else:  # retreat to hover height, then start the next step
            z = self._hover_z(ents, 0.0)
            a, dyaw = self._move(np.array([cmd[0], cmd[1], z]), 0.30, None)
            if abs(cmd[2] - z) < 4e-3:
                self._enter("approach")
        return self._finish(a, dyaw, vac)

    def _finish(self, a, dyaw, vac):
        act = np.array([a[0], a[1], a[2], dyaw, vac], dtype=np.float32)
        if self.noise > 0:
            act[:4] += self.rng.normal(0, self.noise, 4)
        act[:4] = np.clip(act[:4], -1, 1)
        return act
