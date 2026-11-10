"""Scripted privileged expert for dynamic stacking (demonstration source).

It reads ground-truth box poses and belt velocity from ``info["priv"]`` and emits the
same 5-D action a learned policy would, so demonstrations are directly usable for
behaviour cloning. Phases:

    approach -> descend/grasp -> lift -> transfer (intercept) -> track+descend -> release -> retreat

The transfer phase solves a constant-velocity intercept (target position at the time the
tool arrives); the final descent is closed-loop on the live target pose with belt
feed-forward, and the release is timed to absorb the VGC10 vacuum-release delay.
"""

from __future__ import annotations

import numpy as np

from .sim.env import MAX_STEP_M, MAX_STEP_YAW, RELEASE_TIME_S, DynamicStackEnv

HOVER_Z = 0.36
PLACE_GAP = 0.004  # block bottom above target top at release (m)


def _wrap_quarter(yaw):
    """Wrap yaw to (-pi/4, pi/4]: a cube is symmetric under 90 deg rotations."""
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
        self.done_wait = 0.0

    # ----------------------------------------------------------------- helpers
    def _move(self, goal, speed, yaw_goal=None, yaw_speed=1.2):
        """Action that moves the commanded pose toward ``goal`` at <= ``speed`` m/s."""
        env = self.env
        delta = goal - env.ee_cmd
        step = speed * env.dt
        n = np.linalg.norm(delta)
        if n > step:
            delta = delta * step / n
        a = delta / MAX_STEP_M
        dyaw = 0.0
        if yaw_goal is not None:
            e = (yaw_goal - env.yaw_cmd + np.pi) % (2 * np.pi) - np.pi
            dyaw = np.clip(e, -yaw_speed * env.dt, yaw_speed * env.dt) / MAX_STEP_YAW
        return a, dyaw

    def _enter(self, phase):
        self.phase, self.phase_t = phase, 0.0

    # --------------------------------------------------------------------- act
    def act(self, info) -> np.ndarray:
        env, P = self.env, info["priv"]
        self.phase_t += env.dt
        vac = 0.0
        src, tgt = P["src_pos"], P["tgt_pos"]
        v = np.array([P["belt_vel"][0], P["belt_vel"][1], 0.0])
        yaw_src = _wrap_quarter(P["src_yaw"])
        src_h, tgt_h = P["src_half"], P["tgt_half"]
        yaw_tgt = _wrap_quarter(P["tgt_yaw"])  # align the held block with the target box
        held_h = 2 * src_h  # block bottom is this far below the TCP
        tcp_to_src_top = np.array([src[0], src[1], src[2] + src_h])

        if self.phase == "approach":
            goal = tcp_to_src_top + [0, 0, 0.10]
            a, dyaw = self._move(goal, 0.30, yaw_src)
            if np.linalg.norm(goal - env.ee_cmd) < 3e-3 and abs(yaw_src - env.yaw_cmd) < 0.02:
                self._enter("descend")
        elif self.phase == "descend":
            goal = tcp_to_src_top + [0, 0, 0.0005]
            a, dyaw = self._move(goal, 0.10, yaw_src)
            gap = env.ee_cmd[2] - tcp_to_src_top[2]
            vac = 1.0 if gap < 0.012 else 0.0
            if gap < 0.002:
                vac = 1.0
                if P["held"] == "src":
                    self._enter("lift")
        elif self.phase == "lift":
            vac = 1.0
            goal = np.array([env.ee_cmd[0], env.ee_cmd[1], HOVER_Z])
            a, dyaw = self._move(goal, 0.35, yaw_tgt)
            if abs(env.ee_cmd[2] - HOVER_Z) < 3e-3:
                self._enter("transfer")
        elif self.phase == "transfer":
            vac = 1.0
            # constant-velocity intercept: where will the target be when we arrive?
            top = np.array([tgt[0], tgt[1], tgt[2] + tgt_h + held_h + 0.08])
            t_arr = 0.0
            for _ in range(6):
                goal = np.array([tgt[0] + v[0] * t_arr, tgt[1] + v[1] * t_arr, HOVER_Z])
                t_arr = np.linalg.norm(goal - env.ee_cmd) / 0.35
            goal[2] = HOVER_Z
            a, dyaw = self._move(goal, 0.35, yaw_tgt)
            if np.linalg.norm((env.ee_cmd - goal)[:2]) < 6e-3:
                self._enter("track")
            _ = top
        elif self.phase == "track":
            vac = 1.0
            place_z = tgt[2] + tgt_h + held_h + PLACE_GAP
            # closed-loop xy tracking with belt feed-forward; descend only when aligned
            lookahead = 2.0 * env.dt
            xy_goal = tgt[:2] + v[:2] * lookahead
            err_xy = np.linalg.norm(xy_goal - env.ee_cmd[:2])
            descend_ok = err_xy < 0.010
            vz = -0.10 if descend_ok else 0.0
            dz = max(place_z - env.ee_cmd[2], vz * env.dt)
            if env.ee_cmd[2] <= place_z + 1e-4:
                dz = place_z - env.ee_cmd[2]
            goal = np.array([xy_goal[0], xy_goal[1], env.ee_cmd[2] + dz])
            a, dyaw = self._move(goal, 0.6, yaw_tgt)
            gap = env.ee_cmd[2] - place_z
            # command release RELEASE_TIME_S early so the block lets go right at place_z
            if descend_ok and gap < abs(vz) * RELEASE_TIME_S + 0.002:
                self._enter("release")
        elif self.phase == "release":
            place_z = tgt[2] + tgt_h + held_h + PLACE_GAP
            xy_goal = tgt[:2] + v[:2] * 2.0 * env.dt
            goal = np.array([xy_goal[0], xy_goal[1], max(place_z, env.ee_cmd[2] - 0.0)])
            a, dyaw = self._move(goal, 0.6, yaw_tgt)
            vac = 0.0
            if P["held"] is None and self.phase_t > 0.1:
                self._enter("retreat")
        else:  # retreat
            goal = np.array([env.ee_cmd[0], env.ee_cmd[1], HOVER_Z])
            a, dyaw = self._move(goal, 0.25, yaw_tgt)
            vac = 0.0
        act = np.array([a[0], a[1], a[2], dyaw, vac], dtype=np.float32)
        if self.noise > 0:
            act[:4] += self.rng.normal(0, self.noise, 4)
        act[:4] = np.clip(act[:4], -1, 1)
        return act
