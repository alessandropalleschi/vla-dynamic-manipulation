"""Expert success vs control rate and latency (checks the policy interface is rate-independent).

python -m vla_mujoco.sweep_rate --episodes 40 --rates 10 20 50 --latency-scales 0 1 2
"""

from __future__ import annotations

import argparse

import numpy as np

from .expert import StackExpert
from .sim.env import DynamicStackEnv, EnvConfig


def run(hz: float, lat: float, episodes: int, seed0: int) -> tuple[float, float]:
    wins, delays = 0, []
    for i in range(episodes):
        cfg = EnvConfig(seed=seed0 + i, render=False, control_hz=hz, randomize_latency=lat > 0, dr_scale=1.0)
        env = DynamicStackEnv(cfg)
        obs, info = env.reset(seed=cfg.seed)
        if lat > 0:  # override the sampled delays: fixed command latency of lat * 80 ms
            env.act_delay_s = 0.08 * lat
            env.act_delay_steps = int(round(env.act_delay_s / env.dt))
            from collections import deque

            env._act_queue = deque([np.zeros(5)] * env.act_delay_steps)
        ex = StackExpert(env)
        for _ in range(int(cfg.max_time_s * hz)):
            obs, _, term, trunc, info = env.step(ex.act(info))
            if term or trunc:
                break
        wins += info["success"]
        delays.append(env.act_delay_s)
    return wins / episodes, float(np.mean(delays))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--rates", type=float, nargs="+", default=[10, 20, 50])
    ap.add_argument(
        "--latency-scales", type=float, nargs="+", default=[0, 1, 2], help="x 80 ms command latency"
    )
    ap.add_argument("--seed0", type=int, default=20_000)
    a = ap.parse_args()
    print("hz   cmd_latency_ms  success")
    for hz in a.rates:
        for lat in a.latency_scales:
            rate, d = run(hz, lat, a.episodes, a.seed0)
            print(f"{hz:>4.0f} {d * 1000:>14.0f}  {rate:.2f}", flush=True)


if __name__ == "__main__":
    main()
