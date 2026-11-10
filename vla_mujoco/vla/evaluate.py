"""Closed-loop evaluation of a trained policy (or the scripted expert) in MuJoCo.

python -m vla_mujoco.vla.evaluate --policy runs/gru/policy.pt --episodes 20
python -m vla_mujoco.vla.evaluate --expert --episodes 50 --speed-range 0.05 0.12 --occlusion-s 1.0
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ..expert import StackExpert
from ..sim.env import DynamicStackEnv, EnvConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy")
    ap.add_argument("--expert", action="store_true")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=10_000)
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--speed-range", type=float, nargs=2, default=(0.02, 0.07))
    ap.add_argument("--speed-change", action="store_true")
    ap.add_argument("--occlusion-s", type=float, default=0.0)
    ap.add_argument("--distractors", type=int, default=1)
    ap.add_argument("--dr-scale", type=float, default=1.0, help="domain-randomisation range multiplier")
    ap.add_argument("--no-dr", action="store_true", help="disable conveyor/camera/appearance randomisation")
    ap.add_argument("--json", help="write metrics here")
    a = ap.parse_args()
    runner = None
    cams = ("front", "wrist")
    if a.policy:
        import torch  # only needed for learned policies

        from .policy import PolicyRunner, VLAPolicy

        ck = torch.load(a.policy, map_location="cpu")
        cams = tuple(ck["args"]["cameras"])
        model = VLAPolicy(cams, arch=ck["args"]["arch"], chunk=ck["args"]["chunk"])
        model.load_state_dict(ck["model"])
        runner = PolicyRunner(model)
    elif not a.expert:
        ap.error("pass --policy or --expert")
    results = []
    for i in range(a.episodes):
        cfg = EnvConfig(
            seed=a.seed0 + i,
            img_size=a.img_size,
            cameras=cams,
            render=runner is not None,
            belt_speed_range=tuple(a.speed_range),
            speed_change=a.speed_change,
            occlusion_s=a.occlusion_s,
            n_distractors=a.distractors,
            dr_scale=a.dr_scale,
            randomize_conveyor=not a.no_dr,
            randomize_camera=not a.no_dr,
            randomize_appearance=not a.no_dr,
        )
        env = DynamicStackEnv(cfg)
        obs, info = env.reset(seed=cfg.seed)
        ex = StackExpert(env) if a.expert else None
        if runner:
            runner.reset()
        held_ever = False
        for _ in range(int(cfg.max_time_s * cfg.control_hz)):
            act = ex.act(info) if ex else runner.act(obs)
            obs, _, term, trunc, info = env.step(act)
            held_ever |= info["priv"]["held"] == "src"
            if term or trunc:
                break
        P = info["priv"]
        results.append(
            dict(
                seed=cfg.seed,
                success=bool(info["success"]),
                picked=bool(held_ever),
                time=info["t"],
                offset=P["stack_offset"],
                speed=env.belt_speed,
            )
        )
        env.close()
        print(results[-1], flush=True)
    m = dict(
        success_rate=float(np.mean([r["success"] for r in results])),
        pick_rate=float(np.mean([r["picked"] for r in results])),
        episodes=len(results),
    )
    print(m)
    if a.json:
        json.dump({"summary": m, "episodes": results}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
