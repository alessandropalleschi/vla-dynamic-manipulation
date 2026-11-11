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
    ap.add_argument("--img-size", type=int, default=None, help="default: the size the policy was trained on")
    ap.add_argument("--render-hw", type=int, nargs=2, default=None, metavar=("H", "W"))
    ap.add_argument("--speed-range", type=float, nargs=2, default=(0.02, 0.07))
    ap.add_argument("--speed-change", action="store_true")
    ap.add_argument("--occlusion-s", type=float, default=0.0)
    ap.add_argument("--distractors", type=int, default=1, help="bystander objects on the belt")
    ap.add_argument("--table-distractors", type=int, default=0, help="bystander objects on the table")
    ap.add_argument("--task", default="stack", help="stack | belt_to_table | belt_to_belt | tower | random")
    ap.add_argument("--dr-scale", type=float, default=1.0, help="domain-randomisation range multiplier")
    ap.add_argument("--no-dr", action="store_true", help="disable conveyor/camera/appearance randomisation")
    ap.add_argument("--exec-horizon", type=int, default=4, help="actions executed per policy re-plan")
    ap.add_argument("--device", default="auto", help="policy device: auto | cpu | cuda")
    ap.add_argument("--json", help="write metrics here")
    a = ap.parse_args()
    runner = None
    cams = ("front", "wrist")
    if a.policy:
        import torch  # only needed for learned policies

        from .policy import PolicyRunner, VLAPolicy

        ck = torch.load(a.policy, map_location="cpu")
        model = VLAPolicy.from_checkpoint(ck)
        cams = tuple(model.cameras)
        env_cfg = ck.get("env_cfg") or {}
        a.img_size = a.img_size or env_cfg.get("img_size")
        a.render_hw = a.render_hw or env_cfg.get("render_hw")
        dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
        runner = PolicyRunner(model, device=dev, exec_horizon=a.exec_horizon)  # rebuilds the text backend
    elif not a.expert:
        ap.error("pass --policy or --expert")
    results = []
    for i in range(a.episodes):
        cfg = EnvConfig(
            seed=a.seed0 + i,
            img_size=a.img_size or 96,
            render_hw=tuple(a.render_hw) if a.render_hw else None,
            cameras=cams,
            render=runner is not None,
            belt_speed_range=tuple(a.speed_range),
            speed_change=a.speed_change,
            occlusion_s=a.occlusion_s,
            n_distractors=a.distractors,
            n_table_distractors=a.table_distractors,
            task=a.task,
            dr_scale=a.dr_scale,
            **EnvConfig.dr_flags(not a.no_dr),
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
            held_ever |= info["priv"]["held"] in {st[0] for st in info["priv"]["steps"]}
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
                kind=env.task.kind,
                instruction=env.instruction,
                steps_done=int(sum(P["step_ok"])),
            )
        )
        env.close()
        print(results[-1], flush=True)
    m = dict(
        success_rate=float(np.mean([r["success"] for r in results])),
        pick_rate=float(np.mean([r["picked"] for r in results])),
        episodes=len(results),
        by_task={
            k: float(np.mean([r["success"] for r in results if r["kind"] == k]))
            for k in sorted({r["kind"] for r in results})
        },
    )
    print(m)
    if a.json:
        json.dump({"summary": m, "episodes": results}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
