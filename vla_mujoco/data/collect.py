"""Collect expert demonstrations (images + proprio + actions + privileged labels).

    python -m vla_mujoco.data.collect --out data/mj_demos --episodes 200 --workers 4

One compressed ``.npz`` per successful episode. Keys: ``img_<cam>`` (T,H,W,3 uint8),
``proprio`` (T,22), ``action`` (T,5), ``instruction``, and privileged labels used for
auxiliary losses / analysis only: ``tgt_pos`` (T,3), ``tgt_vel_y`` (T,), ``src_pos``,
``occluded`` (T,), ``phase`` (T,).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..expert import StackExpert
from ..sim.env import DynamicStackEnv, EnvConfig

PHASES = ["approach", "descend", "lift", "transfer", "track", "release", "retreat"]


def run_episode(seed: int, cfg_kwargs: dict, noise: float):
    cfg = EnvConfig(seed=seed, **cfg_kwargs)
    env = DynamicStackEnv(cfg)
    obs, info = env.reset(seed=seed)
    ex = StackExpert(env, seed=seed, noise=noise)
    rec = {
        k: []
        for k in [
            "proprio",
            "action",
            "tgt_pos",
            "tgt_vel_y",
            "src_pos",
            "occluded",
            "phase",
            "step",
            "ref_uv",
            "ref_vis",
        ]
    }
    imgs = {c: [] for c in cfg.cameras}
    success = False
    for _ in range(int(cfg.max_time_s * cfg.control_hz)):
        a = ex.act(info)
        P = info["priv"]
        for c in cfg.cameras:
            imgs[c].append(obs["images"][c])
        rec["proprio"].append(obs["proprio"])
        rec["action"].append(a)
        rec["tgt_pos"].append(P["tgt_pos"])
        rec["tgt_vel_y"].append(P["tgt_vel"][1])
        rec["src_pos"].append(P["src_pos"])
        rec["occluded"].append(P["occluded"])
        rec["phase"].append(PHASES.index(ex.phase))
        rec["step"].append(P["step"])
        rec["ref_uv"].append(P["ref_uv"])
        rec["ref_vis"].append(P["ref_vis"])
        obs, _, term, trunc, info = env.step(a)
        success = info["success"]
        if term or trunc:
            break
    env.close()
    out = {k: np.asarray(v) for k, v in rec.items()}
    out.update({f"img_{c}": np.asarray(v, dtype=np.uint8) for c, v in imgs.items()})
    out["instruction"] = np.array(obs["instruction"])
    out["subgoals"] = np.array([st.text for st in env.task.steps])
    out["task_kind"] = np.array(env.task.kind)
    out["cameras"] = np.array(cfg.cameras)
    return seed, success, out


def _job(args):
    return run_episode(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mj_demos")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--img-size", type=int, default=96, help="side of the square image stored / seen by the policy")
    ap.add_argument("--render-hw", type=int, nargs=2, default=None, metavar=("H", "W"),
                    help="native render size, letterboxed to --img-size (e.g. 480 640 or 1080 1920)")  # fmt: skip
    ap.add_argument("--noise", type=float, default=0.0, help="expert action noise (DART-style)")
    ap.add_argument("--speed-range", type=float, nargs=2, default=(0.02, 0.07))
    ap.add_argument("--speed-change", action="store_true")
    ap.add_argument("--occlusion-s", type=float, default=0.0)
    ap.add_argument("--distractors", type=int, default=1, help="bystander objects on the belt")
    ap.add_argument("--table-distractors", type=int, default=0, help="bystander objects on the table")
    ap.add_argument("--task", default="stack", help="stack | belt_to_table | belt_to_belt | tower | random")
    ap.add_argument("--dr-scale", type=float, default=1.0, help="domain-randomisation range multiplier")
    ap.add_argument("--no-dr", action="store_true", help="disable conveyor/camera/appearance randomisation")
    a = ap.parse_args()
    cfg = dict(
        img_size=a.img_size,
        render_hw=tuple(a.render_hw) if a.render_hw else None,
        belt_speed_range=tuple(a.speed_range),
        speed_change=a.speed_change,
        occlusion_s=a.occlusion_s,
        n_distractors=a.distractors,
        n_table_distractors=a.table_distractors,
        task=a.task,
        dr_scale=a.dr_scale,
        **EnvConfig.dr_flags(not a.no_dr),
    )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**cfg, "noise": a.noise}, indent=1))
    jobs = [(a.seed0 + i, cfg, a.noise) for i in range(a.episodes)]
    kept = 0
    # spawn (not fork): forking after torch/OpenMP or GL contexts exist can deadlock
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        for seed, ok, ep in ex.map(_job, jobs):
            if ok:
                np.savez_compressed(out / f"ep_{seed:06d}.npz", **ep)
                kept += 1
            print(f"seed {seed}: {'ok' if ok else 'FAILED (dropped)'} T={len(ep['action'])}", flush=True)
    print(f"saved {kept}/{a.episodes} successful episodes to {out}")


if __name__ == "__main__":
    main()
