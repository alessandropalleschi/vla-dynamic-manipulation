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
from .store import write_episode, write_manifest

PHASES = ["approach", "descend", "lift", "transfer", "track", "release", "retreat"]


class EpisodeRecorder:
    """Accumulates one episode in the dataset layout (images + proprio + action labels + privileged labels)."""

    KEYS = ["proprio", "action", "tgt_pos", "tgt_vel_y", "src_pos", "occluded", "phase", "step", "ref_uv", "ref_vis"]

    def __init__(self, cameras):
        self.cameras = tuple(cameras)
        self.rec = {k: [] for k in self.KEYS}
        self.imgs = {c: [] for c in self.cameras}

    def add(self, obs, info, action, phase_name):
        """Record the state ``obs``/``info`` with ``action`` as its label (the expert's action in DAgger)."""
        P = info["priv"]
        for c in self.cameras:
            self.imgs[c].append(obs["images"][c])
        r = self.rec
        r["proprio"].append(obs["proprio"])
        r["action"].append(action)
        r["tgt_pos"].append(P["tgt_pos"])
        r["tgt_vel_y"].append(P["tgt_vel"][1])
        r["src_pos"].append(P["src_pos"])
        r["occluded"].append(P["occluded"])
        r["phase"].append(PHASES.index(phase_name))
        r["step"].append(P["step"])
        r["ref_uv"].append(P["ref_uv"])
        r["ref_vis"].append(P["ref_vis"])

    def finish(self, env, obs) -> dict:
        out = {k: np.asarray(v) for k, v in self.rec.items()}
        out.update({f"img_{c}": np.asarray(v, dtype=np.uint8) for c, v in self.imgs.items()})
        out["instruction"] = np.array(obs["instruction"])
        out["subgoals"] = np.array([st.text for st in env.task.steps])
        out["task_kind"] = np.array(env.task.kind)
        out["cameras"] = np.array(self.cameras)
        return out


def run_episode(seed: int, cfg_kwargs: dict, noise: float):
    cfg = EnvConfig(seed=seed, **cfg_kwargs)
    env = DynamicStackEnv(cfg)
    obs, info = env.reset(seed=seed)
    ex = StackExpert(env, seed=seed, noise=noise)
    rec = EpisodeRecorder(cfg.cameras)
    success = False
    for _ in range(int(cfg.max_time_s * cfg.control_hz)):
        a = ex.act(info)
        rec.add(obs, info, a, ex.phase)
        obs, _, term, trunc, info = env.step(a)
        success = info["success"]
        if term or trunc:
            break
    env.close()
    out = rec.finish(env, obs)
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
    ap.add_argument("--wrist-mount", choices=["legacy", "oak"], default="legacy",
                    help="oak = OAK-D Pro on the flange adapter plate (see docs/MUJOCO.md)")  # fmt: skip
    ap.add_argument("--overhead-view", choices=["top", "front"], default="top",
                    help="front = top camera beyond the belt looking back at the robot")  # fmt: skip
    ap.add_argument("--cameras", nargs="+", default=["front", "wrist"])
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
        cameras=tuple(a.cameras),
        wrist_mount=a.wrist_mount,
        overhead_view=a.overhead_view,
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
                write_episode(out, seed, ep)
                kept += 1
            print(f"seed {seed}: {'ok' if ok else 'FAILED (dropped)'} T={len(ep['action'])}", flush=True)
    args = {**cfg, "noise": a.noise, "seed0": a.seed0, "episodes_requested": a.episodes}
    write_manifest(out, "demos", "vla_mujoco.data.collect", args, episodes_saved=kept)
    print(f"saved {kept}/{a.episodes} successful episodes to {out}")


if __name__ == "__main__":
    main()
