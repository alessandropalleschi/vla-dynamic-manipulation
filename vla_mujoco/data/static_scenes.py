"""Static scenes for grounding pre-training: images + instruction + where the named objects are.

No robot rollout, so scenes cost ~30 ms each (tens of thousands in minutes). Each scene is a fresh randomised layout
(task kind, number of bystanders, appearance, cameras, ...) with its generated instruction and the image-space
location of the pick object and the support in every camera (``ref_uv`` / ``ref_vis``).

    python -m vla_mujoco.data.static_scenes --out data/scenes --scenes 30000 --workers 4 --img-size 64
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..sim.env import DynamicStackEnv, EnvConfig
from .store import write_manifest


def make_shard(args):
    """Render scenes [start, start + n) straight into the preallocated image arrays (no big pickles, no RAM spike)."""
    start, n, seed0, img_size, cameras, render_hw, out_dir, wrist_mount, overhead_view = args
    env = DynamicStackEnv(
        EnvConfig(seed=seed0, img_size=img_size, render_hw=render_hw, cameras=cameras, task="random",
                  wrist_mount=wrist_mount, overhead_view=overhead_view)
    )  # fmt: skip
    rng = np.random.default_rng(seed0 + 10_000_019)
    maps = {c: np.load(Path(out_dir) / f"img_{c}.npy", mmap_mode="r+") for c in cameras}
    uv, vis, instr, kind = [], [], [], []
    for i in range(n):
        env.cfg.n_distractors = int(rng.integers(1, 4))
        env.cfg.n_table_distractors = int(rng.integers(0, 3))
        obs, info = env.reset(seed=seed0 + i)
        for c in cameras:
            maps[c][start + i] = obs["images"][c]
        P = info["priv"]
        uv.append(P["ref_uv"])
        vis.append(P["ref_vis"])
        instr.append(obs["instruction"])
        kind.append(env.task.kind)
    env.close()
    for m in maps.values():
        m.flush()
    return start, np.stack(uv).astype(np.float32), np.stack(vis), instr, kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/scenes")
    ap.add_argument("--scenes", type=int, default=20000)
    ap.add_argument("--per-shard", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument(
        "--render-hw", type=int, nargs=2, default=None, metavar=("H", "W"), help="native render size, letterboxed"
    )
    ap.add_argument("--seed0", type=int, default=1_000_000)
    ap.add_argument("--cameras", nargs="+", default=["front", "wrist"])
    ap.add_argument("--wrist-mount", choices=["legacy", "oak"], default="legacy",
                    help="oak = OAK-D Pro on the flange adapter plate (see docs/MUJOCO.md)")  # fmt: skip
    ap.add_argument("--overhead-view", choices=["top", "front"], default="top",
                    help="front = top camera beyond the belt looking back at the robot")  # fmt: skip
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cams = tuple(a.cameras)
    (out / "config.json").write_text(
        json.dumps(
            {
                "img_size": a.img_size,
                "render_hw": a.render_hw,
                "wrist_mount": a.wrist_mount,
                "overhead_view": a.overhead_view,
            }
        )
    )
    for c in cams:  # preallocate: workers fill their slice in place
        m = np.lib.format.open_memmap(out / f"img_{c}.npy", mode="w+", dtype=np.uint8,
                                      shape=(a.scenes, a.img_size, a.img_size, 3))  # fmt: skip
        m.flush()
        del m
    jobs = [(i, min(a.per_shard, a.scenes - i), a.seed0 + i, a.img_size, cams,
             tuple(a.render_hw) if a.render_hw else None, str(out), a.wrist_mount, a.overhead_view)
            for i in range(0, a.scenes, a.per_shard)]  # fmt: skip
    uv, vis, instr, kind = [], [], [], []
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        for start, u, v, ins, kd in ex.map(make_shard, jobs):  # ordered: concatenation matches the image rows
            uv.append(u), vis.append(v), instr.extend(ins), kind.extend(kd)
            print(f"scenes {start + len(ins)}/{a.scenes}", flush=True)
    np.savez_compressed(out / "meta.npz", ref_uv=np.concatenate(uv), ref_vis=np.concatenate(vis),
                        instruction=np.array(instr), task_kind=np.array(kind), cameras=np.array(cams))  # fmt: skip
    write_manifest(out, "static_scenes", "vla_mujoco.data.static_scenes", vars(a), scenes=a.scenes)
    print(f"saved {a.scenes} scenes to {out}")


if __name__ == "__main__":
    main()
