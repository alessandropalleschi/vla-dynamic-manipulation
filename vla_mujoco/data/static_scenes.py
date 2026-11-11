"""Static scenes for grounding pre-training: images + instruction + where the named objects are.

No robot rollout, so scenes cost ~30 ms each (tens of thousands in minutes). Each scene is a fresh randomised layout
(task kind, number of bystanders, appearance, cameras, ...) with its generated instruction and the image-space
location of the pick object and the support in every camera (``ref_uv`` / ``ref_vis``).

    python -m vla_mujoco.data.static_scenes --out data/scenes --scenes 30000 --workers 4 --img-size 64
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..sim.env import DynamicStackEnv, EnvConfig


def make_shard(args):
    shard, n, seed0, img_size, cameras = args
    env = DynamicStackEnv(EnvConfig(seed=seed0, img_size=img_size, cameras=cameras, task="random"))
    rng = np.random.default_rng(seed0 + 10_000_019)
    imgs = {c: [] for c in cameras}
    uv, vis, instr, kind = [], [], [], []
    for i in range(n):
        env.cfg.n_distractors = int(rng.integers(1, 4))
        env.cfg.n_table_distractors = int(rng.integers(0, 3))
        obs, info = env.reset(seed=seed0 + i)
        for c in cameras:
            imgs[c].append(obs["images"][c])
        P = info["priv"]
        uv.append(P["ref_uv"])
        vis.append(P["ref_vis"])
        instr.append(obs["instruction"])
        kind.append(env.task.kind)
    env.close()
    out = {f"img_{c}": np.stack(v).astype(np.uint8) for c, v in imgs.items()}
    out.update(ref_uv=np.stack(uv).astype(np.float32), ref_vis=np.stack(vis), instruction=np.array(instr),
               task_kind=np.array(kind), cameras=np.array(cameras))  # fmt: skip
    return shard, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/scenes")
    ap.add_argument("--scenes", type=int, default=20000)
    ap.add_argument("--per-shard", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--seed0", type=int, default=1_000_000)
    ap.add_argument("--cameras", nargs="+", default=["front", "wrist"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n_shards = -(-a.scenes // a.per_shard)
    jobs = [(i, min(a.per_shard, a.scenes - i * a.per_shard), a.seed0 + i * a.per_shard, a.img_size, tuple(a.cameras))
            for i in range(n_shards)]  # fmt: skip
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        for shard, data in ex.map(make_shard, jobs):
            np.savez_compressed(out / f"scenes_{shard:04d}.npz", **data)
            print(f"shard {shard + 1}/{n_shards} ({len(data['instruction'])} scenes)", flush=True)
    print(f"saved {a.scenes} scenes to {out}")


if __name__ == "__main__":
    main()
