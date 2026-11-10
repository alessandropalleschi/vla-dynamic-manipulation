"""Render an expert episode to a GIF (3 cameras side by side).

python -m vla_mujoco.viz --seed 3 --out demo.gif
"""

import argparse

import numpy as np
from PIL import Image

from .expert import StackExpert
from .sim.env import DynamicStackEnv, EnvConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="demo.gif")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--occlusion-s", type=float, default=0.0)
    a = ap.parse_args()
    cams = ("front", "overhead", "wrist")
    env = DynamicStackEnv(EnvConfig(seed=a.seed, img_size=a.size, cameras=cams, occlusion_s=a.occlusion_s))
    obs, info = env.reset(seed=a.seed)
    ex = StackExpert(env)
    frames = []
    for k in range(int(env.cfg.max_time_s * env.cfg.control_hz)):
        obs, _, term, trunc, info = env.step(ex.act(info))
        if k % a.every == 0:
            frames.append(Image.fromarray(np.hstack([obs["images"][c] for c in cams])))
        if term or trunc:
            break
    print(obs["instruction"], "| success:", info["success"], f"| t={info['t']:.1f}s")
    frames[0].save(
        a.out, save_all=True, append_images=frames[1:], duration=int(1000 * env.dt * a.every), loop=0
    )
    frames[-1].save(a.out.replace(".gif", "_last.png"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
