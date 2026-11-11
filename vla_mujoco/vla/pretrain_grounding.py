"""Grounding pre-training on static scenes: which image patch does the instruction refer to?

    python -m vla_mujoco.vla.pretrain_grounding --data data/scenes --out runs/ground_pre --text-encoder pretrained

Trains the vision encoder, fusion transformer, text encoder/adapter and the pick/support grounding heads on thousands
of distinct (scene, instruction) pairs, then ``vla.train --init runs/ground_pre/best.pt`` fine-tunes on demonstrations.
Validation reports top-1 patch accuracy with the right instruction, with another scene's instruction ("wrong text"),
and for a **location-prior baseline** (always guess the most frequent patch for that camera/role). Only a gap between
right-text and wrong-text accuracy, and a margin over the prior, means the text is being grounded.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from .policy import VLAPolicy
from .text import DEFAULT_PRETRAINED, TextBackend, TextConfig
from .train import EMA, augment, lr_at


def load_scenes(path, cameras):
    imgs = {c: [] for c in cameras}
    uv, vis, instr = [], [], []
    for f in sorted(Path(path).glob("scenes_*.npz")):
        z = np.load(f)
        for c in cameras:
            imgs[c].append(z[f"img_{c}"])
        uv.append(z["ref_uv"])
        vis.append(z["ref_vis"])
        instr.extend(str(s) for s in z["instruction"])
    return [np.concatenate(imgs[c]) for c in cameras], np.concatenate(uv), np.concatenate(vis), instr


def patch_index(uv, grid):
    col = np.clip(np.floor(uv[..., 0] * grid), 0, grid - 1).astype(int)
    row = np.clip(np.floor(uv[..., 1] * grid), 0, grid - 1).astype(int)
    return row * grid + col


def prior_accuracy(train_uv, train_vis, val_uv, val_vis, grid):
    """Accuracy of always guessing the most frequent patch for each (camera, role)."""
    correct = total = 0.0
    tr_idx, va_idx = patch_index(train_uv, grid), patch_index(val_uv, grid)
    for c in range(train_uv.shape[1]):
        for r in range(2):
            m = train_vis[:, c, r]
            best = np.bincount(tr_idx[m, c, r], minlength=grid * grid).argmax()
            mv = val_vis[:, c, r]
            correct += float((va_idx[mv, c, r] == best).sum())
            total += float(mv.sum())
    return correct / max(total, 1.0)


def role_hits(gl, u, v):
    """Per-role (pick, support) correct / visible counts, summed over cameras: array (2, 2) = [role][hits, total]."""
    out = np.zeros((2, 2))
    for c, lg in enumerate(gl):
        g = int(round(lg.shape[-1] ** 0.5))
        for r in range(2):
            col = (u[:, :, c, r, 0] * g).floor().clamp(0, g - 1)
            row = (u[:, :, c, r, 1] * g).floor().clamp(0, g - 1)
            vis = v[:, :, c, r].float()
            ok = (lg[:, :, r].argmax(-1) == (row * g + col).long()).float() * vis
            out[r] += [float(ok.sum()), float(vis.sum())]
    return out


@torch.no_grad()
def evaluate(model, data, text, idx, dev, B=128):
    """Top-1 patch accuracy with the right instruction vs another scene's instruction, plus per-role accuracy."""
    model.eval()
    imgs, uv, vis = data
    hits = {"r": np.zeros((2, 2)), "w": np.zeros((2, 2))}
    for i in range(0, len(idx), B):
        j = idx[i : i + B]
        x = [torch.from_numpy(im[j]).permute(0, 3, 1, 2).float().div(255)[:, None].to(dev) for im in imgs]
        pr = torch.zeros(len(j), 1, model.cfg["proprio_dim"], device=dev)
        u = torch.from_numpy(uv[j]).float().to(dev)[:, None]
        v = torch.from_numpy(vis[j]).to(dev)[:, None]
        for tag, t in (("r", text(j)), ("w", text(np.roll(j, 1)))):
            _, _, gl = model.features(x, pr, t, ground=True)
            hits[tag] += role_hits(gl, u, v)
    model.train()
    r, w = hits["r"], hits["w"]
    return {
        "acc": r[:, 0].sum() / max(r[:, 1].sum(), 1.0),
        "acc_wrong": w[:, 0].sum() / max(w[:, 1].sum(), 1.0),
        "pick_acc": r[0, 0] / max(r[0, 1], 1.0),
        "support_acc": r[1, 0] / max(r[1, 1], 1.0),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/scenes")
    ap.add_argument("--out", default="runs/ground_pre")
    ap.add_argument("--text-encoder", choices=["bow", "scratch", "pretrained"], default="pretrained")
    ap.add_argument("--text-name", default=DEFAULT_PRETRAINED)
    ap.add_argument("--cameras", nargs="+", default=["front", "wrist"])
    ap.add_argument(
        "--vision", default="cnn", help="cnn | a Hugging Face ViT name, e.g. facebook/dinov2-small (frozen)"
    )
    ap.add_argument("--patch", type=int, default=16, choices=[8, 16, 32])
    ap.add_argument("--vision-size", type=int, default=224)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    amp = a.amp and dev.startswith("cuda")
    imgs, uv, vis, instr = load_scenes(a.data, a.cameras)
    if a.max_scenes:
        imgs, uv, vis, instr = (
            [im[: a.max_scenes] for im in imgs],
            uv[: a.max_scenes],
            vis[: a.max_scenes],
            instr[: a.max_scenes],
        )
    n = len(instr)
    perm = np.random.default_rng(0).permutation(n)
    n_val = max(64, int(n * a.val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    backend = TextBackend(TextConfig(kind=a.text_encoder, name=a.text_name))
    strings = sorted(set(instr))
    uid = {s: i for i, s in enumerate(strings)}
    print(
        f"{len(tr_idx)} train / {len(val_idx)} val scenes, {len(strings)} unique instructions, encoding text ...",
        flush=True,
    )
    enc = backend.encode(strings)
    sid = torch.tensor([uid[s] for s in instr])

    def text(j):
        t = sid[torch.as_tensor(j)]
        return {k: (v[t].to(dev) if v is not None else None) for k, v in enc.items()}

    side = imgs[0].shape[1]
    grid = side // a.patch if a.vision == "cnn" else round(a.vision_size / 14)
    env_cfg = json.loads((Path(a.data) / "config.json").read_text()) if (Path(a.data) / "config.json").exists() else {}
    prior = prior_accuracy(uv[tr_idx], vis[tr_idx], uv[val_idx], vis[val_idx], grid)
    print(f"location-prior baseline (most frequent patch per camera/role): {prior:.3f}   chance: {1 / grid**2:.3f}")
    model = VLAPolicy(
        a.cameras, arch="mlp", text=backend.cfg, patch=a.patch, vision=a.vision, vision_size=a.vision_size
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    ema = EMA(model, a.ema)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "log.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(
        ["step", "loss", "train_acc", "val_acc", "val_acc_wrong", "gap", "pick_acc", "support_acc", "prior", "sec"]
    )
    best, t0 = -1.0, time.time()
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = float(lr_at(step, a.steps, a.lr, a.warmup))
        j = rng.choice(tr_idx, a.batch)
        x = [torch.from_numpy(im[j]).permute(0, 3, 1, 2).float().div(255)[:, None].to(dev) for im in imgs]
        pr = torch.zeros(a.batch, 1, model.cfg["proprio_dim"], device=dev)
        u = torch.from_numpy(uv[j]).float().to(dev)[:, None]
        v = torch.from_numpy(vis[j]).to(dev)[:, None]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            _, _, gl = model.features(augment(x), pr, text(j), ground=True)
        loss, acc = model.ground_terms([g.float() for g in gl], u, v)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model, step)
        if step % 25 == 0 or step == 1:
            print(
                f"step {step:5d} loss {loss.item():.3f} train_acc {acc.item():.3f}  {time.time() - t0:.0f}s", flush=True
            )
        if step % a.val_every == 0 or step == a.steps:
            m = evaluate(ema.shadow, (imgs, uv, vis), text, val_idx, dev)
            gap = m["acc"] - m["acc_wrong"]
            print(
                f"  [val @ {step}] ground_acc {m['acc']:.3f}  wrong-text {m['acc_wrong']:.3f}  GAP {gap:+.3f}  "
                f"(pick {m['pick_acc']:.3f}, support {m['support_acc']:.3f})  prior {prior:.3f}",
                flush=True,
            )
            log.writerow([step, f"{loss.item():.4f}", f"{acc.item():.4f}", f"{m['acc']:.4f}", f"{m['acc_wrong']:.4f}",
                          f"{gap:.4f}", f"{m['pick_acc']:.4f}", f"{m['support_acc']:.4f}", f"{prior:.4f}",
                          f"{time.time() - t0:.0f}"])  # fmt: skip
            log_f.flush()
            ck = {
                "model": copy.deepcopy(ema.shadow.state_dict()),
                "cfg": model.cfg,
                "env_cfg": env_cfg,
                "args": vars(a),
                "step": step,
            }
            torch.save(ck, out / "policy.pt")
            if m["acc"] > best:
                best = m["acc"]
                torch.save(ck, out / "best.pt")
    print("saved", out / "best.pt")


if __name__ == "__main__":
    main()
