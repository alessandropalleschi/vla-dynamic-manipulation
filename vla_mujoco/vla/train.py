"""Behaviour-cloning trainer for the compact VLA (CPU works; use a GPU for real runs).

python -m vla_mujoco.vla.train --data data/mj_demos --arch gru --steps 3000 --out runs/gru
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .policy import VAC_IDX, VLAPolicy, tokenize


def load_data(path, cameras, max_eps=None):
    eps = []
    for f in sorted(Path(path).glob("ep_*.npz"))[:max_eps]:
        z = np.load(f)
        T = len(z["action"])
        pos = z["tgt_pos"]
        tcp = z["proprio"][:, 12:15]
        aux = np.concatenate([pos - tcp, z["tgt_vel_y"][:, None]], 1).astype(np.float32)
        eps.append(
            dict(
                img=[z[f"img_{c}"] for c in cameras],
                proprio=z["proprio"],
                action=z["action"],
                aux=aux,
                tok=tokenize(str(z["instruction"])),
                T=T,
            )
        )
    return eps


def make_batch(eps, B, L, chunk, dev, rng):
    imgs = [[] for _ in eps[0]["img"]]
    pr, tgt, aux, tok = [], [], [], []
    for _ in range(B):
        e = eps[rng.integers(len(eps))]
        s = int(rng.integers(0, max(1, e["T"] - L + 1)))
        idx = np.arange(s, s + L).clip(max=e["T"] - 1)
        for i, im in enumerate(e["img"]):
            imgs[i].append(im[idx])
        pr.append(e["proprio"][idx])
        aux.append(e["aux"][idx])
        ch = (idx[:, None] + np.arange(chunk)[None]).clip(max=e["T"] - 1)
        tgt.append(e["action"][ch])
        tok.append(e["tok"])
    t = lambda x: torch.from_numpy(np.asarray(x)).to(dev)  # noqa: E731
    imgs = [t(np.stack(x)).permute(0, 1, 4, 2, 3).float().div(255) for x in imgs]
    return (
        imgs,
        t(np.stack(pr)).float(),
        t(np.stack(aux)).float(),
        torch.stack(tok).to(dev),
        t(np.stack(tgt)).float(),
    )


def augment(imgs):
    """Cheap photometric + shift augmentation, per sequence."""
    out = []
    for im in imgs:
        B = im.shape[0]
        g = 1 + 0.15 * (torch.rand(B, 1, 1, 1, 1, device=im.device) - 0.5) * 2
        b = 0.08 * (torch.rand(B, 1, 1, 1, 1, device=im.device) - 0.5) * 2
        im = (im * g + b).clamp(0, 1)
        dx, dy = np.random.randint(-3, 4, 2)
        out.append(torch.roll(im, (int(dy), int(dx)), (3, 4)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mj_demos")
    ap.add_argument("--out", default="runs/gru")
    ap.add_argument("--arch", choices=["mlp", "stack", "gru"], default="gru")
    ap.add_argument("--cameras", nargs="+", default=["front", "wrist"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--aux-w", type=float, default=0.5)
    ap.add_argument("--max-eps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    eps = load_data(a.data, a.cameras, a.max_eps)
    print(f"{len(eps)} episodes, {sum(e['T'] for e in eps)} frames, device {dev}")
    model = VLAPolicy(a.cameras, arch=a.arch, chunk=a.chunk).to(dev)
    allp = np.concatenate([e["proprio"] for e in eps])
    model.p_mean.copy_(torch.from_numpy(allp.mean(0)))
    model.p_std.copy_(torch.from_numpy(allp.std(0) + 1e-3))
    aux_all = np.concatenate([e["aux"] for e in eps])
    aux_mu, aux_sd = (
        torch.from_numpy(aux_all.mean(0)).to(dev),
        torch.from_numpy(aux_all.std(0) + 1e-4).to(dev),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        imgs, pr, aux, tok, tgt = make_batch(eps, a.batch, a.seq, a.chunk, dev, rng)
        pred, aux_p, _ = model(augment(imgs), pr + 0.0 * pr, tok)
        l_act = F.l1_loss(pred[..., :4], tgt[..., :4])
        l_vac = F.binary_cross_entropy_with_logits(pred[..., VAC_IDX], tgt[..., VAC_IDX])
        l_aux = F.smooth_l1_loss(aux_p, (aux - aux_mu) / aux_sd)
        loss = l_act + 0.3 * l_vac + a.aux_w * l_aux
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == 1:
            print(
                f"step {step:5d} loss {loss.item():.4f} act {l_act.item():.4f} vac {l_vac.item():.4f} "
                f"aux {l_aux.item():.4f}  {time.time() - t0:.0f}s",
                flush=True,
            )
    torch.save({"model": model.state_dict(), "args": vars(a)}, out / "policy.pt")
    (out / "train_args.json").write_text(json.dumps(vars(a), indent=1))
    print("saved", out / "policy.pt")


if __name__ == "__main__":
    main()
