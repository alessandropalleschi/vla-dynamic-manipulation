"""Behaviour-cloning trainer for the VLA (CPU works; use a GPU for real runs).

    python -m vla_mujoco.vla.train --data data/mj_demos --out runs/gru_pre --arch gru \
        --text-encoder pretrained --head flow --steps 20000 --amp

Features: episode-level train/val split, EMA weights, warmup + cosine LR, bf16 autocast, gradient clipping,
recurrent burn-in, background batch prefetching, resumable checkpoints (``last.pt``) plus ``best.pt`` (by validation
action error) and ``policy.pt`` (final EMA), a CSV log, and a **language probe**: validation error with the correct
instruction vs. an instruction taken from another episode. A policy that really uses the text has a clear gap.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch

from ..data.store import list_episodes, read_episode
from .decision import decision_mask, split_indices
from .policy import CONT_DIM, N_PHASES, N_STEPS, VLAPolicy
from .text import DEFAULT_PRETRAINED, TextBackend, TextConfig


# --------------------------------------------------------------------------------------------- data
def load_data(path, cameras, max_eps=None, decision_window=10):
    eps = []
    for f in list_episodes(path)[:max_eps]:
        z = read_episode(f)  # images are memory-mapped (folder format), so the dataset need not fit in RAM
        tcp = z["proprio"][:, 12:15]
        # referring-expression supervision: where are the pick object and the support the instruction names?
        aux = np.concatenate([z["src_pos"] - tcp, z["tgt_pos"] - tcp, z["tgt_vel_y"][:, None]], 1).astype(np.float32)
        T = len(z["action"])
        eps.append(
            dict(
                img=[z[f"img_{c}"] for c in cameras],
                proprio=z["proprio"],
                action=z["action"],
                aux=aux,
                phase=z["phase"].astype(np.int64) if "phase" in z else np.zeros(T, np.int64),
                step=np.minimum(z["step"], N_STEPS - 1).astype(np.int64) if "step" in z else np.zeros(T, np.int64),
                ref_uv=z["ref_uv"].astype(np.float32)
                if "ref_uv" in z
                else np.zeros((T, len(cameras), 2, 2), np.float32),
                ref_vis=z["ref_vis"] if "ref_vis" in z else np.zeros((T, len(cameras), 2), bool),
                decision=decision_mask(z["phase"], decision_window) if "phase" in z else np.zeros(T, bool),
                instr=str(z["instruction"]),
                T=T,
            )
        )
    return eps


class Batcher:
    """Samples fixed-length windows; text is looked up from pre-encoded unique instructions."""

    def __init__(self, eps, text, chunk, seq, dev):
        self.eps, self.text, self.chunk, self.seq, self.dev = eps, text, chunk, seq, dev
        self.uid = {s: i for i, s in enumerate(text["strings"])}

    def make(self, B, rng):
        imgs = [[] for _ in self.eps[0]["img"]]
        keys = ("proprio", "aux", "phase", "step", "ref_uv", "ref_vis", "decision")
        cols = {k: [] for k in keys}
        tgt, tid = [], []
        for _ in range(B):
            e = self.eps[rng.integers(len(self.eps))]
            s = int(rng.integers(0, max(1, e["T"] - self.seq + 1)))
            idx = np.arange(s, s + self.seq).clip(max=e["T"] - 1)
            for i, im in enumerate(e["img"]):
                imgs[i].append(im[idx])
            for k in keys:
                cols[k].append(e[k][idx])
            ch = (idx[:, None] + np.arange(self.chunk)[None]).clip(max=e["T"] - 1)
            tgt.append(e["action"][ch])
            tid.append(self.uid[e["instr"]])
        t = lambda x: torch.from_numpy(np.asarray(x)).to(self.dev, non_blocking=True)  # noqa: E731
        out = {
            "imgs": [t(np.stack(x)).permute(0, 1, 4, 2, 3).float().div(255) for x in imgs],
            "proprio": t(np.stack(cols["proprio"])).float(),
            "aux": t(np.stack(cols["aux"])).float(),
            "phase": t(np.stack(cols["phase"])).clamp(max=N_PHASES - 1),
            "step": t(np.stack(cols["step"])),
            "ref_uv": t(np.stack(cols["ref_uv"])).float(),
            "ref_vis": t(np.stack(cols["ref_vis"])),
            "dec": t(np.stack(cols["decision"])),
            "target": t(np.stack(tgt)).float(),
            "tid": torch.tensor(tid),
        }
        return out

    def text_for(self, tid):
        return {k: (v[tid].to(self.dev) if v is not None else None) for k, v in self.text["tensors"].items()}


class Prefetcher:
    def __init__(self, fn, depth=3):
        self.q: queue.Queue = queue.Queue(depth)
        self.fn = fn
        self.stop = False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while not self.stop:
            self.q.put(self.fn())

    def get(self):
        return self.q.get()

    def close(self):
        self.stop = True


def augment(imgs):
    """Per-sequence, per-camera photometric + shift augmentation (on top of simulator randomisation)."""
    out = []
    for im in imgs:  # (B, T, 3, H, W) in [0, 1]
        B, dev = im.shape[0], im.device
        gain = 1 + 0.2 * (torch.rand(B, 1, 3, 1, 1, device=dev) - 0.5) * 2
        bias = 0.1 * (torch.rand(B, 1, 1, 1, 1, device=dev) - 0.5) * 2
        mean = im.mean((2, 3, 4), keepdim=True)
        contrast = 1 + 0.25 * (torch.rand(B, 1, 1, 1, 1, device=dev) - 0.5) * 2
        im = ((im - mean) * contrast + mean) * gain + bias
        im = (im + 0.02 * torch.randn_like(im[:, :1])).clamp(0, 1)
        dx, dy = np.random.randint(-4, 5, 2)
        out.append(torch.roll(im, (int(dy), int(dx)), (3, 4)))
    return out


# ---------------------------------------------------------------------------------------------- utils
class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1 + step) / (10 + step))  # warm-up the decay
        for s, p in zip(self.shadow.parameters(), model.parameters(), strict=True):
            s.mul_(d).add_(p.detach(), alpha=1 - d)
        for s, b in zip(self.shadow.buffers(), model.buffers(), strict=True):
            s.copy_(b)


def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * p)))


@torch.no_grad()
def validate(model, batcher, rng, n_batches, B, burn_in, amp):
    """Validation metrics with EMA weights, including the language probe (wrong-instruction error)."""
    model.eval()
    m = {
        k: []
        for k in ("act_l1", "act_l1_wrong", "vac_acc", "aux", "phase_acc", "step_acc", "ground_acc", "ground_acc_wrong")
    }
    dsum_r = dsum_w = dn = 0.0  # decision frames are sparse: pool sums instead of averaging per-batch means
    with torch.random.fork_rng():
        torch.manual_seed(1234)
        for _ in range(n_batches):
            b = batcher.make(B, rng)
            dev_t = {k: (v.to(batcher.dev) if v is not None else None) for k, v in batcher.text_for(b["tid"]).items()}
            wrong = batcher.text_for(torch.roll(b["tid"], 1))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                z, _, gl = model.features(b["imgs"], b["proprio"], dev_t, ground=True)
                zw, _, glw = model.features(b["imgs"], b["proprio"], wrong, ground=True)
            z, zw = z.float(), zw.float()  # heads run outside autocast: bf16 features would hit float32 weights
            sl = slice(burn_in, None)
            tgt = b["target"][:, sl]
            ch = model.sample_chunk(z[:, sl].float())
            chw = model.sample_chunk(zw[:, sl].float())
            m["act_l1"].append(float((ch[..., :CONT_DIM] - tgt[..., :CONT_DIM]).abs().mean()))
            m["act_l1_wrong"].append(float((chw[..., :CONT_DIM] - tgt[..., :CONT_DIM]).abs().mean()))
            dec = b["dec"][:, sl].float()
            dsum_r += float(((ch[..., :CONT_DIM] - tgt[..., :CONT_DIM]).abs().mean((-1, -2)) * dec).sum())
            dsum_w += float(((chw[..., :CONT_DIM] - tgt[..., :CONT_DIM]).abs().mean((-1, -2)) * dec).sum())
            dn += float(dec.sum())
            m["vac_acc"].append(float(((ch[..., CONT_DIM] > 0.5) == (tgt[..., CONT_DIM] > 0.5)).float().mean()))
            sl_uv, sl_vis = b["ref_uv"][:, sl], b["ref_vis"][:, sl]
            m["ground_acc"].append(float(model.ground_terms([g.float()[:, sl] for g in gl], sl_uv, sl_vis)[1]))
            m["ground_acc_wrong"].append(float(model.ground_terms([g.float()[:, sl] for g in glw], sl_uv, sl_vis)[1]))
            m["aux"].append(float(torch.nn.functional.smooth_l1_loss(model.aux(z[:, sl]), b["aux"][:, sl])))
            m["phase_acc"].append(float((model.aux_phase(z[:, sl]).argmax(-1) == b["phase"][:, sl]).float().mean()))
            m["step_acc"].append(float((model.aux_step(z[:, sl]).argmax(-1) == b["step"][:, sl]).float().mean()))
    out = {k: float(np.mean(v)) for k, v in m.items()}
    out["lang_gap"] = out["act_l1_wrong"] - out["act_l1"]
    out["act_l1_dec"] = dsum_r / dn if dn else float("nan")  # error on decision frames only
    out["lang_gap_dec"] = (dsum_w - dsum_r) / dn if dn else float("nan")  # ... and the wrong-text gap there
    out["ground_gap"] = out["ground_acc"] - out["ground_acc_wrong"]  # accuracy lost when the text is wrong
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/mj_demos")
    ap.add_argument("--extra-train", nargs="*", default=[], help="more demo dirs used for training only (e.g. DAgger)")
    ap.add_argument("--out", default="runs/gru")
    ap.add_argument("--arch", choices=["mlp", "stack", "gru"], default="gru")
    ap.add_argument("--head", choices=["regress", "flow"], default="regress")
    ap.add_argument("--flow-steps", type=int, default=8)
    ap.add_argument("--text-encoder", choices=["bow", "scratch", "pretrained"], default="pretrained")
    ap.add_argument("--text-name", default=DEFAULT_PRETRAINED)
    ap.add_argument("--cameras", nargs="+", default=["front", "wrist"])
    ap.add_argument(
        "--vision", default="cnn", help="cnn | a Hugging Face ViT name, e.g. facebook/dinov2-small (frozen)"
    )
    ap.add_argument(
        "--patch", type=int, default=16, choices=[8, 16, 32], help="CNN patch stride (token grid = size/patch)"
    )
    ap.add_argument("--slots", action="store_true", help="feed grounded pick/support object slots into the belief")
    ap.add_argument("--vision-size", type=int, default=224, help="input side for a pretrained ViT (resized)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=24)
    ap.add_argument("--burn-in", type=int, default=6, help="leading steps of each window excluded from the loss")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--decision-w", type=float, default=1.0, help="action-loss weight on decision frames (1 = off)")
    ap.add_argument("--decision-window", type=int, default=10, help="steps after each approach/transfer start")
    ap.add_argument("--aux-w", type=float, default=1.0)
    ap.add_argument("--ground-w", type=float, default=1.0, help="weight of the patch-grounding loss (0 = off)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--max-eps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast on CUDA")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    ap.add_argument("--init", help="initialise matching weights from a checkpoint (e.g. grounding pre-training)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    if dev.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but torch.cuda is not available (run `make gpu-check`)")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    amp = a.amp and dev.startswith("cuda")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- data + text
    eps = load_data(a.data, a.cameras, a.max_eps, a.decision_window)
    env_cfg = json.loads((Path(a.data) / "config.json").read_text()) if (Path(a.data) / "config.json").exists() else {}
    tr_i, va_i = split_indices(len(eps), a.val_frac)
    train_eps, val_eps = [eps[i] for i in tr_i], [eps[i] for i in va_i]
    extra = [e for d in a.extra_train for e in load_data(d, a.cameras, None, a.decision_window)]
    train_eps += extra  # train-only: the held-out validation split always comes from --data alone
    backend = TextBackend(TextConfig(kind=a.text_encoder, name=a.text_name))
    strings = sorted({e["instr"] for e in [*eps, *extra]})
    enc = backend.encode(strings)
    text = {"strings": strings, "tensors": {k: v for k, v in enc.items()}}
    name = torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu"
    print(
        f"{len(train_eps)} train ({len(extra)} extra) / {len(val_eps)} val episodes, "
        f"{sum(e['T'] for e in train_eps)} frames, "
        f"{len(strings)} unique instructions, text={a.text_encoder}, head={a.head}, arch={a.arch}, "
        f"device {dev} ({name}), amp={amp}"
    )

    # ---- model
    model = VLAPolicy(
        a.cameras,
        arch=a.arch,
        chunk=a.chunk,
        head=a.head,
        flow_steps=a.flow_steps,
        text=backend.cfg,
        patch=a.patch,
        vision=a.vision,
        vision_size=a.vision_size,
        slots=a.slots,
    ).to(dev)
    if a.init:
        sd, own = torch.load(a.init, map_location=dev)["model"], model.state_dict()
        keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape and not k.startswith("p_")}
        model.load_state_dict(keep, strict=False)
        print(f"initialised {len(keep)}/{len(own)} tensors from {a.init}")
    allp = np.concatenate([e["proprio"] for e in eps])
    model.p_mean.copy_(torch.from_numpy(allp.mean(0)))
    model.p_std.copy_(torch.from_numpy(allp.std(0) + 1e-3))
    aux_all = np.concatenate([e["aux"] for e in eps])
    aux_mu, aux_sd = aux_all.mean(0), aux_all.std(0) + 1e-4
    for e in [*eps, *extra]:  # statistics come from --data only, so they stay fixed as extra data grows
        e["aux"] = ((e["aux"] - aux_mu) / aux_sd).astype(np.float32)
    print(f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    ema = EMA(model, a.ema)
    start = 0
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev)
        model.load_state_dict(ck["raw"])
        ema.shadow.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
        print(f"resumed from step {start}")

    tb = Batcher(train_eps, text, a.chunk, a.seq, dev)
    vb = Batcher(val_eps, text, a.chunk, a.seq, dev)
    pre = Prefetcher(lambda: tb.make(a.batch, rng))
    vrng = np.random.default_rng(999)
    log_f = open(out / "log.csv", "a", newline="")
    log = csv.writer(log_f)
    if start == 0:
        log.writerow(
            ["step", "lr", "loss", "act", "vac", "aux", "phase", "step_cls", "val_act_l1", "val_act_l1_wrong",
             "lang_gap", "val_vac_acc", "val_phase_acc", "val_step_acc", "ground_acc", "ground_acc_wrong",
             "act_l1_dec", "lang_gap_dec", "sec"]
        )  # fmt: skip
    best = float("inf")
    t0 = time.time()
    model.train()
    for step in range(start + 1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = float(lr_at(step, a.steps, a.lr, a.warmup))
        b = pre.get()
        txt = tb.text_for(b["tid"])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            z, _, gl = model.features(augment(b["imgs"]), b["proprio"], txt, ground=True)
        z = z.float()
        sl = slice(a.burn_in, None)
        w = 1.0 + (a.decision_w - 1.0) * b["dec"][:, sl].float()
        lt = model.loss_terms(z[:, sl], b["target"][:, sl], w if a.decision_w != 1.0 else None)
        at = model.aux_terms(z[:, sl], b["aux"][:, sl], b["phase"][:, sl], b["step"][:, sl])
        gl = [g.float()[:, sl] for g in gl]
        g_loss, g_acc = model.ground_terms(gl, b["ref_uv"][:, sl], b["ref_vis"][:, sl])
        loss = lt["act"] + 0.3 * lt["vac"] + a.aux_w * at["aux"] + 0.2 * at["phase"] + 0.2 * at["step"]
        loss = loss + a.ground_w * g_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model, step)
        row = None
        if step % a.val_every == 0 or step == a.steps:
            v = validate(ema.shadow, vb, vrng, a.val_batches, a.batch, a.burn_in, amp)
            row = v
            if v["act_l1"] < best:
                best = v["act_l1"]
                torch.save(
                    {
                        "model": ema.shadow.state_dict(),
                        "cfg": model.cfg,
                        "env_cfg": env_cfg,
                        "args": vars(a),
                        "step": step,
                    },
                    out / "best.pt",
                )
            torch.save(
                {"model": ema.shadow.state_dict(), "raw": model.state_dict(), "opt": opt.state_dict(), "cfg": model.cfg,
                 "args": vars(a), "step": step},
                out / "last.pt",
            )  # fmt: skip
            print(
                f"  [val @ {step}] act_l1 {v['act_l1']:.4f}  wrong-text {v['act_l1_wrong']:.4f}  "
                f"lang_gap {v['lang_gap']:+.4f} (decision frames {v['lang_gap_dec']:+.4f})  "
                f"ground_acc {v['ground_acc']:.3f} "
                f"(wrong-text {v['ground_acc_wrong']:.3f})  vac_acc {v['vac_acc']:.3f}  phase_acc {v['phase_acc']:.3f}",
                flush=True,
            )
        if step % 25 == 0 or step == 1 or row is not None:
            print(
                f"step {step:6d} loss {loss.item():.4f} act {lt['act'].item():.4f} vac {lt['vac'].item():.4f} "
                f"aux {at['aux'].item():.4f} phase {at['phase'].item():.3f} "
                f"ground {g_loss.item():.3f} (acc {g_acc.item():.2f})  {time.time() - t0:.0f}s",
                flush=True,
            )
            val_cols = (
                "act_l1",
                "act_l1_wrong",
                "lang_gap",
                "vac_acc",
                "phase_acc",
                "step_acc",
                "ground_acc",
                "ground_acc_wrong",
            )
            train_cols = [
                loss.item(),
                lt["act"].item(),
                lt["vac"].item(),
                at["aux"].item(),
                at["phase"].item(),
                at["step"].item(),
            ]
            log.writerow(
                [step, f"{opt.param_groups[0]['lr']:.2e}"]
                + [f"{x:.4f}" for x in train_cols]
                + ([f"{row[k]:.4f}" for k in val_cols] if row else [""] * len(val_cols))
                + [f"{time.time() - t0:.0f}"]
            )
            log_f.flush()
    pre.close()
    torch.save(
        {"model": ema.shadow.state_dict(), "cfg": model.cfg, "env_cfg": env_cfg, "args": vars(a), "step": a.steps},
        out / "policy.pt",
    )
    (out / "train_args.json").write_text(json.dumps(vars(a), indent=1))
    print("saved", out / "policy.pt", "(final EMA) and", out / "best.pt")


if __name__ == "__main__":
    main()
