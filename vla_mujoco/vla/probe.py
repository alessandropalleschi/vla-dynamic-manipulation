"""Language probe with confidence intervals: does the policy's behaviour depend on the instruction?

    python -m vla_mujoco.vla.probe --policy runs/gru_flow/best.pt --data data/mj_demos

For every held-out validation episode (the same episode-level split the trainer uses) the policy is run teacher-forced
over the whole episode with the **true instruction** and with ``--n-wrong`` instructions taken from *other* episodes
(identical images, proprioception and, for the flow head, identical sampling noise, so the comparison is paired).
Per episode, ``gap = error(wrong text) - error(true text)`` for actions and ``accuracy(true) - accuracy(wrong)`` for
grounding, so **positive always means the language is used**. Episodes are the independent units, so the report gives a
bootstrap 95% CI over episodes, the share of episodes with a positive gap and an exact sign test. A gap is only
reported as significant when the whole CI is above zero. Everything is also computed on *decision frames* (start of
approach / transfer), where the text, not the arm's motion, determines the next action.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .decision import split_indices
from .policy import CONT_DIM, VLAPolicy
from .stats import summarize
from .text import TextBackend
from .train import load_data


@torch.no_grad()
def episode_metrics(model, backend, ep, instr, dev, seed, burn_in=5, amp=True):
    """Action error and grounding accuracy (all frames / decision frames) for one episode and one instruction."""
    T = ep["T"]
    imgs = [torch.from_numpy(np.array(im)).permute(0, 3, 1, 2).float().div(255)[None].to(dev) for im in ep["img"]]
    pr = torch.from_numpy(np.asarray(ep["proprio"]))[None].to(dev)
    text = {k: (v.to(dev) if v is not None else None) for k, v in backend.encode([instr]).items()}
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and dev.startswith("cuda")):
        z, _, gl = model.features(imgs, pr, text, ground=True)
    torch.manual_seed(seed)  # same flow noise for every instruction variant of this episode
    chunk = model.sample_chunk(z[0].float())  # (T, chunk, 5)
    idx = (np.arange(T)[:, None] + np.arange(model.chunk)[None]).clip(max=T - 1)
    tgt = torch.from_numpy(np.asarray(ep["action"])[idx]).float().to(dev)
    err = (chunk[..., :CONT_DIM] - tgt[..., :CONT_DIM]).abs().mean((-1, -2)).cpu().numpy()  # (T,)
    dec = np.asarray(ep["decision"]).copy()
    valid = np.arange(T) >= burn_in
    uv = torch.from_numpy(np.asarray(ep["ref_uv"]))[None].float().to(dev)
    vis = torch.from_numpy(np.asarray(ep["ref_vis"]))[None].to(dev)
    glf = [g.float() for g in gl]

    def acc(mask):
        m = torch.from_numpy(mask).to(dev)[None, :, None, None]
        return float(model.ground_terms(glf, uv, vis & m)[1]) if bool((vis & m).any()) else float("nan")

    return {
        "err_all": float(err[valid].mean()),
        "err_dec": float(err[dec & valid].mean()) if (dec & valid).any() else float("nan"),
        "ground_all": acc(valid),
        "ground_dec": acc(dec & valid),
    }


def run_probe(model, backend, eps, dev, n_wrong=4, seed=0, burn_in=5, amp=True, progress=True):
    rng = np.random.default_rng(seed)
    strings = sorted({e["instr"] for e in eps})
    rows = []
    for i, ep in enumerate(eps):
        others = [s for s in strings if s != ep["instr"]]
        if not others:
            continue
        wrong = [others[j] for j in rng.choice(len(others), min(n_wrong, len(others)), replace=False)]
        right = episode_metrics(model, backend, ep, ep["instr"], dev, seed + i, burn_in, amp)
        wr = [episode_metrics(model, backend, ep, w, dev, seed + i, burn_in, amp) for w in wrong]
        mean_w = {k: float(np.nanmean([r[k] for r in wr])) if not all(np.isnan(r[k]) for r in wr) else float("nan")
                  for k in right}  # fmt: skip
        rows.append({"right": right, "wrong": mean_w})
        if progress and (i + 1) % 10 == 0:
            print(f"  probed {i + 1}/{len(eps)} episodes", flush=True)
    d = lambda key, sign: [sign * (r["wrong"][key] - r["right"][key]) for r in rows]  # noqa: E731
    return {
        "episodes": len(rows),
        "n_wrong_per_episode": n_wrong,
        "action_gap_all": summarize(d("err_all", +1), seed=seed),  # wrong - right error: positive = text used
        "action_gap_decision": summarize(d("err_dec", +1), seed=seed),
        "ground_gap_all": summarize(d("ground_all", -1), seed=seed),  # right - wrong accuracy: positive = text used
        "ground_gap_decision": summarize(d("ground_dec", -1), seed=seed),
        "action_error_right_decision": float(np.nanmean([r["right"]["err_dec"] for r in rows])),
        "ground_acc_right_all": float(np.nanmean([r["right"]["ground_all"] for r in rows])),
    }


def format_report(res: dict) -> str:
    head = (
        f"{'metric (positive = text used)':34s} {'mean gap':>9s} {'95% CI':>20s} "
        f"{'% eps>0':>8s} {'sign p':>8s}  verdict"
    )
    lines = [
        f"language probe: {res['episodes']} held-out episodes x {res['n_wrong_per_episode']} wrong instructions",
        head,
    ]
    for key, label in (("action_gap_all", "action error, all frames"),
                       ("action_gap_decision", "action error, decision frames"),
                       ("ground_gap_all", "grounding accuracy, all frames"),
                       ("ground_gap_decision", "grounding accuracy, decision frames")):  # fmt: skip
        s = res[key]
        verdict = "SIGNIFICANT" if s["significant"] else "not distinguishable from 0"
        lines.append(f"{label:34s} {s['mean']:+9.4f} [{s['ci_lo']:+8.4f},{s['ci_hi']:+8.4f}] "
                     f"{100 * s['frac_positive']:7.0f}% {s['sign_p']:8.4f}  {verdict}")  # fmt: skip
    lines.append(f"(true-text action error on decision frames {res['action_error_right_decision']:.4f}; "
                 f"grounding accuracy {res['ground_acc_right_all']:.3f})")  # fmt: skip
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-wrong", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=None, help="default: the fraction the policy was trained with")
    ap.add_argument("--decision-window", type=int, default=10)
    ap.add_argument("--max-eps", type=int, default=None, help="limit the number of loaded episodes (debug)")
    ap.add_argument("--split", choices=["val", "all"], default="val",
                    help="'all' includes training episodes (optimistic; for debugging only)")  # fmt: skip
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json", help="write the full result here")
    a = ap.parse_args()
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    ck = torch.load(a.policy, map_location="cpu")
    model = VLAPolicy.from_checkpoint(ck).eval().to(dev)
    backend = TextBackend(model.text_cfg)
    eps = load_data(a.data, model.cameras, a.max_eps, a.decision_window)
    val_frac = a.val_frac if a.val_frac is not None else ck.get("args", {}).get("val_frac", 0.1)
    _, va = split_indices(len(eps), val_frac)
    chosen = [eps[i] for i in (va if a.split == "val" else range(len(eps)))]
    if len(chosen) < 20:
        print(f"WARNING: only {len(chosen)} episodes; confidence intervals will be wide")
    res = run_probe(model, backend, chosen, dev, a.n_wrong, a.seed)
    res["policy"], res["data"], res["split"] = str(a.policy), str(a.data), a.split
    print(format_report(res))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
