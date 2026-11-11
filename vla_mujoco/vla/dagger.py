"""DAgger with a curriculum: the learner drives, the expert relabels every state it visits.

Behaviour cloning only ever sees states the expert visited, so a policy that drifts a few centimetres off the
demonstrated path has never seen how to recover (closed-loop success was 0/30 despite a good teacher-forced probe).
DAgger fixes the distribution shift: roll out the *current policy* (optionally mixed with the expert), record the
states it reaches, label them with the **expert's action for that state**, add them to the training set, retrain,
repeat.

Curriculum: precision is learned first on the easiest scenes and scene complexity grows only once the policy
passes a closed-loop success gate (``STAGES``). Each iteration is::

    evaluate the current policy (beta = 0, fixed seeds)  -> promote to the next stage if above the gate
    collect episodes with beta-mixed control (expert takes over a share of the steps, decayed each iteration)
    train on base demos + all DAgger data (warm start from the previous checkpoint)

Failed episodes are kept: they are exactly where the corrections are. The held-out validation split always comes
from the base demos only, so it never contains DAgger data.

    python -m vla_mujoco.vla.dagger --init runs/ft_w1/policy.pt --out runs/dagger
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..data.collect import EpisodeRecorder
from ..data.store import write_episode, write_manifest
from ..expert import StackExpert
from ..sim.env import DynamicStackEnv, EnvConfig

# Each stage overrides EnvConfig fields. `gate` = closed-loop success needed (beta = 0) to move on.
STAGES = [
    dict(
        name="s0_static_pick",  # pick a static table object, place on one slow belt object, nothing else in the scene
        env=dict(task="stack", n_distractors=0, n_table_distractors=0, belt_speed_range=(0.02, 0.03), dr_scale=0.5),
        gate=0.6,
        max_iters=5,
    ),
    dict(
        name="s1_moving_pick",  # pick a moving belt object, one bystander, wider speeds
        env=dict(
            task="belt_to_table", n_distractors=1, n_table_distractors=0, belt_speed_range=(0.02, 0.05), dr_scale=0.75
        ),
        gate=0.5,
        max_iters=5,
    ),
    dict(
        name="s2_full",  # the full training distribution: all task kinds, crowded scenes, full randomisation
        env=dict(task="random", n_distractors=2, n_table_distractors=1, belt_speed_range=(0.02, 0.07), dr_scale=1.0),
        gate=0.4,
        max_iters=6,
    ),
]
EVAL_SEED0, DAGGER_SEED0 = 50_000, 100_000
_CACHE: dict = {}


def beta_schedule(it: int, beta0: float = 0.5, decay: float = 0.5, floor_after: int = 3) -> float:
    """Share of control steps taken by the expert in iteration ``it`` of a stage (0 once the policy should lead)."""
    return 0.0 if it >= floor_after else beta0 * decay**it


def _load(ckpt: str, device: str):
    import torch

    from .policy import VLAPolicy
    from .text import TextBackend

    key = (ckpt, device)
    if key not in _CACHE:
        ck = torch.load(ckpt, map_location="cpu")
        model = VLAPolicy.from_checkpoint(ck).eval().to(device)
        _CACHE[key] = (model, TextBackend(model.text_cfg), ck.get("env_cfg") or {})
    return _CACHE[key]


def rollout(job):
    """One episode driven by a beta-mix of learner and expert. Returns (seed, stats, episode-or-None).

    A shadow expert runs on every step, so every visited state gets the expert's action as its label, whoever
    actually acted. Mixing is per ``exec_horizon``-step segment; while the expert drives, the learner's belief
    still updates (``observe``) so it never loses track of the scene."""
    seed, env_kwargs, ckpt, beta, exec_horizon, record = job
    import torch

    torch.set_num_threads(1)  # several rollout workers share the CPU; a full thread pool each would oversubscribe it

    from .policy import PolicyRunner

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner, env_cfg = None, {}
    if beta < 1.0:
        model, backend, env_cfg = _load(ckpt, device)
        cams, size, hw = tuple(model.cameras), env_cfg.get("img_size") or 128, env_cfg.get("render_hw")
        runner = PolicyRunner(model, device=device, exec_horizon=exec_horizon, text_backend=backend)
    else:  # pure expert: no learner needed, only the dataset's image setup
        cams, size, hw = ("front", "wrist"), 128, None
        if ckpt:
            ck = torch.load(ckpt, map_location="cpu")
            env_cfg = ck.get("env_cfg") or {}
            cams, size, hw = tuple(ck["cfg"]["cameras"]), env_cfg.get("img_size") or 128, None
    hardware = {k: env_cfg[k] for k in ("wrist_mount", "overhead_view") if k in env_cfg}  # what the policy trained on
    cfg = EnvConfig(
        seed=seed, img_size=size, render_hw=tuple(hw) if hw else None, cameras=cams, **{**hardware, **env_kwargs}
    )
    env = DynamicStackEnv(cfg)
    obs, info = env.reset(seed=seed)
    ex = StackExpert(env, seed=seed)
    rec = EpisodeRecorder(cams) if record else None
    rng = np.random.default_rng(seed + 7)
    use_expert, n_expert, picked, steps = False, 0, False, 0
    for t in range(int(cfg.max_time_s * cfg.control_hz)):
        a_expert = ex.act(info)  # label for the state the learner is in
        if t % max(1, exec_horizon) == 0:
            use_expert = beta >= 1.0 or bool(rng.random() < beta)
        if use_expert:
            if runner is not None:
                runner.observe(obs)
            a, n_expert = a_expert, n_expert + 1
        else:
            a = runner.act(obs)
        if rec is not None:
            rec.add(obs, info, a_expert, ex.phase)
        obs, _, term, trunc, info = env.step(a)
        steps += 1
        picked |= info["priv"]["held"] in {s[0] for s in info["priv"]["steps"]}
        if term or trunc:
            break
    stats = dict(seed=seed, success=bool(info["success"]), picked=bool(picked), steps=steps,
                 expert_frac=n_expert / max(steps, 1), kind=env.task.kind)  # fmt: skip
    ep = None
    if rec is not None:
        ep = rec.finish(env, obs)
        ep["success"], ep["beta"] = np.array(stats["success"]), np.array(beta)
    env.close()
    return seed, stats, ep


def run_jobs(jobs, workers: int):
    ctx = mp.get_context("spawn")  # CUDA / GL contexts must not be forked
    with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
        yield from ex.map(rollout, jobs)


def evaluate(ckpt: str, stage: dict, n: int, workers: int, exec_horizon: int) -> dict:
    jobs = [(EVAL_SEED0 + i, stage["env"], ckpt, 0.0, exec_horizon, False) for i in range(n)]
    res = [s for _, s, _ in run_jobs(jobs, workers)]
    return dict(
        success=float(np.mean([r["success"] for r in res])), picked=float(np.mean([r["picked"] for r in res])), n=n
    )


def collect(ckpt, stage, n, beta, seed0, out: Path, workers, exec_horizon) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(seed0 + i, stage["env"], ckpt, beta, exec_horizon, True) for i in range(n)]
    stats = []
    for seed, s, ep in run_jobs(jobs, workers):
        write_episode(out, seed, ep)
        stats.append(s)
    write_manifest(
        out, "dagger", "vla_mujoco.vla.dagger", dict(stage=stage["name"], beta=beta, init=str(ckpt)), episodes=n
    )
    return dict(
        episodes=n,
        beta=beta,
        frames=int(sum(s["steps"] for s in stats)),
        success=float(np.mean([s["success"] for s in stats])),
        expert_frac=float(np.mean([s["expert_frac"] for s in stats])),
    )


def train_command(
    ck_args: dict, base_data: str, extra: list[str], init: str, out: str, steps: int, lr: float
) -> list[str]:
    """`vla.train` command that rebuilds the checkpoint's architecture and continues from its weights."""
    a = ck_args
    cmd = [sys.executable, "-m", "vla_mujoco.vla.train", "--data", base_data, "--out", out, "--init", init,
           "--steps", str(steps), "--lr", str(lr), "--warmup", "100", "--batch", "16", "--seq", "24", "--burn-in", "6",
           "--amp", "--val-every", str(max(250, steps // 4)), "--val-batches", "8",
           "--arch", a["arch"], "--head", a["head"], "--flow-steps", str(a["flow_steps"]),
           "--text-encoder", a["text_encoder"], "--text-name", a["text_name"], "--cameras", *a["cameras"],
           "--patch", str(a["patch"]), "--vision", a["vision"], "--vision-size", str(a["vision_size"]),
           "--chunk", str(a["chunk"]), "--ground-w", str(a.get("ground_w", 1.0)),
           "--decision-w", str(a.get("decision_w", 1.0)),
           "--decision-window", str(a.get("decision_window", 10))]  # fmt: skip
    if a.get("slots"):
        cmd.append("--slots")
    if extra:
        cmd += ["--extra-train", *extra]
    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help="starting policy checkpoint (e.g. runs/ft_w1/policy.pt)")
    ap.add_argument("--base-data", default="data/mj_demos")
    ap.add_argument("--data-out", default="data/dagger")
    ap.add_argument("--out", default="runs/dagger")
    ap.add_argument("--stages", default="all", help="comma list of stage names, or 'all'")
    ap.add_argument("--episodes", type=int, default=60, help="DAgger episodes per iteration")
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--train-steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--exec-horizon", type=int, default=1)
    ap.add_argument("--max-iters", type=int, default=None, help="override every stage's iteration cap")
    ap.add_argument("--beta0", type=float, default=0.5)
    a = ap.parse_args()

    import torch

    out, data_out = Path(a.out), Path(a.data_out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = str(a.init)
    ck_args = torch.load(ckpt, map_location="cpu")["args"]
    stages = STAGES if a.stages == "all" else [s for s in STAGES if s["name"] in a.stages.split(",")]
    extra_dirs: list[str] = []
    log_path, g = out / "log.jsonl", 0

    def log(rec):
        rec["time"] = time.strftime("%H:%M:%S")
        print(json.dumps(rec), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    for stage in stages:
        cap = a.max_iters if a.max_iters is not None else stage["max_iters"]
        for it in range(cap + 1):
            ev = evaluate(ckpt, stage, a.eval_episodes, a.workers, a.exec_horizon)
            log(dict(event="eval", stage=stage["name"], iteration=it, ckpt=ckpt, gate=stage["gate"], **ev))
            if ev["success"] >= stage["gate"]:
                log(dict(event="promote", stage=stage["name"], iteration=it))
                break
            if it == cap:
                log(dict(event="stage_cap_reached", stage=stage["name"], success=ev["success"]))
                break
            beta = beta_schedule(it, a.beta0)
            d = data_out / f"{stage['name']}_it{it}"
            col = collect(ckpt, stage, a.episodes, beta, DAGGER_SEED0 + 1000 * g, d, a.workers, a.exec_horizon)
            log(dict(event="collect", stage=stage["name"], iteration=it, dir=str(d), **col))
            extra_dirs.append(str(d))
            run_dir = out / f"{stage['name']}_it{it}"
            cmd = train_command(ck_args, a.base_data, extra_dirs, ckpt, str(run_dir), a.train_steps, a.lr)
            with open(out / f"{stage['name']}_it{it}.log", "w") as lf:
                rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                log(dict(event="FAILED", reason="training", log=str(out / f"{stage['name']}_it{it}.log")))
                sys.exit(1)
            ckpt, g = str(run_dir / "policy.pt"), g + 1
            log(dict(event="trained", stage=stage["name"], iteration=it, ckpt=ckpt))
    log(dict(event="DONE", final_ckpt=ckpt))


if __name__ == "__main__":
    main()
