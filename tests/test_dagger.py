"""DAgger: beta schedule, mixed rollouts with expert relabelling, train-only extra data, the loop at toy scale."""

import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vla_mujoco.data.store import read_episode, write_episode  # noqa: E402
from vla_mujoco.vla import dagger  # noqa: E402
from vla_mujoco.vla.policy import PolicyRunner, VLAPolicy  # noqa: E402
from vla_mujoco.vla.text import TextBackend, TextConfig  # noqa: E402

TINY_STAGE = dict(name="tiny", env=dict(task="stack", n_distractors=0, n_table_distractors=0,
                                       belt_speed_range=(0.02, 0.03), dr_scale=0.5), gate=2.0, max_iters=1)  # fmt: skip


def tiny_checkpoint(path, size=32):
    be = TextBackend(TextConfig(kind="bow"))
    m = VLAPolicy(arch="gru", text=be.cfg, patch=8, slots=True, chunk=4)
    args = dict(arch="gru", head="regress", flow_steps=8, text_encoder="bow", text_name="x", cameras=["front", "wrist"],
                patch=8, vision="cnn", vision_size=224, slots=True, chunk=4, ground_w=1.0, decision_w=1.0)  # fmt: skip
    torch.save({"model": m.state_dict(), "cfg": m.cfg, "args": args, "env_cfg": {"img_size": size}}, path)


def test_beta_schedule_decays_to_zero_so_the_learner_eventually_leads():
    b = [dagger.beta_schedule(i) for i in range(5)]
    assert b[0] == 0.5 and b[1] == 0.25 and b[2] == 0.125 and b[3] == 0.0 and b[4] == 0.0
    assert all(x >= y for x, y in zip(b, b[1:], strict=False))


def test_curriculum_stages_get_harder_and_are_valid_env_configs():
    from vla_mujoco.sim.env import DynamicStackEnv, EnvConfig

    names = [s["name"] for s in dagger.STAGES]
    assert len(set(names)) == len(names) and all(0 < s["gate"] < 1 for s in dagger.STAGES)
    objs = [s["env"]["n_distractors"] + s["env"]["n_table_distractors"] for s in dagger.STAGES]
    assert objs == sorted(objs) and dagger.STAGES[-1]["env"]["task"] == "random"
    assert dagger.STAGES[0]["env"]["dr_scale"] < dagger.STAGES[-1]["env"]["dr_scale"]
    for s in dagger.STAGES:  # every stage must build and reset
        env = DynamicStackEnv(EnvConfig(seed=0, render=False, **s["env"]))
        env.reset(seed=0)


def test_observe_updates_the_belief_without_acting_and_drops_the_queue():
    from vla_mujoco.sim.env import DynamicStackEnv, EnvConfig

    env = DynamicStackEnv(EnvConfig(seed=0, img_size=32, **TINY_STAGE["env"]))
    obs, _ = env.reset(seed=0)
    be = TextBackend(TextConfig(kind="bow"))
    runner = PolicyRunner(VLAPolicy(arch="gru", text=be.cfg, patch=8, chunk=4), exec_horizon=4, text_backend=be)
    runner.act(obs)
    assert len(runner.queue) == 3 and runner.replans == 1
    h0 = runner.h.clone()
    runner.observe(obs)
    assert runner.queue == [] and not torch.equal(runner.h, h0)  # belief advanced, stale chunk dropped
    runner.act(obs)
    assert runner.replans == 2  # re-planned from the new state


def test_rollout_labels_are_expert_actions_whoever_drives(tmp_path):
    ck = tmp_path / "p.pt"
    tiny_checkpoint(ck)
    seed, s1, ep1 = dagger.rollout((3, TINY_STAGE["env"], str(ck), 1.0, 1, True))  # expert drives every step
    assert s1["expert_frac"] == 1.0 and s1["success"] and len(ep1["action"]) == s1["steps"]
    _, s0, ep0 = dagger.rollout((3, TINY_STAGE["env"], str(ck), 0.0, 1, True))  # an untrained learner drives
    assert s0["expert_frac"] == 0.0 and not s0["success"]
    assert np.abs(ep0["action"][:, :3]).max() > 0.1  # labels are the expert's, not the learner's ~0 output
    # the learner is stuck far from where the expert wants it, so labels keep pointing somewhere: not all-zero
    assert ep0["action"].shape[1] == 5 and len(ep0["phase"]) == len(ep0["action"]) and bool(ep0["beta"] == 0.0)
    _, sm, _ = dagger.rollout((3, TINY_STAGE["env"], str(ck), 0.5, 2, False))
    assert 0.0 < sm["expert_frac"] < 1.0  # a genuine mix, segments of exec_horizon steps


def test_failed_rollouts_are_kept_and_extra_train_data_never_enters_validation(tmp_path, monkeypatch):
    from vla_mujoco.data import collect
    from vla_mujoco.vla import train

    ck = tmp_path / "p.pt"
    tiny_checkpoint(ck)
    base = tmp_path / "base"
    monkeypatch.setattr(
        sys, "argv", ["c", *f"--out {base} --episodes 6 --workers 1 --img-size 32 --task stack".split()]
    )
    collect.main()
    _, stats, ep = dagger.rollout((5, TINY_STAGE["env"], str(ck), 0.0, 1, True))
    assert not stats["success"]  # a failed episode ...
    extra = tmp_path / "extra"
    write_episode(extra, 5, ep)  # ... is still written, with its labels
    assert bool(read_episode(extra / "ep_000005")["success"]) is False

    cmd = dagger.train_command(torch.load(ck)["args"], str(base), [str(extra)], str(ck), str(tmp_path / "o"), 3, 1e-3)
    assert "--extra-train" in cmd and str(extra) in cmd and "--slots" in cmd and cmd[cmd.index("--arch") + 1] == "gru"
    n_val = len(train.split_indices(6, 0.1)[1])
    eps = train.load_data(base, ["front", "wrist"])
    tr, va = train.split_indices(len(eps), 0.1)
    extra_eps = train.load_data(extra, ["front", "wrist"])
    assert len(va) == n_val and len(extra_eps) == 1  # validation = base demos only, whatever is added for training


def test_full_dagger_loop_at_toy_scale(tmp_path, monkeypatch):
    """eval -> collect (mixed control, failures kept) -> train on base + DAgger data -> eval, end to end."""
    from vla_mujoco.data import collect

    base, out, data_out = tmp_path / "base", tmp_path / "run", tmp_path / "dag"
    monkeypatch.setattr(
        sys, "argv", ["c", *f"--out {base} --episodes 6 --workers 1 --img-size 32 --task stack".split()]
    )
    collect.main()
    ck = tmp_path / "init.pt"
    tiny_checkpoint(ck)
    monkeypatch.setattr(dagger, "STAGES", [TINY_STAGE])
    argv = (
        f"x --init {ck} --base-data {base} --data-out {data_out} --out {out} --episodes 2 --eval-episodes 1 --workers 1"
    )
    monkeypatch.setattr(sys, "argv", [*argv.split(), "--train-steps", "3", "--max-iters", "1", "--exec-horizon", "1"])
    dagger.main()
    import json

    events = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    kinds = [e["event"] for e in events]
    assert kinds == ["eval", "collect", "trained", "eval", "stage_cap_reached", "DONE"], kinds
    assert events[1]["episodes"] == 2 and events[1]["beta"] == 0.5 and events[1]["frames"] > 0
    assert (out / "tiny_it0" / "policy.pt").exists() and any(data_out.glob("tiny_it0/ep_*"))
    assert events[2]["ckpt"].endswith("tiny_it0/policy.pt") and events[-1]["final_ckpt"] == events[2]["ckpt"]
