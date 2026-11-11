"""Policy model, text backends, flow head, chunked execution and a tiny training run (needs torch)."""

import sys

import numpy as np
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from vla_mujoco.sim.env import DynamicStackEnv, EnvConfig  # noqa: E402
from vla_mujoco.vla.policy import CONT_DIM, PolicyRunner, VLAPolicy  # noqa: E402
from vla_mujoco.vla.text import TextBackend, TextConfig  # noqa: E402

TEXTS = ["put the red cube onto the blue block", "stack the second cylinder from the conveyor on the larger disc"]


def batch(model, backend, B=2, T=5, size=48):
    imgs = [torch.rand(B, T, 3, size, size) for _ in model.cameras]
    return imgs, torch.randn(B, T, 22), backend.encode(TEXTS[:B])


@pytest.mark.parametrize("kind", ["bow", "scratch"])
@pytest.mark.parametrize("head", ["regress", "flow"])
@pytest.mark.parametrize("arch", ["mlp", "stack", "gru"])
def test_forward_loss_and_sampling(kind, head, arch):
    backend = TextBackend(TextConfig(kind=kind))
    m = VLAPolicy(arch=arch, head=head, text=backend.cfg, chunk=6)
    imgs, pr, txt = batch(m, backend)
    z, _ = m.features(imgs, pr, txt)
    tgt = torch.rand(2, 5, 6, 5)
    tgt[..., :CONT_DIM] = tgt[..., :CONT_DIM] * 2 - 1
    lt = m.loss_terms(z, tgt)
    at = m.aux_terms(z, torch.randn(2, 5, 7), torch.randint(0, 7, (2, 5)), torch.randint(0, 4, (2, 5)))
    sum([*lt.values(), *at.values()]).backward()
    ch = m.sample_chunk(z[:, -1])
    assert ch.shape == (2, 6, 5) and ch[..., :CONT_DIM].abs().max() <= 1 and 0 <= ch[..., CONT_DIM].min()


def test_text_changes_the_belief_for_every_backend():
    for kind in ("bow", "scratch"):
        backend = TextBackend(TextConfig(kind=kind))
        m = VLAPolicy(arch="mlp", text=backend.cfg).eval()
        imgs, pr, _ = batch(m, backend, B=1)
        za, _ = m.features(imgs, pr, backend.encode([TEXTS[0]]))
        zb, _ = m.features(imgs, pr, backend.encode(["put the blue block onto the red cube"]))
        assert not torch.allclose(za, zb)  # word order matters


@pytest.mark.parametrize("arch", ["gru", "stack", "mlp"])
@pytest.mark.parametrize("head", ["regress", "flow"])
def test_runner_replans_every_k_steps(arch, head):
    env = DynamicStackEnv(EnvConfig(seed=0, img_size=48))
    obs, _ = env.reset(seed=0)
    backend = TextBackend(TextConfig(kind="bow"))
    runner = PolicyRunner(
        VLAPolicy(arch=arch, head=head, chunk=8, text=backend.cfg), exec_horizon=4, text_backend=backend
    )
    for _ in range(12):
        a = runner.act(obs)
        assert a.shape == (5,) and np.all(np.abs(a[:4]) <= 1) and 0 <= a[4] <= 1
        obs, *_ = env.step(a)
    assert runner.replans == 3


def test_pretrained_backend_if_available():
    pytest.importorskip("transformers")
    try:
        backend = TextBackend(TextConfig(kind="pretrained"))
    except Exception as e:  # no network / hub unavailable
        pytest.skip(f"pretrained text encoder unavailable: {e}")
    out = backend.encode(TEXTS)
    assert out["emb"].shape[0] == 2 and out["emb"].shape[-1] == backend.cfg.emb_dim
    m = VLAPolicy(arch="mlp", text=backend.cfg)
    imgs, pr, txt = batch(m, backend)
    z, _ = m.features(imgs, pr, txt)
    assert z.shape[:2] == (2, 5)
    assert not any("lm" in n for n in m.state_dict())  # the frozen LM is not part of the checkpoint


def test_training_run_checkpoint_roundtrip(tmp_path, monkeypatch):
    from vla_mujoco.data import collect
    from vla_mujoco.vla import train

    data, out = tmp_path / "d", tmp_path / "r"
    collect_args = f"--out {data} --episodes 3 --workers 1 --img-size 48 --task stack"
    monkeypatch.setattr(sys, "argv", ["collect", *collect_args.split()])
    collect.main()
    base = f"--data {data} --out {out} --arch gru --head flow --text-encoder scratch --batch 2 --seq 10 --burn-in 2"
    base += " --val-every 2 --val-batches 1 --patch 8 --slots"
    monkeypatch.setattr(sys, "argv", ["train", *base.split(), "--steps", "4", "--warmup", "1"])
    train.main()
    for f in ("policy.pt", "best.pt", "last.pt", "log.csv"):
        assert (out / f).exists()
    model = VLAPolicy.from_checkpoint(torch.load(out / "policy.pt", map_location="cpu"))
    assert model.head_kind == "flow" and model.arch == "gru" and model.text_cfg.kind == "scratch"
    # CLI options must reach the model (a silently ignored flag would invalidate every experiment)
    assert model.cfg["patch"] == 8 and model.cfg["slots"] is True and model.slots
    # resuming continues from the saved step
    monkeypatch.setattr(sys, "argv", ["train", *base.split(), "--steps", "6", "--resume"])
    train.main()
    assert torch.load(out / "last.pt", map_location="cpu")["step"] == 6


def test_slots_make_the_action_pathway_depend_on_the_grounding_head():
    """With grounded object slots the belief is a function of the grounding head; without them it is not."""
    backend = TextBackend(TextConfig(kind="bow"))
    for slots, expect_grad in ((True, True), (False, False)):
        m = VLAPolicy(arch="mlp", text=backend.cfg, slots=slots)
        imgs, pr, txt = batch(m, backend)
        z, _ = m.features(imgs, pr, txt)
        z.sum().backward()
        g = m.g_pick.weight.grad
        assert (g is not None and float(g.abs().sum()) > 0) == expect_grad, slots


def test_slots_checkpoint_roundtrip_and_text_dependence():
    backend = TextBackend(TextConfig(kind="bow"))
    m = VLAPolicy(arch="gru", text=backend.cfg, slots=True).eval()
    m2 = VLAPolicy.from_checkpoint({"model": m.state_dict(), "cfg": m.cfg})
    assert m2.slots
    imgs, pr, _ = batch(m, backend, B=1)
    za, _ = m.features(imgs, pr, backend.encode([TEXTS[0]]))
    zb, _ = m.features(imgs, pr, backend.encode(["put the blue block onto the red cube"]))
    assert not torch.allclose(za, zb)


def test_validation_works_when_features_come_back_in_bfloat16(tmp_path, monkeypatch):
    """Under --amp the fusion/GRU output is bf16; validation once crashed feeding it to float32 heads (GPU only)."""
    from vla_mujoco.data import collect
    from vla_mujoco.vla import train

    data = tmp_path / "d"
    monkeypatch.setattr(
        sys, "argv", ["c", *f"--out {data} --episodes 3 --workers 1 --img-size 32 --task stack".split()]
    )
    collect.main()
    eps = train.load_data(data, ["front", "wrist"])
    backend = TextBackend(TextConfig(kind="bow"))
    strings = sorted({e["instr"] for e in eps})
    text = {"strings": strings, "tensors": backend.encode(strings)}
    model = VLAPolicy(arch="gru", text=backend.cfg, patch=8, slots=True).eval()
    real = model.features

    def bf16_features(*a, **k):
        out = real(*a, **k)
        return (out[0].bfloat16(), *out[1:])

    monkeypatch.setattr(model, "features", bf16_features)
    batcher = train.Batcher(eps, text, model.chunk, 10, "cpu")
    v = train.validate(model, batcher, np.random.default_rng(0), 1, 2, 2, amp=False)
    assert np.isfinite(v["act_l1"]) and "lang_gap_dec" in v
