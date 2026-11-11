"""Decision-frame weighting, the shared validation split, confidence-interval statistics and the language probe."""

import sys

import numpy as np
import pytest

from vla_mujoco.vla.decision import APPROACH, TRANSFER, decision_mask, split_indices
from vla_mujoco.vla.stats import bootstrap_ci, sign_test_p, summarize

# phases: approach descend lift transfer track release retreat
PH = [0, 0, 0, 0, 1, 1, 2, 3, 3, 3, 4, 5, 6, 0, 0, 1]


def test_decision_mask_marks_the_start_of_each_approach_and_transfer():
    m = decision_mask(np.array(PH), window=2)
    assert m.tolist() == [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0]
    assert (APPROACH, TRANSFER) == (0, 3)


def test_decision_mask_never_leaks_into_the_next_phase_and_handles_edges():
    m = decision_mask(np.array([0, 3, 4, 4, 0]), window=10)  # window longer than the phase
    assert m.tolist() == [1, 1, 0, 0, 1]
    assert not decision_mask(np.array([1, 2, 4, 5, 6])).any()  # no approach/transfer -> nothing
    assert decision_mask(np.array([0] * 30), window=10).sum() == 10  # one long approach: only its start counts
    assert decision_mask(np.array([], int)).shape == (0,)


def test_split_is_deterministic_disjoint_and_covers_everything():
    tr, va = split_indices(100, 0.1)
    assert len(va) == 10 and set(tr).isdisjoint(va) and len(set(tr) | set(va)) == 100
    assert np.array_equal(split_indices(100, 0.1)[1], va)
    tr2, va2 = split_indices(3, 0.1)  # tiny datasets: never an empty train or validation set
    assert len(va2) >= 1 and len(tr2) >= 1


def test_bootstrap_ci_and_sign_test_match_known_values():
    assert sign_test_p([1] * 9 + [-1]) == pytest.approx(2 * (1 + 10) / 1024)
    assert sign_test_p([1] * 5 + [-1] * 5) == 1.0 and sign_test_p([0, 0]) == 1.0
    rng = np.random.default_rng(0)
    x = rng.normal(0.3, 1.0, 2000)
    mean, lo, hi = bootstrap_ci(x, seed=1)
    assert lo < 0.3 < hi and hi - lo < 0.15 and lo < mean < hi
    assert bootstrap_ci([])[0] != bootstrap_ci([])[0]  # nan for empty input


def test_summarize_separates_a_real_gap_from_noise():
    rng = np.random.default_rng(3)
    real = summarize(0.02 + rng.normal(0, 0.01, 40))
    noise = summarize(rng.normal(0, 0.01, 40))
    zero = summarize(np.zeros(12))
    assert real["significant"] and real["frac_positive"] > 0.9 and real["sign_p"] < 0.001
    assert not noise["significant"] and not zero["significant"] and zero["frac_positive"] == 0.0
    assert summarize([float("nan"), 0.1, 0.2])["n"] == 2  # NaN episodes (no decision frames) are skipped


torch = pytest.importorskip("torch")
from vla_mujoco.vla.policy import VLAPolicy  # noqa: E402
from vla_mujoco.vla.text import TextBackend, TextConfig  # noqa: E402


@pytest.mark.parametrize("head", ["regress", "flow"])
def test_weighted_loss_is_a_weighted_mean_and_unweighted_is_unchanged(head):
    torch.manual_seed(0)
    m = VLAPolicy(arch="mlp", head=head, text=TextConfig(kind="bow"), chunk=4)
    z = torch.randn(2, 6, m.hidden)
    tgt = torch.rand(2, 6, 4, 5) * 2 - 1
    tgt[..., 4] = (tgt[..., 4] > 0).float()
    torch.manual_seed(1)
    plain = m.loss_terms(z, tgt)["act"]
    torch.manual_seed(1)
    ones = m.loss_terms(z, tgt, torch.ones(2, 6))["act"]
    assert torch.allclose(plain, ones, atol=1e-6)  # weights of 1 reproduce the unweighted loss exactly
    w = torch.ones(2, 6)
    w[:, :2] = 50.0
    torch.manual_seed(1)
    heavy = m.loss_terms(z, tgt, w)["act"]
    torch.manual_seed(1)
    head_only = m.loss_terms(z[:, :2], tgt[:, :2])["act"]
    assert (
        abs(float(heavy) - float(head_only)) < abs(float(plain) - float(head_only)) + 1e-9
    )  # pulled toward the heavy frames
    assert not torch.allclose(heavy, plain)


def test_probe_end_to_end_on_a_tiny_model(tmp_path, monkeypatch):
    """Collect -> train -> probe: the report has every metric, and a model that ignores the text has gap 0 exactly."""
    from vla_mujoco.data import collect
    from vla_mujoco.vla import probe, train

    data, out = tmp_path / "d", tmp_path / "r"
    monkeypatch.setattr(
        sys, "argv", ["c", *f"--out {data} --episodes 8 --workers 1 --img-size 32 --task random".split()]
    )
    collect.main()
    base = f"--data {data} --out {out} --arch mlp --text-encoder bow --batch 2 --seq 8 --burn-in 2 --val-every 2"
    monkeypatch.setattr(sys, "argv", ["t", *base.split(), "--val-batches", "1", "--steps", "2", "--patch", "8",
                                      "--decision-w", "4", "--val-frac", "0.5"])  # fmt: skip
    train.main()
    ck = torch.load(out / "policy.pt", map_location="cpu")
    model = VLAPolicy.from_checkpoint(ck).eval()
    from vla_mujoco.vla.train import load_data

    eps = load_data(data, model.cameras)
    res = probe.run_probe(model, TextBackend(model.text_cfg), eps[:4], "cpu", n_wrong=2, progress=False)
    assert res["episodes"] == 4
    for k in ("action_gap_all", "action_gap_decision", "ground_gap_all", "ground_gap_decision"):
        assert {"mean", "ci_lo", "ci_hi", "frac_positive", "sign_p", "significant"} <= set(res[k])
    assert "SIGNIFICANT" not in probe.format_report(res) and "held-out episodes" in probe.format_report(res)
