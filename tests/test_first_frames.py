"""The first frames of an episode must be trainable: the loss skips each window's burn-in, so windows have to be able to
start *before* the episode (regression test: frames 0..burn_in-1 used to receive no loss at all)."""

import numpy as np
import pytest

pytest.importorskip("torch")
from vla_mujoco.vla.train import Batcher  # noqa: E402

T, SEQ, BURN = 40, 12, 6


def fake_episode():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (T, 4, 4, 3), dtype=np.uint8)
    img[:, 0, 0, 0] = np.arange(T)  # pixel (0,0) of the first channel encodes the frame index
    frame_no = np.tile(np.arange(T, dtype=np.float32)[:, None], (1, 3))
    return dict(img=[img], proprio=frame_no, aux=np.zeros((T, 7), np.float32),
                phase=np.zeros(T, np.int64), step=np.zeros(T, np.int64), ref_uv=np.zeros((T, 1, 2, 2), np.float32),
                ref_vis=np.zeros((T, 1, 2), bool), decision=np.zeros(T, bool), action=np.zeros((T, 5), np.float32),
                instr="x", T=T)  # fmt: skip


def frames_in_loss(batcher, n=400):
    """Frame indices that land at window positions >= BURN (the part the loss covers)."""
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(n // 8):
        b = batcher.make(8, rng)
        seen |= set(b["proprio"][:, BURN:, 0].numpy().astype(int).ravel().tolist())
    return seen


def test_old_behaviour_never_trains_the_first_frames():
    b = Batcher([fake_episode()], {"strings": ["x"]}, 4, SEQ, "cpu")  # start_p = 0: windows start at s >= 0
    assert min(frames_in_loss(b)) >= BURN  # frames 0..5 never appear in the loss region


def test_anchored_windows_put_every_frame_including_frame_zero_in_the_loss():
    b = Batcher([fake_episode()], {"strings": ["x"]}, 4, SEQ, "cpu", BURN, 0.5)
    seen = frames_in_loss(b)
    assert set(range(T)) <= seen  # every frame, 0 included, is covered by the loss region sometimes


def test_anchored_window_layout_is_frame_zero_padding_then_the_episode():
    b = Batcher([fake_episode()], {"strings": ["x"]}, 4, SEQ, "cpu", BURN, 1.0)
    out = b.make(2, np.random.default_rng(0))
    p = out["proprio"][0, :, 0].numpy().astype(int)
    assert p[: BURN + 1].tolist() == [0] * (BURN + 1)  # BURN padded copies of frame 0, then the real frame 0
    assert p[BURN + 1 :].tolist() == list(range(1, SEQ - BURN))  # then the episode continues normally
    img = out["imgs"][0][0, :, 0, 0, 0].numpy() * 255  # images follow the same indices as proprioception
    assert np.allclose(img[: BURN + 1], 0) and np.allclose(img[BURN + 1], 1)
