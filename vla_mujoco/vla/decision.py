"""Decision frames: the steps where the instruction, not the arm's own motion, determines what happens next.

The demonstrations are dominated by frames where the goal is already implied by the proprioceptive velocity (the arm
is mid-way to the object), so a teacher-forced action loss gives language almost no gradient. The informative frames
are the ones right after the expert *chooses* a target: the start of an ``approach`` (which object to pick) and of a
``transfer`` (which object to place onto). Pure numpy, no torch.
"""

from __future__ import annotations

import numpy as np

# indices into vla_mujoco.data.collect.PHASES
APPROACH, TRANSFER = 0, 3


def decision_mask(phase: np.ndarray, window: int = 10) -> np.ndarray:
    """Boolean (T,) mask: True for ``window`` steps from the moment each approach / transfer phase begins."""
    phase = np.asarray(phase)
    T = len(phase)
    mask = np.zeros(T, bool)
    prev = np.concatenate([[-1], phase[:-1]])
    for t in np.flatnonzero((phase != prev) & np.isin(phase, (APPROACH, TRANSFER))):
        mask[t : t + window] = True
    return mask & np.isin(phase, (APPROACH, TRANSFER))  # stays inside the phase that started it


def split_indices(n: int, val_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic episode-level train / validation split (shared by the trainer and the probe)."""
    order = np.random.default_rng(0).permutation(n)
    n_val = max(1, int(round(n * val_frac))) if n > 4 else 0
    val = order[:n_val] if n_val else order[:1]
    train = order[n_val:] if n_val else order
    return train, val
