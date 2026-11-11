"""On-disk dataset format: memory-mappable images + a reproducibility manifest.

Training must not need the whole dataset in RAM (a 600-episode set at 128 px is ~13 GB raw). Episodes are therefore
stored as a folder with **uncompressed** per-camera ``.npy`` image arrays, which ``np.load(..., mmap_mode="r")`` maps
so that only the pages a batch touches are read, plus one small compressed ``meta.npz`` with everything else::

    data/mj_demos/ep_000007/img_front.npy  img_wrist.npy  meta.npz
    data/mj_demos/manifest.json            (git commit, command line, args, counts)

Legacy single-file episodes (``ep_*.npz``, images inside) are still readable.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

import vla_mujoco


def write_episode(out: Path, seed: int, ep: dict) -> Path:
    """Write one episode: images as raw .npy (memmap-able), the rest into meta.npz."""
    d = Path(out) / f"ep_{seed:06d}"
    d.mkdir(parents=True, exist_ok=True)
    meta = {}
    for k, v in ep.items():
        if k.startswith("img_"):
            np.save(d / f"{k}.npy", np.ascontiguousarray(v, dtype=np.uint8))
        else:
            meta[k] = v
    np.savez_compressed(d / "meta.npz", **meta)
    return d


def list_episodes(path) -> list[Path]:
    p = Path(path)
    return sorted([*p.glob("ep_*.npz"), *(d for d in p.glob("ep_*") if d.is_dir())])


def read_episode(path: Path) -> dict:
    """Mapping of arrays; images are read-only memmaps for the folder format."""
    path = Path(path)
    if path.is_dir():
        z = dict(np.load(path / "meta.npz"))
        for f in path.glob("img_*.npy"):
            z[f.stem] = np.load(f, mmap_mode="r")
        return z
    return np.load(path)  # legacy npz (NpzFile supports ``in`` and ``[]``)


def _git(*args: str) -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001  (no git / not a checkout: the manifest just omits it)
        return ""


def write_manifest(out: Path, kind: str, module: str, args: dict, **extra) -> Path:
    """Record how a dataset was produced so it can be rebuilt exactly (``make data`` documents the commands)."""
    manifest = {
        "kind": kind,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD") or None,
        "git_dirty": bool(_git("status", "--porcelain", "--", "vla_mujoco")),
        "vla_mujoco_version": vla_mujoco.__version__,
        "command": " ".join(["python", "-m", module, *sys.argv[1:]]),
        "argv": sys.argv[1:],
        "args": args,
        **extra,
    }
    p = Path(out) / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2, default=str))
    return p
