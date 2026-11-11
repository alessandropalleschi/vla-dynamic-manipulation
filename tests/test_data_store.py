"""Dataset format: memory-mappable episodes, legacy compatibility, in-place scene writing, manifests."""

import json
import subprocess
import sys

import numpy as np

from vla_mujoco.data.store import list_episodes, read_episode, write_episode, write_manifest


def fake_episode(T=6, size=8):
    rng = np.random.default_rng(0)
    return {
        "img_front": rng.integers(0, 255, (T, size, size, 3), dtype=np.uint8),
        "img_wrist": rng.integers(0, 255, (T, size, size, 3), dtype=np.uint8),
        "action": rng.normal(size=(T, 5)).astype(np.float32),
        "instruction": np.array("put the cube onto the disc"),
    }


def test_folder_episode_roundtrip_is_memory_mapped_and_exact(tmp_path):
    ep = fake_episode()
    d = write_episode(tmp_path, 7, ep)
    assert d.name == "ep_000007" and (d / "meta.npz").exists()
    z = read_episode(d)
    assert isinstance(z["img_front"], np.memmap) and not z["img_front"].flags.writeable
    for k in ("img_front", "img_wrist", "action"):
        assert np.array_equal(z[k], ep[k])
    assert str(z["instruction"]) == "put the cube onto the disc"
    assert np.array_equal(z["img_front"][[1, 3, 4]], ep["img_front"][[1, 3, 4]])  # fancy indexing, as the batcher does


def test_legacy_single_file_episodes_still_load_and_list_in_order(tmp_path):
    ep = fake_episode()
    np.savez_compressed(tmp_path / "ep_000001.npz", **ep)
    write_episode(tmp_path, 2, ep)
    names = [p.name for p in list_episodes(tmp_path)]
    assert names == ["ep_000001.npz", "ep_000002"]
    legacy, new = (read_episode(p) for p in list_episodes(tmp_path))
    assert np.array_equal(legacy["img_front"], new["img_front"]) and "action" in legacy


def test_manifest_records_how_to_rebuild(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--episodes", "5", "--seed0", "9"])
    p = write_manifest(tmp_path, "demos", "vla_mujoco.data.collect", {"seed0": 9}, episodes_saved=5)
    m = json.loads(p.read_text())
    assert m["command"] == "python -m vla_mujoco.data.collect --episodes 5 --seed0 9"
    assert m["kind"] == "demos" and m["args"] == {"seed0": 9} and m["episodes_saved"] == 5
    assert "git_commit" in m and "git_dirty" in m and m["vla_mujoco_version"]


def run(*args):
    return subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True, timeout=300)


def test_static_scenes_are_written_in_place_by_workers_and_load_back(tmp_path):
    from vla_mujoco.vla.pretrain_grounding import load_scenes

    out = tmp_path / "scenes"
    r = run("vla_mujoco.data.static_scenes", "--out", str(out), "--scenes", "25", "--per-shard", "10",
            "--workers", "2", "--img-size", "32")  # fmt: skip
    assert r.returncode == 0, r.stderr[-800:]
    imgs, uv, vis, instr = load_scenes(out, ["front", "wrist"])
    assert imgs[0].shape == (25, 32, 32, 3) and isinstance(imgs[0], np.memmap)
    assert uv.shape == (25, 2, 2, 2) and vis.shape == (25, 2, 2) and len(instr) == 25
    # every row was filled by some worker (none left as the zero-initialised blank), including the last short shard
    assert all(im[i].any() for im in imgs for i in range(25))
    assert len(set(instr)) > 5  # distinct scenes, not 25 copies of one
    m = json.loads((out / "manifest.json").read_text())
    assert m["kind"] == "static_scenes" and m["scenes"] == 25 and "--scenes 25" in m["command"]


def test_scene_generation_is_deterministic_per_seed(tmp_path):
    outs = []
    for name in ("a", "b"):
        out = tmp_path / name
        r = run("vla_mujoco.data.static_scenes", "--out", str(out), "--scenes", "6", "--per-shard", "3",
                "--workers", "1", "--img-size", "32")  # fmt: skip
        assert r.returncode == 0, r.stderr[-800:]
        outs.append(out)
    a, b = (np.load(o / "img_front.npy") for o in outs)
    assert np.array_equal(a, b)
    assert str(np.load(outs[0] / "meta.npz")["instruction"][0]) == str(np.load(outs[1] / "meta.npz")["instruction"][0])
