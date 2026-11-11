"""Guards the repo-hygiene setup: hooks in the dev container, the large-file/artifact guard, Git LFS for meshes."""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def read(rel):
    return (ROOT / rel).read_text()


@pytest.mark.parametrize("profile", ["gpu", "cpu"])
def test_every_devcontainer_profile_installs_the_hooks(profile):
    cfg = json.loads(read(f".devcontainer/{profile}/devcontainer.json"))
    assert "post-create.sh" in cfg["postCreateCommand"]
    post = read(".devcontainer/post-create.sh")
    assert "install-hooks.sh" in post
    # the hooks are installed after the train+dev extras (the dev extra provides pre-commit)
    assert post.index('pip install -e ".[train,dev]"') < post.index("install-hooks.sh")


def test_install_hooks_script_installs_and_verifies():
    s = read(".devcontainer/install-hooks.sh")
    assert "set -euo pipefail" in s
    assert "git lfs install" in s and "pre_commit install --install-hooks" in s
    assert "exit 1" in s  # verification fails loudly
    assert "install-hooks.sh" in read("Makefile")  # `make hooks` / `make install` share the same script


def test_image_and_extras_provide_the_tools():
    assert "git-lfs" in read(".devcontainer/Dockerfile")
    assert "pre-commit" in re.search(r"dev = \[(.*?)\]", read("pyproject.toml")).group(1)


def test_precommit_config_blocks_large_files_and_artifacts():
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(read(".pre-commit-config.yaml"))
    hooks = {h["id"]: h for repo in cfg["repos"] for h in repo["hooks"]}
    big = hooks["check-added-large-files"]
    maxkb = int(next(a for a in big["args"] if a.startswith("--maxkb=")).split("=")[1])
    assert maxkb <= 1000
    guard = hooks["no-artifacts"]
    pattern = re.compile(guard["files"])
    for bad in (
        "runs4/x.txt",
        "runs/a/b.log",
        "data/ep.npz",
        "data_oak/manifest.json",
        "model.pt",
        "x/ckpt.safetensors",
        "demo.mp4",
    ):
        assert pattern.search(bad), bad
    for ok in ("vla_mujoco/data/collect.py", "vla_mujoco/vla/train.py", "docs/MUJOCO.md", "tests/test_repo_hygiene.py"):
        assert not pattern.search(ok), ok  # source code under a `data` package must stay committable


def test_meshes_are_lfs_tracked_and_ignored_outputs_cover_all_run_dirs():
    attrs = read(".gitattributes")
    for ext in ("stl", "dae", "obj"):
        assert re.search(rf"^\*\.{ext} filter=lfs", attrs, re.M), ext
    assert re.search(r"^runs\*/$", read(".gitignore"), re.M)
    assert re.search(r"^data/\*$", read(".gitignore"), re.M)
    assert re.search(r"^/data_\*/$", read(".gitignore"), re.M)


def test_dockerfile_drops_the_expired_yarn_apt_source_before_updating():
    """The base image's Yarn apt key has expired; `apt-get update` fails with NO_PUBKEY unless the source is removed."""
    df = "\n".join(line for line in read(".devcontainer/Dockerfile").splitlines() if not line.lstrip().startswith("#"))
    assert "yarn.list" in df
    assert df.index("yarn.list") < df.index("apt-get update") < df.index("apt-get install")


# ----------------------------------------------------------------- Claude / Codex in the dev container
@pytest.mark.parametrize("profile", ["gpu", "cpu"])
def test_devcontainer_has_claude_codex_extension_features_and_persistent_state(profile):
    cfg = json.loads(read(f".devcontainer/{profile}/devcontainer.json"))
    assert "anthropic.claude-code" in cfg["customizations"]["vscode"]["extensions"]
    assert any(f.startswith("ghcr.io/devcontainers/features/node") for f in cfg["features"])  # npm for the CLIs
    assert cfg["initializeCommand"] == "bash .devcontainer/init-host.sh"  # runs on the host before the mounts
    targets = {m.split("target=")[1].split(",")[0]: m for m in cfg["mounts"]}
    state = {"/home/vscode/.claude", "/home/vscode/.claude.json", "/home/vscode/.codex"}
    assert state <= set(targets)  # other mounts (cache volumes) may coexist
    assert all("${localEnv:HOME}" in targets[t] and "VLA_DEV_PROFILE" in targets[t] for t in state)
    assert all("type=bind" in targets[t] for t in state)
    assert cfg["remoteUser"] == "vscode"  # the mount targets live in this user's home


def test_post_create_installs_both_clis_and_vscode_recommends_the_extension():
    post = read(".devcontainer/post-create.sh")
    assert "@anthropic-ai/claude-code" in post and "@openai/codex" in post
    assert "anthropic.claude-code" in json.loads(read(".vscode/extensions.json"))["recommendations"]


def test_init_host_script_creates_private_persistent_state_without_clobbering(tmp_path):
    import os
    import stat
    import subprocess

    script = ROOT / ".devcontainer" / "init-host.sh"
    env = {**os.environ, "HOME": str(tmp_path), "VLA_DEV_PROFILE": "t"}

    def run():
        return subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)

    assert run().returncode == 0
    claude, js, codex = tmp_path / ".claude-t", tmp_path / ".claude-t.json", tmp_path / ".codex-t"
    assert claude.is_dir() and codex.is_dir() and js.is_file() and js.read_text().strip() == "{}"
    assert stat.S_IMODE(claude.stat().st_mode) == 0o700 and stat.S_IMODE(js.stat().st_mode) == 0o600
    js.write_text('{"keep": "me"}')  # a second run must not overwrite an existing login
    (codex / "auth.json").write_text("secret")
    assert run().returncode == 0
    assert js.read_text() == '{"keep": "me"}' and (codex / "auth.json").read_text() == "secret"
    # a directory where the JSON file should be makes docker mount the wrong thing: fail loudly instead
    js.unlink()
    js.mkdir()
    bad = run()
    assert bad.returncode == 1 and "is a directory" in bad.stderr


def test_devcontainer_json_has_no_duplicate_keys_and_keeps_all_mounts():
    """A duplicated key (e.g. two "mounts") is legal JSON but silently drops one of them; guard against it."""

    def no_dupes(pairs):
        keys = [k for k, _ in pairs]
        assert len(keys) == len(set(keys)), f"duplicate keys: {sorted({k for k in keys if keys.count(k) > 1})}"
        return dict(pairs)

    for profile in ("gpu", "cpu"):
        cfg = json.loads(read(f".devcontainer/{profile}/devcontainer.json"), object_pairs_hook=no_dupes)
        targets = {m.split("target=")[1].split(",")[0] for m in cfg["mounts"]}
        assert {"/home/vscode/.claude", "/home/vscode/.claude.json", "/home/vscode/.codex"} <= targets
        assert {"/home/vscode/.cache/pip", "/home/vscode/.cache/pre-commit"} <= targets  # cache volumes


def test_training_dependencies_are_declared():
    """The default text encoder needs transformers; it was once installed by hand and missing in a fresh container."""
    import tomllib

    train = tomllib.loads(read("pyproject.toml"))["project"]["optional-dependencies"]["train"]
    assert {"torch", "transformers"} <= {d.split("[")[0].split(">")[0].split("=")[0].strip() for d in train}
    assert '".[train,dev]"' in read(".devcontainer/post-create.sh")
