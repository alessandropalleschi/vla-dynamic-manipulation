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
    # the hooks are installed after the dev extra (which provides pre-commit) and before the slow checks
    assert post.index('pip install -e ".[dev]"') < post.index("install-hooks.sh") < post.index("pytest")


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
    for bad in ("runs4/x.txt", "runs/a/b.log", "data/ep.npz", "model.pt", "x/ckpt.safetensors", "demo.mp4"):
        assert pattern.search(bad), bad
    for ok in ("vla_mujoco/data/collect.py", "vla_mujoco/vla/train.py", "docs/MUJOCO.md", "tests/test_repo_hygiene.py"):
        assert not pattern.search(ok), ok  # source code under a `data` package must stay committable


def test_meshes_are_lfs_tracked_and_ignored_outputs_cover_all_run_dirs():
    attrs = read(".gitattributes")
    for ext in ("stl", "dae", "obj"):
        assert re.search(rf"^\*\.{ext} filter=lfs", attrs, re.M), ext
    assert re.search(r"^runs\*/$", read(".gitignore"), re.M)
    assert re.search(r"^data/\*$", read(".gitignore"), re.M)


def test_dockerfile_drops_the_expired_yarn_apt_source_before_updating():
    """The base image's Yarn apt key has expired; `apt-get update` fails with NO_PUBKEY unless the source is removed."""
    df = "\n".join(line for line in read(".devcontainer/Dockerfile").splitlines() if not line.lstrip().startswith("#"))
    assert "yarn.list" in df
    assert df.index("yarn.list") < df.index("apt-get update") < df.index("apt-get install")
