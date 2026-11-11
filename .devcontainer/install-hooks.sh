#!/usr/bin/env bash
# Installs the git hooks (pre-commit guard + Git LFS) and verifies they are active.
# Used by the dev container (post-create) and by `make hooks`; safe to run repeatedly.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"

# a bind-mounted repo is owned by another uid inside the container, which makes git refuse to operate on it
git config --global --add safe.directory "$PWD" 2>/dev/null || true

git lfs install --local --force                     # LFS filter + hooks for this repo (meshes are LFS-tracked)
"$PY" -m pre_commit install --install-hooks   # also pre-fetches the hook environments, so commits work offline

hook="$(git rev-parse --git-path hooks/pre-commit)"
if ! grep -q "pre-commit" "$hook" 2>/dev/null; then
  echo "ERROR: the pre-commit hook is not installed at $hook (is core.hooksPath set?)" >&2
  exit 1
fi
if ! git config --get filter.lfs.clean >/dev/null; then
  echo "ERROR: the Git LFS filter is not configured for this repository" >&2
  exit 1
fi
echo "git hooks active: pre-commit (large-file / artifact guard, ruff) and Git LFS"
