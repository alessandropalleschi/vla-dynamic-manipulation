#!/usr/bin/env bash
# Runs on the HOST before the dev container starts (devcontainer `initializeCommand`).
# Creates the per-profile Claude / Codex state that the container bind-mounts, so logins and settings survive
# container rebuilds (same idea as the `profile_init` service in the compose setup). Safe to run repeatedly.
#
#   ~/.claude-<profile>/        -> /home/vscode/.claude
#   ~/.claude-<profile>.json    -> /home/vscode/.claude.json
#   ~/.codex-<profile>/         -> /home/vscode/.codex
#
# The profile name comes from $VLA_DEV_PROFILE (default: vla_dev); use different names to keep separate logins.
# A bind mount of a missing host path would make docker create it as a root-owned directory (and a directory where
# a file is expected), which is why the paths are created up front with the right type and permissions.
set -euo pipefail

PROFILE="${VLA_DEV_PROFILE:-vla_dev}"
H="${HOME:?HOME is not set}"
CLAUDE_DIR="$H/.claude-$PROFILE"
CLAUDE_JSON="$H/.claude-$PROFILE.json"
CODEX_DIR="$H/.codex-$PROFILE"

if [ -d "$CLAUDE_JSON" ]; then
  echo "ERROR: $CLAUDE_JSON is a directory, expected a file (remove it and re-run)" >&2
  exit 1
fi

mkdir -p "$CLAUDE_DIR" "$CODEX_DIR"
[ -e "$CLAUDE_JSON" ] || echo '{}' > "$CLAUDE_JSON"

chmod 700 "$CLAUDE_DIR" "$CODEX_DIR"
chmod 600 "$CLAUDE_JSON"
echo "dev container profile '$PROFILE': Claude/Codex state in $CLAUDE_DIR, $CLAUDE_JSON, $CODEX_DIR"
