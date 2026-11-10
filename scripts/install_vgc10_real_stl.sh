#!/usr/bin/env bash
# Installer runs git on Ubuntu as the unprivileged project user; XML update runs inside Docker.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SRC="$ROOT/external/onrobot_ros_assets"
MESH="$SRC/onrobot_vg_description/meshes/vgc10/visual/base_link_4cups.stl"
if [[ ! -f "$MESH" ]]; then
  echo 'Fetching the public VGC10 ROS 4-cup STL from UOsaka-Harada-Laboratory/onrobot...'
  mkdir -p "$ROOT/external"
  if [[ ! -d "$SRC/.git" ]]; then
    git clone --depth 1 --filter=blob:none --sparse \
      https://github.com/UOsaka-Harada-Laboratory/onrobot.git "$SRC"
  fi
  git -C "$SRC" sparse-checkout set onrobot_vg_description/meshes/vgc10/visual
fi
if [[ ! -s "$MESH" ]]; then
  echo "ERROR: Four-cup mesh not found: $MESH" >&2
  echo 'Inspect the upstream repo; do not substitute a generic placeholder.' >&2
  exit 1
fi
printf 'Verified authentic ROS-description 4-cup mesh: %s (%s bytes)\n' "$MESH" "$(stat -c %s "$MESH")"
# Run under Docker to avoid root-owned generated URDF permissions on the host.
docker compose -f compose.yaml -f compose.uf850.yaml -f compose.uf850.vacuum.yaml \
  exec -T tools /usr/bin/python3 /workspace/scripts/enable_vgc10_real_stl.py "$@"
echo 'Restart Gazebo using the usual three-file compose command, then capture a side image.'
