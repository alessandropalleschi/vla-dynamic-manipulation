#!/usr/bin/env bash
# Run from the project root after unzipping the patch.
set -euo pipefail
cd "$(dirname "$0")/.."
WORLD=simulation/worlds/conveyor_station.sdf
if [[ ! -f "$WORLD" ]]; then
  echo "Missing $WORLD; run from existing conveyor-belief-vla root" >&2
  exit 2
fi
python3 scripts/configure_overhead_camera.py --world "$WORLD"
python3 scripts/visualize_vgc10_on_gantry.py --world "$WORLD"
python3 scripts/check_v2.py
printf '\nOverhead camera configured. Run: docker compose restart sim\n'
