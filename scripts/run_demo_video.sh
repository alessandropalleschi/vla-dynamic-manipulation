#!/usr/bin/env bash
# Host launcher: reset simulation, capture one expert episode, encode it to MP4.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SPEED="${1:-0.0}"
command -v ffmpeg >/dev/null || { echo 'Install ffmpeg: sudo apt install ffmpeg' >&2; exit 1; }
RUN_ID="demo_$(date -u +%Y%m%dT%H%M%SZ)_$$_$RANDOM"
OUT_DIR="data/$RUN_ID"
mkdir -p "$OUT_DIR"
echo "Recording $RUN_ID at conveyor speed $SPEED m/s"
docker compose up -d sim tools
docker compose restart sim
set +e
docker compose exec -T tools bash -c \
  'source /opt/ros/jazzy/setup.bash && exec /usr/bin/python3 /workspace/scripts/capture_demo.py --speed "$1" --out-dir "$2"' \
  _ "$SPEED" "/workspace/$OUT_DIR"
CAPTURE_RC=$?
set -e
if ! compgen -G "$OUT_DIR/frames/frame_*.png" >/dev/null; then
  echo "No PNG frames recorded. Inspect docker compose logs --tail=80 sim" >&2
  exit 1
fi
bash scripts/encode_existing_demo.sh "$OUT_DIR"
if [[ -f scripts/inspect_stack.py ]]; then
  docker compose exec -T tools bash -c \
    'source /opt/ros/jazzy/setup.bash && /usr/bin/python3 /workspace/scripts/inspect_stack.py' \
    > "$OUT_DIR/stack_report.txt" 2>&1 || true
fi
printf 'Expert log: %s\nPhase annotations: %s\n' "$OUT_DIR/expert.log" "$OUT_DIR/phases.csv"
if [[ $CAPTURE_RC -ne 0 ]]; then
  echo "WARNING: expert/capture exited with code $CAPTURE_RC. The video is retained." >&2
  exit "$CAPTURE_RC"
fi
