#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEMO="$(realpath "$1")"

python3 - "$DEMO" <<'PY'
import subprocess
import sys
from pathlib import Path

demo = Path(sys.argv[1])
frames = sorted((demo / "frames").glob("frame_*.png"))

if not frames:
    raise SystemExit("ERROR: No frames found")

print(f"Encoding {len(frames)} frames...", flush=True)

output = demo / "demonstration.mp4"

for codec in ("libx264", "mpeg4"):
    cmd = [
        "ffmpeg", "-y",
        "-f", "image2pipe",
        "-framerate", "20",
        "-vcodec", "png",
        "-i", "pipe:0",
        "-c:v", codec,
        "-pix_fmt", "yuv420p",
        str(output),
    ]

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    try:
        for frame in frames:
            process.stdin.write(frame.read_bytes())
        process.stdin.close()
        code = process.wait()
    except (BrokenPipeError, OSError) as exc:
        print(f"Encoding error: {exc}", file=sys.stderr)
        process.stdin.close()
        code = process.wait()

    if code == 0 and output.exists() and output.stat().st_size:
        print(f"SUCCESS: {output}")
        break
else:
    raise SystemExit("ERROR: MP4 encoding failed")
PY

ln -sfnT "$(basename "$DEMO")" data/latest_demo
