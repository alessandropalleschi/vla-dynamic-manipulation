#!/usr/bin/env python3
"""Encode a Gazebo demo's PNG files via FFmpeg stdin, avoiding image2 filename access.
Requires host Python 3 and FFmpeg. Does not require ROS, Pillow, or Docker.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def encode(run_dir: Path, fps: int) -> int:
    if not shutil.which('ffmpeg'):
        print('ERROR: ffmpeg not found. sudo apt install ffmpeg', file=sys.stderr)
        return 1
    frames_dir = run_dir / 'frames'
    frames = sorted(frames_dir.glob('frame_*.png'))
    if not frames:
        print(f'ERROR: No PNG frames under {frames_dir}', file=sys.stderr)
        return 1
    # Verify sequence and basic PNG magic; prevent silently truncated videos.
    for index, path in enumerate(frames):
        expected = f'frame_{index:06d}.png'
        if path.name != expected:
            print(f'ERROR: Missing/out-of-order frame near {path.name}; expected {expected}', file=sys.stderr)
            return 1
        try:
            with path.open('rb') as f:
                if f.read(8) != b'\x89PNG\r\n\x1a\n':
                    print(f'ERROR: Not a PNG file: {path}', file=sys.stderr)
                    return 1
        except OSError as e:
            print(f'ERROR: Cannot read {path}: {e}', file=sys.stderr)
            print('If necessary: sudo chmod -R a+rX data/demo_<id>', file=sys.stderr)
            return 1
    output = run_dir / 'demonstration.mp4'
    print(f'Encoding {len(frames)} frames at {fps} FPS to {output}', flush=True)
    for codec_args in [
        ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23'],
        ['-c:v', 'mpeg4', '-q:v', '4'],
    ]:
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y',
               '-f', 'image2pipe', '-framerate', str(fps), '-vcodec', 'png',
               '-i', 'pipe:0', *codec_args,
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
        print(f"Trying encoder {codec_args[1]} (stdin streaming)", flush=True)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        io_error = None
        try:
            for frame in frames:
                with frame.open('rb') as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        proc.stdin.write(chunk)
        except (OSError, BrokenPipeError) as exc:
            io_error = exc
        finally:
            try:
                proc.stdin.close()
            except (OSError, BrokenPipeError):
                pass
            result = proc.wait()
        if result == 0 and io_error is None and output.is_file() and output.stat().st_size > 0:
            print(f'Video ready: {output}', flush=True)
            return 0
        print(f'Encoder failure rc={result} input_error={io_error}', file=sys.stderr)
    print('ERROR: Both encoders failed; the original PNGs remain intact.', file=sys.stderr)
    return 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('demo_dir', type=Path)
    p.add_argument('--fps', type=int, default=20)
    a = p.parse_args()
    if not 1 <= a.fps <= 120:
        p.error('--fps must be 1..120')
    return encode(a.demo_dir.expanduser().resolve(), a.fps)


if __name__ == '__main__':
    raise SystemExit(main())
