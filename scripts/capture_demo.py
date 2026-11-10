#!/usr/bin/env python3
"""Record an automatic conveyor demonstration from the ROS RGB camera.

Starts the expert *after* camera frames arrive. Writes annotated PNG frames,
per-frame metadata, phase markers, and expert logs. The host wrapper encodes MP4.
No Gazebo object poses are used for the visuals or for the expert policy.
"""
import argparse
import csv
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import rclpy
from PIL import Image as PILImage, ImageDraw, ImageFont
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String


FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def convert_image(msg):
    """Decode raw ROS Image, including padded scanlines, into RGB Pillow image."""
    fmt = msg.encoding.lower()
    width, height, step = int(msg.width), int(msg.height), int(msg.step)
    raw = bytes(msg.data)
    if fmt == 'rgb8':
        return PILImage.frombytes('RGB', (width, height), raw, 'raw', 'RGB', step, 1)
    if fmt == 'bgr8':
        return PILImage.frombytes('RGB', (width, height), raw, 'raw', 'BGR', step, 1)
    if fmt == 'mono8':
        return PILImage.frombytes('L', (width, height), raw, 'raw', 'L', step, 1).convert('RGB')
    if fmt == 'rgba8':
        return PILImage.frombytes('RGBA', (width, height), raw, 'raw', 'RGBA', step, 1).convert('RGB')
    if fmt == 'bgra8':
        return PILImage.frombytes('RGBA', (width, height), raw, 'raw', 'BGRA', step, 1).convert('RGB')
    raise ValueError(f'Unsupported camera image encoding {msg.encoding!r}; expected rgb8/bgr8/mono8/rgba8/bgra8')


def img_sim_time(msg):
    return msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec


class DemoRecorder(Node):
    def __init__(self, out_dir):
        super().__init__('conveyor_demo_capture')
        self.out_dir = Path(out_dir)
        self.frames_dir = self.out_dir / 'frames'
        self.keyframes_dir = self.out_dir / 'keyframes'
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.keyframes_dir.mkdir(parents=True, exist_ok=True)
        self.frame_count = 0
        self.start_sim_time = None
        self.last_sim_time = None
        self.phase = 'waiting'
        self.attached = 'unknown'
        self.last_keyframe_phase = None
        self.events_file = (self.out_dir / 'frames.csv').open('w', newline='', encoding='utf-8')
        self.events = csv.writer(self.events_file)
        self.events.writerow(['frame_index', 'sim_time_s', 'sim_elapsed_s', 'phase', 'gripper'])
        self.phase_file = (self.out_dir / 'phases.csv').open('w', newline='', encoding='utf-8')
        self.phases = csv.writer(self.phase_file)
        self.phases.writerow(['frame_index_near_event', 'sim_time_s_last_frame', 'phase'])
        self.create_subscription(Image, '/camera/image_raw', self.on_image, qos_profile_sensor_data)
        self.create_subscription(String, '/expert/phase', self.on_phase, 10)
        self.create_subscription(String, '/gripper/attached_state', self.on_attached, 10)
        try:
            self.font = ImageFont.truetype(FONT_PATH, 17)
        except OSError:
            self.font = ImageFont.load_default()
        self.error = None

    def on_phase(self, msg):
        if msg.data != self.phase:
            self.phase = msg.data
            self.phases.writerow([max(0, self.frame_count - 1),
                                  '' if self.last_sim_time is None else f'{self.last_sim_time:.6f}',
                                  self.phase])
            self.phase_file.flush()
            self.get_logger().info(f'Phase: {self.phase} at camera frame {self.frame_count}')

    def on_attached(self, msg):
        self.attached = msg.data

    def on_image(self, msg):
        if self.error is not None:
            return
        try:
            image = convert_image(msg)
            sim_t = img_sim_time(msg)
            if self.start_sim_time is None:
                self.start_sim_time = sim_t
                self.get_logger().info(
                    f'Camera ready: {msg.width}x{msg.height} {msg.encoding}, stamp={sim_t:.3f}.')
            if self.last_sim_time is not None and sim_t < self.last_sim_time - 0.1:
                raise RuntimeError('Simulation time went backwards: did Gazebo restart during capture?')
            self.last_sim_time = sim_t
            elapsed = sim_t - self.start_sim_time
            line1 = f'SIM +{elapsed:6.2f}s  |  frame {self.frame_count:04d}  |  gripper: {self.attached}'
            line2 = f'PHASE: {self.phase}'
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, image.width, 49), fill=(15, 15, 15))
            draw.text((10, 4), line1, font=self.font, fill=(255, 255, 255))
            draw.text((10, 26), line2, font=self.font, fill=(255, 255, 255))
            filename = self.frames_dir / f'frame_{self.frame_count:06d}.png'
            image.save(filename, compress_level=1)
            self.events.writerow([self.frame_count, f'{sim_t:.6f}', f'{elapsed:.6f}',
                                  self.phase, self.attached])
            if self.phase != self.last_keyframe_phase:
                safe = re.sub(r'[^a-zA-Z0-9_-]+', '_', self.phase)
                image.save(self.keyframes_dir / f'{self.frame_count:06d}_{safe}.png')
                self.last_keyframe_phase = self.phase
            self.frame_count += 1
            if self.frame_count % 20 == 0:
                self.events_file.flush()
        except Exception as exc:
            self.error = str(exc)
            self.get_logger().error(self.error)

    def close(self):
        self.events_file.close()
        self.phase_file.close()


def spin_until(node, condition, timeout):
    limit = time.monotonic() + timeout
    while rclpy.ok() and not condition() and time.monotonic() < limit and not node.error:
        rclpy.spin_once(node, timeout_sec=0.10)
    if node.error:
        raise RuntimeError(node.error)
    return condition()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--speed', type=float, default=0.0, help='Conveyor speed, 0 to 0.10 m/s')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--max-expert-wall-s', type=float, default=90.0)
    parser.add_argument('--postroll-wall-s', type=float, default=2.0)
    args = parser.parse_args()
    if not math.isfinite(args.speed) or not 0 <= args.speed <= 0.10:
        parser.error('--speed must be between 0 and 0.10')
    if args.max_expert_wall_s <= 0 or args.postroll_wall_s < 0:
        parser.error('timeouts must be positive (postroll can be zero)')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = DemoRecorder(args.out_dir)
    expert = None
    try:
        if not spin_until(node, lambda: node.frame_count >= 3, 30.0):
            raise TimeoutError('No camera frames received for 30 seconds; verify /camera/image_raw')
        # Let video start *before* any robot movement.
        spin_until(node, lambda: node.frame_count >= 8, 10.0)
        node.phase = 'starting_expert'
        expert_log = (args.out_dir / 'expert.log').open('w', encoding='utf-8')
        try:
            cmd = ['/usr/bin/python3', '/workspace/scripts/run_expert.py',
                   '--speed', str(args.speed)]
            node.get_logger().info(f'Running expert: {cmd}')
            expert = subprocess.Popen(cmd, stdout=expert_log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + args.max_expert_wall_s
            while rclpy.ok() and expert.poll() is None and time.monotonic() < deadline and not node.error:
                rclpy.spin_once(node, timeout_sec=0.1)
            if node.error:
                raise RuntimeError(node.error)
            if expert.poll() is None:
                raise TimeoutError('Expert did not finish before time limit')
            result = expert.returncode
            node.get_logger().info(f'Expert exited with code {result}; recording postroll...')
            end = time.monotonic() + args.postroll_wall_s
            while rclpy.ok() and time.monotonic() < end and not node.error:
                rclpy.spin_once(node, timeout_sec=0.10)
            if node.error:
                raise RuntimeError(node.error)
            return result
        finally:
            expert_log.close()
    except (TimeoutError, RuntimeError) as err:
        print(f'ERROR: {err}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('Interrupted; finalizing any collected frames.', file=sys.stderr)
        return 130
    finally:
        if expert is not None and expert.poll() is None:
            expert.terminate()
            try:
                expert.wait(timeout=3)
            except subprocess.TimeoutExpired:
                expert.kill()
                expert.wait(timeout=2)
        print(f'Captured {node.frame_count} frames to {node.frames_dir}', flush=True)
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
