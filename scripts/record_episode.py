#!/usr/bin/env python3
"""Record overhead RGB + simulation state around an unmodified v9 expert run.

Images are saved as lossless PNG using only stdlib. Stream JSONL records
carry simulation timestamps for later alignment; labels are privileged
Gazebo truth and must not be fed to a vision-only policy.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import zlib

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import Pose
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Bool, String

CAMERA = '/station/overhead/image'
ORANGE = '/simulation/lower_box/pose'
GREEN = '/simulation/upper_box/pose'


def png_chunk(kind, data):
    body = kind + data
    return struct.pack('!I', len(data)) + body + struct.pack('!I', zlib.crc32(body) & 0xffffffff)


def save_png(msg, destination):
    """Convert an uncompressed ROS sensor_msgs/Image into a lossless PNG."""
    w, h, enc = int(msg.width), int(msg.height), msg.encoding.lower()
    if not 0 < w <= 2048 or not 0 < h <= 2048:
        raise ValueError(f'Unexpected frame dimensions: {w}x{h}')
    channels = {'rgb8': 3, 'bgr8': 3, 'mono8': 1, 'rgba8': 4, 'bgra8': 4}.get(enc)
    if channels is None:
        raise ValueError(f'Unsupported image encoding: {enc}')
    step = int(msg.step)
    raw = bytes(msg.data)
    if step < w * channels or len(raw) < step * h:
        raise ValueError('Corrupted image step or buffer length')
    color_type = 0 if enc == 'mono8' else 2
    scan = bytearray()
    for y in range(h):
        row = raw[y * step : y * step + w * channels]
        if enc == 'bgr8':
            row = bytes(c for i in range(0, len(row), 3) for c in (row[i+2], row[i+1], row[i]))
        elif enc in ('rgba8', 'bgra8'):
            order = (0, 1, 2) if enc == 'rgba8' else (2, 1, 0)
            row = b''.join(bytes((row[i+order[0]], row[i+order[1]], row[i+order[2]])) for i in range(0, len(row), 4))
        scan.append(0)  # PNG filter: none
        scan.extend(row)
    ihdr = struct.pack('!2I5B', w, h, 8, color_type, 0, 0, 0)
    destination.write_bytes(
        b'\x89PNG\r\n\x1a\n'
        + png_chunk(b'IHDR', ihdr)
        + png_chunk(b'IDAT', zlib.compress(scan, level=1))
        + png_chunk(b'IEND', b'')
    )


class Recorder(Node):
    def __init__(self, out, fps):
        super().__init__('uf850_vla_episode_recorder')
        self.out = out
        (out / 'frames').mkdir(parents=True, exist_ok=True)
        self.frames = (out / 'frames.jsonl').open('w', buffering=1)
        self.events = (out / 'events.jsonl').open('w', buffering=1)
        self.frame_count = 0
        self.last_frame_t = None
        self.clock = None
        self.joints = None
        self.green = None
        self.orange = None
        self.attached = None
        self.vacuum = None
        self.cups = None
        self.camera_info = None
        self.error = None
        self.period = 1.0 / fps
        self.create_subscription(Clock, '/clock', self.on_clock, 10)
        self.create_subscription(Image, CAMERA, self.on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/station/overhead/camera_info', self.on_camera_info, qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', self.on_joint, 20)
        self.create_subscription(Pose, ORANGE, lambda m: self.on_pose(m, 'orange'), qos_profile_sensor_data)
        self.create_subscription(Pose, GREEN, lambda m: self.on_pose(m, 'green'), qos_profile_sensor_data)
        self.create_subscription(Bool, '/vgc10/attached', self.on_attach, 10)
        self.create_subscription(String, '/vgc10/diagnostics', lambda m: self.on_vacuum(m, 'diagnostics'), 10)
        self.create_subscription(String, '/vgc10/cup_contacts', lambda m: self.on_vacuum(m, 'cups'), 10)
        self.create_subscription(Bool, '/vgc10/channel_a/enable', lambda m: self.event('command_a', bool(m.data)), 10)
        self.create_subscription(Bool, '/vgc10/channel_b/enable', lambda m: self.event('command_b', bool(m.data)), 10)

    def event(self, name, value):
        self.events.write(json.dumps({'sim_time': self.clock, 'topic': name, 'value': value})+'\n')

    def on_clock(self, m):
        self.clock = m.clock.sec + m.clock.nanosec * 1e-9

    def on_camera_info(self, m):
        if self.camera_info is None:
            self.camera_info = {'width': m.width, 'height': m.height, 'k': list(m.k), 'd': list(m.d), 'frame_id': m.header.frame_id}
            (self.out / 'camera_info.json').write_text(json.dumps(self.camera_info, indent=2))

    def on_joint(self, m):
        self.joints = dict(zip(m.name, map(float, m.position)))
        self.event('joints', self.joints)

    def on_pose(self, m, label):
        p, q = m.position, m.orientation
        value = {'xyz': [p.x, p.y, p.z], 'xyzw': [q.x, q.y, q.z, q.w]}
        setattr(self, label, value)
        self.event(label, value)

    def on_attach(self, m):
        self.attached = bool(m.data)
        self.event('attached', self.attached)

    def on_vacuum(self, m, label):
        setattr(self, 'vacuum' if label == 'diagnostics' else 'cups', m.data)
        self.event(label, m.data)

    def on_image(self, m):
        if self.error:
            return
        try:
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            if t == 0:
                t = self.clock
            if t is None:
                return
            if self.last_frame_t is not None:
                if t < self.last_frame_t - .05:
                    raise RuntimeError('Image time moved backwards; did Gazebo restart?')
                if t - self.last_frame_t < self.period - .015:
                    return
            index = self.frame_count
            name = f'{index:06d}.png'
            save_png(m, self.out / 'frames' / name)
            self.frames.write(json.dumps({
                'frame': name, 'image_sim_time': t, 'clock_at_callback': self.clock,
                'encoding': m.encoding, 'width': m.width, 'height': m.height,
                'latest_joints': self.joints, 'latest_orange': self.orange,
                'latest_green': self.green, 'latest_attached': self.attached,
                'latest_vacuum': self.vacuum, 'latest_cups': self.cups,
            })+'\n')
            self.frame_count += 1
            self.last_frame_t = t
            if self.frame_count % 50 == 0:
                print(f'RECORDING: {self.frame_count} RGB frames', flush=True)
        except Exception as e:
            self.error = str(e)
            print('RECORDER ERROR:', self.error, file=sys.stderr, flush=True)

    def close_files(self):
        self.frames.close()
        self.events.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--fps', type=float, default=5.0)
    ap.add_argument('--max-expert-wall-s', type=float, default=180.0)
    args = ap.parse_args()
    if not (1 <= args.fps <= 10):
        ap.error('fps must be 1–10')
    if os.environ.get('ROS_DOMAIN_ID') != '42':
        ap.error('Requires isolated ROS_DOMAIN_ID=42')
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {
        'domain_id': 42, 'image_topic': CAMERA, 'camera_fps_saved': args.fps,
        'expert': 'scripts/expert_moving_stack.py',
        'joint_velocity_scale': 0.6, 'joint_acceleration_scale': 0.6,
        'conveyor_speed_m_s': float(os.getenv('VLA_BELT_SPEED_M_S', '0.020')),
        'observation_preroll_sim_s': float(os.getenv('VLA_WARMUP_SIM_S', '0.0')),
        'policy_observation': ['overhead RGB', 'robot joints', 'vacuum state'],
        'privileged_training_labels': ['green_box', 'orange_box'],
        'note': 'Per-image latest state is approximate; use event timestamps for precise resampling.'
    }
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    rclpy.init()
    node = Recorder(args.out, args.fps)
    # v9 bridges the orange pose itself; only bridge overhead camera and green pose.
    bridge = subprocess.Popen([
        'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
        CAMERA + '@sensor_msgs/msg/Image[gz.msgs.Image',
        '/station/overhead/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        GREEN + '@geometry_msgs/msg/Pose[gz.msgs.Pose',
    ], stdout=(args.out / 'bridge.log').open('w'), stderr=subprocess.STDOUT)
    expert = None
    result = 2
    try:
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline and node.frame_count < 4 and not node.error:
            if bridge.poll() is not None:
                raise RuntimeError('ROS–Gazebo image bridge exited; inspect bridge.log')
            rclpy.spin_once(node, timeout_sec=.1)
        if node.error or node.frame_count < 4:
            raise RuntimeError(node.error or 'No overhead camera images; check Gazebo Sensors / OGRE2')
        if node.clock is None or node.joints is None:
            raise RuntimeError('Missing /clock or /joint_states; cannot synchronize demo')
        print('PASS: Overhead RGB camera, clock, and robot joints received', flush=True)
        env = dict(os.environ, UF850_VEL_SCALE='0.60', UF850_ACC_SCALE='0.60')
        with (args.out / 'expert.log').open('w') as log:
            expert = subprocess.Popen([
                '/usr/bin/python3', '/workspace/scripts/expert_moving_stack.py',
                '--ack-simulation-only',
                '--reports', str(args.out / 'plans'),
            ], stdout=log, stderr=subprocess.STDOUT, env=env)
            until = time.monotonic() + args.max_expert_wall_s
            while expert.poll() is None and time.monotonic() < until and not node.error:
                rclpy.spin_once(node, timeout_sec=.08)
            if node.error:
                raise RuntimeError(node.error)
            if expert.poll() is None:
                raise TimeoutError('Expert exceeded time limit')
            result = expert.returncode
        end = time.monotonic() + 2.0
        while time.monotonic() < end and not node.error:
            rclpy.spin_once(node, timeout_sec=.1)
        print(f'EXPERT EXIT: {result} | RGB FRAMES: {node.frame_count}', flush=True)
    except (RuntimeError, TimeoutError) as e:
        print('BLOCKED:', e, file=sys.stderr)
        result = 2
    finally:
        if expert is not None and expert.poll() is None:
            expert.terminate()
            try:
                expert.wait(timeout=4)
            except subprocess.TimeoutExpired:
                expert.kill(); expert.wait()
        try:
            subprocess.run(['gz', 'topic', '-t', '/conveyor/speed', '-m', 'gz.msgs.Double', '-p', 'data: 0.0'], timeout=7)
        except Exception as e:
            print('WARNING: could not command belt stop:', e, file=sys.stderr)
        for p in sorted((args.out / 'plans').glob('*.json')):
            shutil.copy2(p, args.out / ('expert_' + p.name))
        trace = (args.out / 'plans' / 'run_trace.jsonl')
        if trace.exists():
            shutil.copy2(trace, args.out / 'expert_trace.jsonl')
        bridge.terminate()
        try:
            bridge.wait(timeout=3)
        except subprocess.TimeoutExpired:
            bridge.kill(); bridge.wait()
        node.close_files()
        (args.out / 'result.json').write_text(json.dumps({
            'expert_exit_code': result, 'frames': node.frame_count,
            'has_green_truth': node.green is not None,
            'has_orange_truth': node.orange is not None,
            'has_camera_info': node.camera_info is not None,
        }, indent=2))
        node.destroy_node()
        rclpy.shutdown()
    return 0 if result == 0 and node.frame_count >= 10 and node.green is not None and node.orange is not None else 2


if __name__ == '__main__':
    sys.exit(main())
