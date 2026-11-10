#!/usr/bin/env python3
"""Record one simulator episode with *both* policy observations and privileged truth.

Runs in the existing ROS 2 Jazzy 'tools' container. Imports the existing
video recorder, so the same annotated MP4 is available; in addition saves
unannotated RGB images, timestamped observations/actions, and true box poses.
"""
import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from capture_demo import DemoRecorder, convert_image, img_sim_time, spin_until

FIELDS = [
    'frame', 'sim_time_s', 'phase', 'gripper', 'rgb_file',
    'lower_x','lower_y','lower_z','lower_qx','lower_qy','lower_qz','lower_qw','lower_age_s',
    'upper_x','upper_y','upper_z','upper_qx','upper_qy','upper_qz','upper_qw','upper_age_s',
    'joint_x','joint_y','joint_z','joint_age_s',
    'cmd_x','cmd_y','cmd_z','belt_cmd',
]


class EpisodeRecorder(DemoRecorder):
    def __init__(self, out_dir):
        super().__init__(out_dir)
        self.rgb_dir = Path(out_dir) / 'rgb'
        self.rgb_dir.mkdir(exist_ok=True)
        self.latest_pose = {}
        self.latest_joint = None
        self.latest_command = {}
        self.telemetry_file = (Path(out_dir) / 'telemetry.csv').open('w', newline='', encoding='utf-8')
        self.telemetry = csv.DictWriter(self.telemetry_file, fieldnames=FIELDS)
        self. telemetry.writeheader()
        for label in ('lower', 'upper'):
            self.create_subscription(
                PoseStamped, f'/simulation/{label}_box/pose',
                lambda m, key=label: self.on_pose(key, m), 10)
        self.create_subscription(JointState, '/joint_states', self.on_joints, 10)
        for axis in ('x', 'y', 'z'):
            self.create_subscription(Float64, f'/gantry/{axis}_cmd',
                                     lambda m, key=axis: self.on_command(key, m), 10)
        self.create_subscription(Float64, '/conveyor/speed',
                                 lambda m: self.on_command('belt', m), 10)
        self.belt_pub = self.create_publisher(Float64, '/conveyor/speed', 10)

    def on_pose(self, name, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.latest_pose[name] = (stamp, msg)

    def on_joints(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.latest_joint = (stamp, {
            n.split('::')[-1].split('/')[-1]: float(p)
            for n, p in zip(msg.name, msg.position)
        })

    def on_command(self, name, msg):
        self.latest_command[name] = float(msg.data)

    def on_image(self, msg):
        if self.error:
            return
        idx = self.frame_count
        stamp = img_sim_time(msg)
        try:
            raw = convert_image(msg)
            name = f'frame_{idx:06d}.jpg'
            raw.save(self.rgb_dir / name, format='JPEG', quality=92)
            row = dict.fromkeys(FIELDS, '')
            row.update(frame=idx, sim_time_s=f'{stamp:.6f}', phase=self.phase,
                       gripper=self.attached, rgb_file=f'rgb/{name}')
            for key in ('lower', 'upper'):
                data = self.latest_pose.get(key)
                if not data:
                    continue
                sample_stamp, pose_msg = data
                age = stamp - sample_stamp
                row[f'{key}_age_s'] = f'{age:.4f}'
                # Reject stale and future truth to prevent misleading evaluation.
                if not (-0.04 <= age <= 0.35):
                    continue
                p = pose_msg.pose.position
                q = pose_msg.pose.orientation
                row.update({f'{key}_x': f'{p.x:.6f}', f'{key}_y': f'{p.y:.6f}',
                            f'{key}_z': f'{p.z:.6f}', f'{key}_qx': f'{q.x:.6f}',
                            f'{key}_qy': f'{q.y:.6f}', f'{key}_qz': f'{q.z:.6f}',
                            f'{key}_qw': f'{q.w:.6f}'})
            if self.latest_joint:
                sample_stamp, values = self.latest_joint
                age = stamp - sample_stamp
                row['joint_age_s'] = f'{age:.4f}'
                if -0.04 <= age <= 0.35:
                    for axis in ('x', 'y', 'z'):
                        if f'{axis}_joint' in values:
                            row[f'joint_{axis}'] = f'{values[f"{axis}_joint"]:.6f}'
            for axis in ('x','y','z'):
                if axis in self.latest_command:
                    row[f'cmd_{axis}'] = f'{self.latest_command[axis]:.6f}'
            if 'belt' in self.latest_command:
                row['belt_cmd'] = f'{self.latest_command["belt"]:.6f}'
            self.telemetry.writerow(row)
            if idx % 20 == 0:
                self.telemetry_file.flush()
            super().on_image(msg)
        except Exception as exc:
            self.error = str(exc)
            self.get_logger().error(f'Episode data capture error: {exc}')

    def close(self):
        self.telemetry_file.close()
        super().close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed', required=True, type=float)
    ap.add_argument('--out-dir', required=True, type=Path)
    ap.add_argument('--postroll-sim-s', type=float, default=4.0,
                    help='Conveyor keeps running for this many SIMULATION seconds after expert exit')
    ap.add_argument('--expert-timeout-wall-s', type=float, default=100.0)
    args = ap.parse_args()
    if not (0 <= args.speed <= 0.10) or not math.isfinite(args.speed):
        ap.error('speed must be finite in [0.0, 0.10]')
    if not (1.0 <= args.postroll_sim_s <= 30.0):
        ap.error('postroll-sim-s must be 1..30')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = EpisodeRecorder(args.out_dir)
    expert = None
    result = 99
    finish_stamp = None
    try:
        if not spin_until(node, lambda: node.frame_count >= 8, 40.0):
            raise RuntimeError('Camera did not publish 8 frames within 40 seconds')
        # A future VLA must not access these true pose topics; this collector does.
        if not spin_until(node, lambda: all(name in node.latest_pose for name in ('lower','upper'))
                          and node.latest_joint is not None, 10.0):
            raise RuntimeError('Missing true box poses or joint states. Check ROS bridge.yaml')
        with (args.out_dir / 'expert.log').open('w', encoding='utf-8') as log:
            cmd = ['/usr/bin/python3','/workspace/scripts/run_expert.py',
                   '--speed',str(args.speed),'--keep-belt-running']
            print('Expert:', ' '.join(cmd), flush=True)
            expert = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + args.expert_timeout_wall_s
            while expert.poll() is None and time.monotonic() < deadline and not node.error:
                rclpy.spin_once(node, timeout_sec=0.05)
            if node.error:
                raise RuntimeError(node.error)
            if expert.poll() is None:
                raise RuntimeError('Expert timed out')
            result = int(expert.returncode)
            finish_stamp = node.last_sim_time
            print(f'Expert returned {result} at sim_time={finish_stamp}', flush=True)
            # Continue belt motion after successful release for actual stability testing.
            # In failure cases, hold the conveyor stopped and preserve the failure video.
            last_publish = 0.0
            postroll_deadline = time.monotonic() + max(15.0, 4 * args.postroll_sim_s)
            while node.last_sim_time is not None and node.last_sim_time - finish_stamp < args.postroll_sim_s:
                if node.error:
                    raise RuntimeError(node.error)
                if time.monotonic() > postroll_deadline:
                    raise RuntimeError('Simulation clock stalled during postroll')
                if time.monotonic() - last_publish > 0.1:
                    node.belt_pub.publish(Float64(data=args.speed if result == 0 else 0.0))
                    last_publish = time.monotonic()
                rclpy.spin_once(node, timeout_sec=0.05)
        return result
    except Exception as exc:
        print(f'COLLECTOR ERROR: {exc}', file=sys.stderr)
        return 3
    finally:
        if expert is not None and expert.poll() is None:
            expert.terminate()
            try:
                expert.wait(timeout=3)
            except subprocess.TimeoutExpired:
                expert.kill()
                expert.wait()
        # Don't leave a moving belt on an aborted run.
        for _ in range(5):
            node.belt_pub.publish(Float64(data=0.0))
            rclpy.spin_once(node, timeout_sec=0.05)
        status = {'expert_exit_code': result, 'camera_frames': node.frame_count,
                  'expert_finish_sim_time_s': finish_stamp,
                  'last_camera_sim_time_s': node.last_sim_time,
                  'requested_speed_m_s': args.speed,
                  'requested_postroll_s': args.postroll_sim_s,
                  'collector_error': node.error}
        (args.out_dir / 'capture_status.json').write_text(json.dumps(status, indent=2) + '\n')
        print(f'Captured {node.frame_count} frames; output: {args.out_dir}', flush=True)
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
