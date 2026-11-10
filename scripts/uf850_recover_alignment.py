#!/usr/bin/env python3
"""Recover an attached green box after a blocked moving release."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from geometry_msgs.msg import Pose
from rclpy.qos import qos_profile_sensor_data

import uf850_moving_stack_v8 as moving
import uf850_stationary_stack_v7 as v7
from uf850_moveit_common import URDF_PATH, WORLD_PATH

SOURCE = Path('/workspace/data/uf850_stack_moving_v8/05_place_near.json')
REPORTS = Path('/workspace/data/uf850_stack_moving_v8_recovery')
GREEN_TOPIC = '/simulation/upper_box/pose'


def run():
    v7.check(
        os.environ.get('ROS_DOMAIN_ID') == '42',
        'Only isolated Gazebo simulation is supported'
    )

    os.environ['UF850_VEL_SCALE'] = '0.60'
    os.environ['UF850_ACC_SCALE'] = '0.60'

    model = v7.RobotKinematics(URDF_PATH)
    world = v7.world_objects(WORLD_PATH)
    station = moving.MovingStation(model, world, REPORTS)

    green = {'xyz': None, 'time': 0.0}

    def on_green(msg):
        green['xyz'] = np.array([
            msg.position.x,
            msg.position.y,
            msg.position.z
        ])
        green['time'] = time.monotonic()

    station.node.create_subscription(
        Pose, GREEN_TOPIC, on_green, qos_profile_sensor_data
    )

    bridge = subprocess.Popen(
        [
            'ros2', 'run', 'ros_gz_bridge',
            'parameter_bridge',
            GREEN_TOPIC + '@geometry_msgs/msg/Pose[gz.msgs.Pose'
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    def fresh_green():
        deadline = time.monotonic() + 12.0

        while time.monotonic() < deadline:
            station.spin(0.12)

            if (
                green['xyz'] is not None
                and time.monotonic() - green['time'] < 0.65
            ):
                return green['xyz'].copy()

            v7.check(
                bridge.poll() is None,
                'Green-box pose bridge exited'
            )

        raise v7.Blocked('No fresh green-box pose')

    def release_metrics(T, orange):
        box_T = T @ station.attach_tf

        half_height = float(np.sum(
            np.abs(box_T[2, :3])
            * np.asarray(station.world['upper_box']['size']) / 2
        ))

        orange_top = (
            orange[2]
            + station.world['lower_box']['size'][2] / 2
        )

        gap = box_T[2, 3] - half_height - orange_top

        xy = np.linalg.norm(
            box_T[:2, 3] - orange[:2]
        )

        return float(xy), float(gap)

    try:
        station.ready()
        station.belt(False)

        current = station.check_fresh()

        v7.check(
            station.state['attached'] is True,
            'Green box is not attached'
        )

        diag = station.parse_diag()

        v7.check(
            diag.get('a') == 1 and diag.get('b') == 1,
            'Vacuum A/B must both be ON'
        )

        v7.check(
            SOURCE.is_file(),
            'Saved moving-placement trajectory missing'
        )

        saved = json.loads(SOURCE.read_text())

        v7.check(
            saved.get('success')
            and saved.get('holding_box')
            and saved.get('stage') == '05_place_near',
            'Unexpected saved trajectory'
        )

        expected = np.asarray(
            saved['waypoints'][-1]['positions']
        )

        v7.check(
            np.max(np.abs(current - expected)) < 0.025,
            'Robot is not at the previous placement endpoint'
        )

        station.spin(0.4)
        settled = station.check_fresh()

        v7.check(
            np.max(np.abs(settled - current)) < 0.004,
            'Robot is still moving'
        )

        current = settled
        T0 = model.T(current)

        green_xyz = fresh_green()
        orange = station.live_orange()

        v7.check(
            T0[2, 2] > 0.99,
            'Suction tool is not vertical'
        )

        error = np.linalg.norm(
            green_xyz[:2] - orange[:2]
        )

        v7.check(
            error < 0.025,
            'Too much misalignment for a small correction'
        )

        station.attach_tf = (
            np.linalg.inv(T0)
            @ v7.make_box_pose(green_xyz)
        )

        v7.check(
            np.linalg.norm(station.attach_tf[:2, 3]) < 0.015,
            'Unexpected carried-box transform'
        )

        station.vac_energized = True

        # Update MoveIt's orange obstacle to its real,
        # stopped position.
        item = station.world['lower_box']
        item['center_world'] = orange.copy()
        item['center'] = (
            orange - np.asarray(v7.SPAWN_WORLD)
        )

        xy, gap = release_metrics(T0, orange)

        v7.check(
            0.005 <= gap <= 0.025,
            'Unsafe vertical release gap'
        )

        print()
        print('PASS: Green box attached')
        print('PASS: Vacuum A/B ON')
        print('PASS: Robot at saved placement endpoint')
        print(
            f'CURRENT: XY={xy*1000:.2f} mm, '
            f'gap={gap*1000:.2f} mm'
        )

        if xy > 0.006:
            # Correct the measured green-box position,
            # preserving the TCP height and orientation.
            target = T0[:3, 3].copy()

            target[:2] += (
                orange[:2] - green_xyz[:2]
            )

            q = model.solve_near(
                current,
                target,
                T0[:3, :3]
            )

            plan = station.plan(
                '05_static_realign',
                q,
                holding=True
            )

            # Independent clearance checks for each waypoint.
            for waypoint in plan['waypoints']:
                T = model.T(waypoint['positions'])
                _, clearance = release_metrics(T, orange)

                v7.check(
                    abs(T[2, 3] - T0[2, 3]) < 0.005,
                    'Correction path changes height by >=5 mm'
                )

                v7.check(
                    np.linalg.norm(
                        T[:2, 3] - T0[:2, 3]
                    ) < 0.030,
                    'Correction path detours >30 mm'
                )

                v7.check(
                    T[2, 2] > 0.99
                    and clearance >= 0.005,
                    'Correction path violates box clearance'
                )

            print('PASS: Collision-aware XY correction planned')

            # Explicitly use the stationary motion method:
            # this must NOT restart the conveyor.
            v7.SimStation.move(
                station,
                '05_static_realign',
                plan,
                carrying=True
            )

        station.spin(0.4)

        final_T = model.T(station.check_fresh())
        orange = station.live_orange()
        green_xyz = fresh_green()

        xy, gap = release_metrics(final_T, orange)

        actual_xy = np.linalg.norm(
            green_xyz[:2] - orange[:2]
        )

        print()
        print(
            f'RELEASE CHECK: '
            f'predicted XY={xy*1000:.2f} mm, '
            f'Gazebo XY={actual_xy*1000:.2f} mm, '
            f'gap={gap*1000:.2f} mm'
        )

        v7.check(
            station.state['attached'] is True
            and station.parse_diag().get('a') == 1
            and station.parse_diag().get('b') == 1,
            'Attachment or vacuum state changed'
        )

        v7.check(
            xy <= 0.006
            and actual_xy <= 0.006
            and 0.005 <= gap <= 0.025
            and final_T[2, 2] > 0.99,
            'Release geometry unsafe; vacuum remains ON'
        )

        station.vacuum(False, seconds=0.5)
        station.spin(0.6)

        v7.check(
            station.state['attached'] is False,
            'Green box did not detach'
        )

        print('PASS: Green box released at supported position')

        release_T = model.T(station.check_fresh())

        placed = (
            release_T @ station.attach_tf
        )[:3, 3]

        retreat = (
            release_T[:3, 3]
            + np.array([0.0, 0.0, 0.15])
        )

        v7.go_stage(
            station, model,
            '06_retreat',
            retreat,
            release_T[:3, :3],
            False, True,
            released_position=placed
        )

        station.attach_tf = None

        print('PASS: Robot retreated')
        print('PASS: Recovery completed')

    finally:
        try:
            station.close()
        finally:
            if bridge.poll() is None:
                bridge.terminate()

                try:
                    bridge.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    bridge.kill()


if __name__ == '__main__':
    if '--ack-simulation-only' not in sys.argv:
        raise SystemExit(
            'BLOCKED: --ack-simulation-only required'
        )

    try:
        run()
    except (
        v7.Blocked,
        RuntimeError,
        ValueError,
        OSError,
        KeyError,
        ArithmeticError,
        subprocess.SubprocessError,
        KeyboardInterrupt
    ) as exc:
        print('BLOCKED:', exc, file=sys.stderr)
        raise SystemExit(2)
