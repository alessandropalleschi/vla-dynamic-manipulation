#!/usr/bin/env python3
"""Resume the moving stack from a verified attached transfer-hover pose."""

import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from geometry_msgs.msg import Pose
from rclpy.qos import qos_profile_sensor_data

import uf850_moving_stack_v8 as moving
import uf850_stationary_stack_v7 as v7
from uf850_moveit_common import URDF_PATH, WORLD_PATH

REPORTS = Path('/workspace/data/uf850_stack_moving_v8')
GREEN_TOPIC = '/simulation/upper_box/pose'


def run():
    v7.check(
        os.environ.get('ROS_DOMAIN_ID') == '42',
        'Only isolated Gazebo ROS_DOMAIN_ID=42 is supported'
    )

    # Same speed settings as the moving trial.
    os.environ['UF850_VEL_SCALE'] = '0.60'
    os.environ['UF850_ACC_SCALE'] = '0.60'

    planner = inspect.getsource(v7.SimStation.plan)

    if 'UF850_VEL_SCALE' not in planner:
        v7.check(
            'max_velocity_scaling_factor=.60' in planner
            and 'max_acceleration_scaling_factor=.60' in planner,
            'MoveIt planner is not configured for 0.60 scaling'
        )

    v7.MAX_PLAN_SPEED = 0.61
    v7.MAX_PLAN_ACCEL = 0.61

    model = v7.RobotKinematics(URDF_PATH)
    world = v7.world_objects(WORLD_PATH)
    targets = v7.stacking_targets(world)

    station = moving.MovingStation(model, world, REPORTS)

    green = {
        'xyz': None,
        'received_at': 0.0,
    }

    # FIX: pump ROS callbacks while the Gazebo CLI command runs.
    def pumped_belt_speed(value):
        cmd = [
            'gz', 'topic',
            '-t', '/conveyor/speed',
            '-m', 'gz.msgs.Double',
            '-p', f'data: {value:.6f}'
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        try:
            deadline = time.monotonic() + 7.0

            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        'Gazebo conveyor command exceeded 7 seconds'
                    )

                # Keep joint states, clock, attachment and vacuum
                # diagnostics fresh during the external command.
                station.spin(0.05)

            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode, cmd
                )
        finally:
            if process.poll() is None:
                process.kill()

            process.wait()

    moving.belt_speed = pumped_belt_speed

    def on_green(message):
        green['xyz'] = np.array([
            message.position.x,
            message.position.y,
            message.position.z,
        ], dtype=float)

        green['received_at'] = time.monotonic()

    station.node.create_subscription(
        Pose,
        GREEN_TOPIC,
        on_green,
        qos_profile_sensor_data
    )

    bridge = subprocess.Popen(
        [
            'ros2', 'run',
            'ros_gz_bridge',
            'parameter_bridge',
            GREEN_TOPIC
            + '@geometry_msgs/msg/Pose[gz.msgs.Pose'
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    try:
        station.ready()

        # Keep the conveyor stopped during recovery and planning.
        station.belt(False)
        station.spin(0.4)

        v7.check(
            station.state['attached'] is True,
            'Green box is not attached'
        )

        diag = station.parse_diag()

        v7.check(
            diag.get('a') == 1 and diag.get('b') == 1,
            'Vacuum channels A and B must both be ON'
        )

        # Confirm that the previous failure happened BEFORE
        # executing the final descent.
        saved_path = REPORTS / '04_transfer.json'

        v7.check(
            saved_path.is_file(),
            'Saved transfer plan not found'
        )

        saved = json.loads(saved_path.read_text())

        v7.check(
            saved.get('success') is True
            and saved.get('holding_box') is True,
            'Saved transfer is not a valid carrying plan'
        )

        expected = np.array(
            saved['waypoints'][-1]['positions'],
            dtype=float
        )

        actual = station.check_fresh()

        v7.check(
            np.max(np.abs(actual - expected)) < 0.025,
            'Robot is no longer at the transfer endpoint'
        )

        # Confirm the arm is stationary.
        station.spin(0.45)
        current = station.check_fresh()

        v7.check(
            np.max(np.abs(current - actual)) < 0.004,
            'Robot is still moving'
        )

        tcp = model.T(current)

        v7.check(
            np.linalg.norm(
                tcp[:2, 3] - targets['place_hover'][:2]
            ) < 0.03
            and abs(
                tcp[2, 3] - targets['place_hover'][2]
            ) < 0.035
            and tcp[2, 2] > 0.99,
            'Robot is not at the safe placement hover'
        )

        # Obtain the actual position of the carried green box.
        deadline = time.monotonic() + 12.0

        while (
            green['xyz'] is None
            or time.monotonic() - green['received_at'] >= 0.9
        ):
            v7.check(
                time.monotonic() < deadline,
                'No fresh green-box pose'
            )

            v7.check(
                bridge.poll() is None,
                'Green-box Gazebo bridge exited'
            )

            station.spin(0.12)

        green_xyz = green['xyz'].copy()

        station.attach_tf = (
            np.linalg.inv(tcp)
            @ v7.make_box_pose(green_xyz)
        )

        local = station.attach_tf[:3, 3]

        v7.check(
            np.linalg.norm(local[:2]) < 0.015
            and -0.055 < local[2] < -0.01,
            f'Unexpected carried-box transform: {local}'
        )

        # Restore the internal vacuum flag ONLY after checking
        # the actual Gazebo attachment and channel states.
        station.vac_energized = True

        print()
        print('PASS: Green box attached')
        print('PASS: Vacuum A and B ON')
        print('PASS: Robot at verified transfer hover')
        print(
            'PASS: Live green-box position:',
            np.round(green_xyz, 4)
        )

        # ---------------------------------------------
        # Fresh moving-target placement planning
        # ---------------------------------------------

        orange = station.live_orange()
        orientation = tcp[:3, :3]

        lead = 2.0
        plan = None

        for iteration in range(3):
            prediction = orange.copy()
            prediction[0] += moving.BELT_M_S * lead

            v7.check(
                prediction[0] <= -0.38,
                'Interception outside tested conveyor region'
            )

            item = station.world['lower_box']

            item['center_world'] = prediction.copy()
            item['center'] = (
                prediction - np.asarray(v7.SPAWN_WORLD)
            )

            target = targets['place_near'].copy()
            target[:2] = prediction[:2]

            q = model.solve_near(
                station.check_fresh(),
                target,
                orientation
            )

            plan = station.plan(
                '05_place_near',
                q,
                holding=True
            )

            duration = float(
                plan['waypoints'][-1]['seconds']
            )

            lead = duration + 0.45

            print(
                'PREDICT:',
                f'release lead={lead:.3f}s',
                f'orange X={prediction[0]:.4f}'
            )

        v7.check(
            plan is not None,
            'MoveIt did not generate a placement plan'
        )

        print()
        print(
            'MOVING INTERCEPT:',
            f'descent={duration:.3f}s'
        )

        # This starts the belt, then executes the trajectory.
        # The patched command keeps ROS feedback alive.
        station.move(
            '05_place_near',
            plan,
            carrying=True
        )

        # Existing moving controller checks live alignment
        # and support clearance BEFORE disabling vacuum.
        station.vacuum(False)

        station.spin(0.55)

        v7.check(
            station.state['attached'] is False,
            'Green box failed to detach'
        )

        print('PASS: Moving placement completed')

        # ---------------------------------------------
        # Retreat after successful release
        # ---------------------------------------------

        release = model.T(station.check_fresh())

        released_green = (
            release @ station.attach_tf
        )[:3, 3]

        retreat = (
            release[:3, 3]
            + np.array([0, 0, 0.15])
        )

        v7.go_stage(
            station,
            model,
            '06_retreat',
            retreat,
            release[:3, :3],
            False,
            True,
            released_position=released_green
        )

        print('PASS: Robot retreated')
        print('PASS: Moving stacking sequence completed')

    finally:
        try:
            # Stops conveyor. Does not turn vacuum OFF
            # when an unsuccessful trial leaves a box carried.
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
        TimeoutError,
        KeyError,
        ArithmeticError,
        subprocess.SubprocessError,
        KeyboardInterrupt
    ) as exc:
        print('BLOCKED:', exc, file=sys.stderr)
        raise SystemExit(2)
