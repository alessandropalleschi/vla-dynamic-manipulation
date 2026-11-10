#!/usr/bin/env python3
"""Simulation-only VGC10 moving-target trial. Requires validated v7 station.

Belt remains stopped during grasp/lift/transfer. It moves only during the
final descent and release. No guarantees for real-robot hardware.
"""
import argparse
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import uf850_stationary_stack_v7 as v7
from uf850_moveit_common import URDF_PATH, WORLD_PATH

BELT_M_S = float(os.environ.get('VLA_BELT_SPEED_M_S', '0.020'))
WARMUP_SIM_S = float(os.environ.get('VLA_WARMUP_SIM_S', '0.0'))
MAX_X_TRAVEL_M = 0.075
RELEASE_XY_TOL_M = 0.008


def belt_speed(value):
    cmd = ['gz', 'topic', '-t', '/conveyor/speed', '-m', 'gz.msgs.Double', '-p', f'data: {value:.6f}']
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, timeout=7)


class MovingStation(v7.SimStation):
    def __init__(self, model, world, reports):
        super().__init__(model, world, reports)
        from geometry_msgs.msg import Pose
        from rclpy.qos import qos_profile_sensor_data
        self.orange = None
        self.orange_rx = 0.0
        self.bridge = subprocess.Popen(
            ['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
             '/simulation/lower_box/pose@geometry_msgs/msg/Pose[gz.msgs.Pose'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        def got_orange(m):
            self.orange = np.array([m.position.x, m.position.y, m.position.z], dtype=float)
            self.orange_rx = time.monotonic()
        self.node.create_subscription(Pose, '/simulation/lower_box/pose', got_orange,
                                      qos_profile_sensor_data)
        self.belt_running = False
        self.belt_x0 = None
        self.belt_watchdog = None

    def live_orange(self):
        self.spin(.12)
        v7.check(self.orange is not None and time.monotonic() - self.orange_rx < 0.9,
                 'Missing/stale live orange-box position (ROS Gazebo bridge)')
        return self.orange.copy()

    def ready(self):
        super().ready()
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            self.spin(.12)
            if self.bridge.poll() is not None:
                raise v7.Blocked('Orange-box pose bridge exited')
            if self.orange is not None and time.monotonic() - self.orange_rx < .8:
                print('PASS: live Gazebo orange-box pose:', np.round(self.orange, 4))
                return
        raise v7.Blocked('No orange-box pose; check ros_gz_bridge/Gazebo networking')

    def _send_belt(self, value):
        """Publish conveyor commands while servicing ROS feedback."""
        cmd = [
            'gz', 'topic',
            '-t', '/conveyor/speed',
            '-m', 'gz.msgs.Double',
            '-p', f'data: {value:.6f}'
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        try:
            deadline = time.monotonic() + 7.0

            while proc.poll() is None:
                if time.monotonic() >= deadline:
                    raise v7.Blocked(
                        'Conveyor command timed out'
                    )

                self.spin(0.04)

            v7.check(
                proc.returncode == 0,
                'Conveyor command failed'
            )

        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()

    def belt(self, on):
        if on:
            v7.check(not self.belt_running, 'Conveyor already active')
            self.belt_x0 = self.live_orange()[0]
            # Independent backup stop, in case the Python controller is terminated.
            self.belt_watchdog = subprocess.Popen(
                ['bash', '-lc', 'sleep 12; gz topic -t /conveyor/speed '
                 '-m gz.msgs.Double -p "data: 0.0"'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self._send_belt(BELT_M_S)
            self.belt_running = True
            print(f'BELT ON: +X at {BELT_M_S:.3f} m/s')
        else:
            # Even if we think the belt is stopped, send the stop command.
            self._send_belt(0.0)
            self.belt_running = False
            if self.belt_watchdog is not None:
                if self.belt_watchdog.poll() is None:
                    self.belt_watchdog.terminate()
                self.belt_watchdog = None
            print('BELT OFF')

    def move(self, stage, plan, carrying=False):
        if stage != '05_place_near':
            return super().move(stage, plan, carrying=carrying)
        v7.check(carrying and self.state['attached'] is True,
                 'Moving placement requires an attached green box')
        self.belt(True)
        try:
            return super().move(stage, plan, carrying=carrying)
        except BaseException:
            self.belt(False)
            raise

    def vacuum(self, on, seconds=.7):
        if on or not self.belt_running:
            return super().vacuum(on, seconds)
        v7.check(self.state['attached'] is True and self.vac_energized,
                 'Cannot release: carried green box or vacuum state missing')
        orange = self.live_orange()
        v7.check(orange[0] - self.belt_x0 < MAX_X_TRAVEL_M,
                 'Orange moved beyond bounded interception window')
        tcp = self.model.T(self.check_fresh())
        green_T = tcp @ self.attach_tf
        green = green_T[:3, 3]
        delta = green[:2] - orange[:2]
        height = float(np.sum(np.abs(green_T[2, :3]) * np.asarray(self.world['upper_box']['size']) / 2))
        gap = green[2] - height - (orange[2] + self.world['lower_box']['size'][2] / 2)
        print('LIVE MOVING RELEASE: horizontal error', round(float(np.linalg.norm(delta))*1000, 2),
              'mm; support gap', round(gap*1000, 2), 'mm')
        if not (np.linalg.norm(delta) <= RELEASE_XY_TOL_M and .005 <= gap <= .025
                and tcp[2, 2] > .99):
            self.belt(False)
            raise v7.Blocked('Unsafe moving-target release: belt stopped, vacuum left ON')
        try:
            super().vacuum(False, seconds=.35)
        finally:
            self.belt(False)

    def close(self):
        try:
            self.belt(False)
        except Exception as exc:
            print('WARNING: belt stop command failed:', exc, file=sys.stderr)
        if self.bridge.poll() is None:
            self.bridge.terminate()
        super().close()


def plan_fresh_hover(station, model, targets):
    """Fresh 6D IK + MoveIt plan for the green box in this episode.

    The old validated joint pose is an *IK seed only*, NOT a saved goal or
    a trajectory. Every attempt solves the currently randomized green pickup
    pose and obtains a new collision-aware plan from live joint feedback.
    """
    current = station.check_fresh()
    green = station.world['upper_box']
    target = np.asarray(targets['pickup_seal'], dtype=float).copy()
    target[2] += 0.116
    v7.check(target[2] > green['center_world'][2] + green['size'][2] / 2 + .10,
             'Pickup hover does not clear the green box')

    # A known robot configuration is only a numerical starting guess for IK.
    # No file in data/ is read to choose or execute this hover.
    nominal = np.array([-2.987797158, -0.445483165, -0.930707982,
                         0.178675985, -0.460686831, -0.740066600])
    nominal = np.clip(nominal, model.lo_vec + 0.01, model.hi_vec - 0.01)
    base_R = model.T(nominal)[:3, :3]
    orientations = [base_R] + [v7.rot_rpy([0.0, 0.0, a])
        for a in (0.0, math.pi / 2, math.pi, -math.pi / 2)]
    rng = np.random.default_rng(int(os.environ.get('VLA_IK_SEED', '91')))
    seeds = [nominal, current, (nominal + current) / 2]
    seeds.extend(np.clip(nominal + rng.normal(0, .14, 6),
                         model.lo_vec + .01, model.hi_vec - .01)
                 for _ in range(9))

    # Four 40.4mm lip disks at 44mm pitch, with physical box dimensions.
    pitch, radius = .044, .0202
    cx, cy = np.asarray(green['center_world'][:2], dtype=float)
    half_x, half_y = np.asarray(green['size'][:2], dtype=float) / 2
    found = []
    for R in orientations:
        for seed in seeds:
            try:
                q = model.solve_near(seed, target, R,
                                     tol_pos=.0015, tol_angle=.017)
                T = model.T(q)
                if T[2, 2] < .995 or np.linalg.norm(T[:3, 3] - target) > .0015:
                    continue
                if any(
                    abs((T[:3, 3] + T[:3, :3] @ [x*pitch/2, y*pitch/2, 0])[0] - cx)
                        + radius > half_x + 1e-6
                    or abs((T[:3, 3] + T[:3, :3] @ [x*pitch/2, y*pitch/2, 0])[1] - cy)
                        + radius > half_y + 1e-6
                    for x in (-1, 1) for y in (-1, 1)
                ):
                    continue
                if any(np.linalg.norm(q - old[1]) < .01 for old in found):
                    continue
                found.append((float(np.linalg.norm(q - current)), q))
            except (v7.Blocked, ValueError, np.linalg.LinAlgError):
                continue
        if len(found) >= 5:
            break
    found.sort(key=lambda item: item[0])
    v7.check(found, 'No IK hover with all four cups fitting randomized green box')

    failures = []
    for _, q in found[:8]:
        try:
            plan = station.plan('00_hover', q, holding=False)
            print('PASS: Fresh hover IK and MoveIt plan for green XYZ',
                  np.round(green['center_world'], 4).tolist(), flush=True)
            return plan
        except (v7.Blocked, RuntimeError) as exc:
            failures.append(str(exc))
    raise v7.Blocked('MoveIt rejected all randomized hover candidates: '
                     + '; '.join(failures[-3:]))


def run(args):
    v7.check(os.getenv('ROS_DOMAIN_ID') == '42', 'Requires isolated Gazebo ROS_DOMAIN_ID=42')
    v7.check(shutil.which('gz') and shutil.which('ros2'), 'Gazebo and ROS command-line tools required')
    src = inspect.getsource(v7.SimStation.plan)
    if 'UF850_VEL_SCALE' not in src:
        v7.check(bool(re.search(r'max_velocity_scaling_factor\s*=\s*0?\.60?\b', src)) and
                 bool(re.search(r'max_acceleration_scaling_factor\s*=\s*0?\.60?\b', src)),
                 'v7 planner is not configured for 0.60 velocity and acceleration scaling')
    v7.check('Remove the world obstacle before attaching' not in inspect.getsource(v7.SimStation.apply_scene),
             'Old redundant planning-scene REMOVE bug still present')
    os.environ['UF850_VEL_SCALE'] = os.environ['UF850_ACC_SCALE'] = '0.60'

    # v7 had a deliberately conservative extra 0.30 rad/s guard. The MoveIt
    # model also caps each joint at <=1rad/s and 1rad/s^2 before applying .60.
    # Preserve a separate .61rad/s/.61rad/s^2 trajectory sanity ceiling.
    v7.MAX_PLAN_SPEED = 0.61
    v7.MAX_PLAN_ACCEL = 0.61

    model = v7.RobotKinematics(args.urdf)
    world = v7.world_objects(args.world)
    t = v7.stacking_targets(world)
    v7.verify_pose_snapshot(args.snapshot, args.world)

    station = MovingStation(model, world, Path(args.reports))
    try:
        station.ready()
        station.belt(False)
        p = station.live_orange()
        v7.check(np.linalg.norm(p-world['lower_box']['center_world']) < .005,
                 'Orange box not at reset pose; restart Gazebo before trial')
        # RANDOMIZED_OBSERVATION_PREROLL
        # Observe moving target before pickup; stop to keep the validated
        # green pickup and transfer scene collision-aware and deterministic.
        v7.check(0.012 <= BELT_M_S <= 0.040,
                 'Belt speed must be in [0.012, 0.040] m/s')
        v7.check(0.0 <= WARMUP_SIM_S <= 8.0,
                 'Preroll must be in [0, 8] simulation seconds')
        if WARMUP_SIM_S > 0:
            start_clock = station.state['clock']
            v7.check(start_clock is not None, 'Missing Gazebo simulation clock')
            station.belt(True)
            try:
                deadline = time.monotonic() + max(25., WARMUP_SIM_S * 5.)
                while station.state['clock'] < start_clock + WARMUP_SIM_S:
                    v7.check(time.monotonic() < deadline,
                             'Gazebo clock stopped during preroll')
                    station.spin(.08)
            finally:
                station.belt(False)
            station.spin(.4)
            shifted = station.live_orange()
            dx = float(shifted[0] - p[0])
            v7.check(.003 < dx < .090,
                     f'Conveyor preroll had unexpected travel: {dx:.4f}m')
            v7.check(abs(shifted[1] - p[1]) < .006,
                     'Unexpected lateral drift during preroll')
            p = shifted
            station.world['lower_box']['center_world'] = p.copy()
            station.world['lower_box']['center'] = p - np.asarray(v7.SPAWN_WORLD)
            t = v7.stacking_targets(station.world)
            v7.check(p[0] + .030 <= -.385,
                     'Orange box left the tested intercept region')
            print('PREROLL PASS: orange stopped at',
                  np.round(p, 4).tolist(), flush=True)

        v7.check(station.state['attached'] is False and
                 'a=0,b=0' in station.state['diag'].replace(' ', ''),
                 'Gripper must start detached with both vacuum channels OFF')
        print('STAGE 00: solve and plan a NEW pickup hover')
        hover = plan_fresh_hover(station, model, t)
        station.move('00_hover', hover)
        orient = model.T(station.check_fresh())[:3, :3]
        v7.go_stage(station, model, '01_precontact', t['pickup_pre'], orient, False, True)
        v7.go_stage(station, model, '02_seal', t['pickup_seal'], orient, False, True)
        station.spin(.3)
        station.check_seal()
        contact_T = model.T(station.check_fresh())
        station.attach_tf = np.linalg.inv(contact_T) @ v7.make_box_pose(world['upper_box']['center_world'])
        station.vacuum(True, seconds=.75)
        deadline = time.monotonic() + 4
        while station.state['attached'] is not True and time.monotonic() < deadline:
            station.vacuum(True, seconds=.2)
        v7.check(station.state['attached'] is True, 'Gazebo did not attach green box')
        cur = model.T(station.check_fresh())
        v7.go_stage(station, model, '03_lift', cur[:3, 3] + [0, 0, .15], cur[:3, :3],
                    True, True, support_contact=True)
        cur = model.T(station.check_fresh())
        v7.go_stage(station, model, '04_transfer', t['place_hover'], cur[:3, :3], True, True)
        v7.check(station.state['attached'] is True, 'Lost object during transfer')
        cur = model.T(station.check_fresh())
        orient = cur[:3, :3]

        # V9: position the green box 30 mm ahead with the belt OFF.
        # Then wait for the actual moving orange box to arrive.
        orange_now = station.live_orange()

        v7.check(
            np.linalg.norm(
                orange_now - station.world['lower_box']['center_world']
            ) < .004,
            'Orange moved unexpectedly before intercept planning'
        )

        target = t['place_near'].copy()
        target[:2] = orange_now[:2] + [0.030, 0]

        v7.check(
            target[0] <= -.38,
            'Interception outside tested work area'
        )

        cur = model.T(station.check_fresh())
        q = model.solve_near(
            station.check_fresh(),
            target,
            cur[:3, :3]
        )

        # Use the real stopped orange-box position in MoveIt.
        station.world['lower_box']['center_world'] = orange_now.copy()
        station.world['lower_box']['center'] = (
            orange_now - np.asarray(v7.SPAWN_WORLD)
        )

        plan = station.plan(
            '05_place_near',
            q,
            holding=True
        )

        # Execute descent WITHOUT starting the conveyor.
        v7.SimStation.move(
            station,
            '05_place_near',
            plan,
            carrying=True
        )

        tcp = model.T(station.check_fresh())
        green_T = tcp @ station.attach_tf
        green = green_T[:3, 3].copy()

        orange_now = station.live_orange()

        ahead = float(green[0] - orange_now[0])
        lateral = abs(float(green[1] - orange_now[1]))

        half_h = float(np.sum(
            np.abs(green_T[2, :3])
            * np.asarray(world['upper_box']['size']) / 2
        ))

        gap = float(
            green[2] - half_h
            - (
                orange_now[2]
                + world['lower_box']['size'][2] / 2
            )
        )

        v7.check(
            .019 <= ahead <= .043 and lateral < .006,
            f'Unexpected interception setup: '
            f'lead={ahead:.4f}m, Y={lateral:.4f}m'
        )

        v7.check(
            .005 <= gap <= .025 and tcp[2, 2] > .99,
            f'Unsafe placement clearance: {gap:.4f}m'
        )

        v7.check(
            station.state['attached'] is True
            and station.vac_energized,
            'Green box no longer attached'
        )

        print(
            'INTERCEPT READY:',
            'green X', round(green[0], 4),
            'orange X', round(orange_now[0], 4),
            'lead', round(ahead * 1000, 1), 'mm',
            flush=True
        )

        # Start the belt only after positioning the robot.
        station.belt(True)

        started = time.monotonic()
        samples = []
        confirmed_motion = False
        resent = False

        while time.monotonic() - started < 10.0:

            orange = station.live_orange()
            now = station.state['clock']

            if not samples or now > samples[-1][0] + .06:
                samples.append((now, float(orange[0])))

            samples = [
                (ts, x)
                for ts, x in samples
                if ts >= now - .65
            ]

            travel = float(orange[0] - orange_now[0])

            xy = float(np.linalg.norm(
                green[:2] - orange[:2]
            ))

            if travel > .003:
                confirmed_motion = True

            # A successful command is not proof of belt motion.
            if (
                not confirmed_motion
                and not resent
                and time.monotonic() - started > 1.7
            ):
                print(
                    'No belt movement; republishing 0.020 m/s',
                    flush=True
                )

                if hasattr(station, '_send_belt'):
                    station._send_belt(BELT_M_S)
                else:
                    belt_speed(BELT_M_S)

                station.spin(.15)
                resent = True

            if (
                not confirmed_motion
                and time.monotonic() - started > 4.0
            ):
                raise v7.Blocked(
                    'Conveyor command had no physical effect: '
                    'orange box did not move'
                )

            v7.check(
                -.003 <= travel < MAX_X_TRAVEL_M,
                f'Unexpected conveyor travel: {travel:.4f}m'
            )

            v7.check(
                orange[0] <= green[0] + .007,
                'Orange passed the interception point'
            )

            # Measure actual belt speed using simulation time.
            speed = 0.0

            if (
                len(samples) >= 3
                and samples[-1][0] - samples[0][0] > .22
            ):
                speed = (
                    samples[-1][1] - samples[0][1]
                ) / (
                    samples[-1][0] - samples[0][0]
                )

            # Release only when motion AND alignment are verified.
            if (
                confirmed_motion
                and xy <= .004
                and 0.65 * BELT_M_S < speed < 1.45 * BELT_M_S
            ):
                print(
                    f'BELT VERIFIED: +{travel*1000:.1f}mm, '
                    f'speed={speed:.4f}m/s, '
                    f'XY error={xy*1000:.2f}mm',
                    flush=True
                )

                # Preserve v8's independent 8mm XY and
                # 5-25mm support-gap release checks.
                station.vacuum(False, seconds=.12)
                station.spin(.55)
                break

        else:
            raise v7.Blocked(
                'Timeout waiting for moving orange box'
            )
        v7.check(station.state['attached'] is False, 'Box failed to detach')
        print('PASS: Release completed while orange box was moving')
        release = model.T(station.check_fresh())
        green_xyz = (release @ station.attach_tf)[:3, 3]
        retreat = release[:3, 3] + [0, 0, .15]
        v7.go_stage(station, model, '06_retreat', retreat, release[:3, :3],
                    False, True, released_position=green_xyz)
        print('PASS: Moving-target sequence completed. Check Gazebo stack physics.')
    finally:
        station.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--urdf', type=Path, default=URDF_PATH)
    ap.add_argument('--world', type=Path, default=WORLD_PATH)
    ap.add_argument('--snapshot', type=Path, default=Path('/workspace/data/uf850_gz_pose_snapshot.json'))

    ap.add_argument('--reports', default='/workspace/data/expert_runs/current')
    ap.add_argument('--ack-simulation-only', action='store_true')
    args = ap.parse_args()
    if not args.ack_simulation_only:
        ap.error('Requires --ack-simulation-only; robot will move in Gazebo')
    try:
        run(args)
    except (v7.Blocked, ValueError, OSError, KeyError, RuntimeError, ArithmeticError,
            subprocess.SubprocessError, KeyboardInterrupt) as exc:
        print('BLOCKED:', exc, file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
