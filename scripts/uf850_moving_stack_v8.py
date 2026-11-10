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

BELT_M_S = 0.02
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


def choose_hover(model, planfile, targets):
    path = Path(planfile)
    v7.check(path.exists(), 'Missing validated hover plan; regenerate a longer-cup hover IK first')
    d = json.loads(path.read_text())
    v7.check(d.get('success') is True and d.get('joint_names') == v7.NAMES,
             'Saved hover file is not a verified MoveIt plan')
    q = np.asarray(d['waypoints'][-1]['positions'], dtype=float)
    T = model.T(q)
    v7.check(abs(T[2, 3] - (targets['pickup_seal'][2] + .116)) < .035 and
             np.linalg.norm(T[:2, 3] - targets['pickup_seal'][:2]) < .025 and
             T[2, 2] > .99,
             'Hover plan uses outdated TCP or incorrect green-box position')
    return q


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
    qhover = choose_hover(model, args.hover_plan, t)
    station = MovingStation(model, world, Path(args.reports))
    try:
        station.ready()
        station.belt(False)
        p = station.live_orange()
        v7.check(np.linalg.norm(p-world['lower_box']['center_world']) < .005,
                 'Orange box not at reset pose; restart Gazebo before trial')
        v7.check(station.state['attached'] is False and
                 'a=0,b=0' in station.state['diag'].replace(' ', ''),
                 'Gripper must start detached with both vacuum channels OFF')
        print('STAGE 00: new collision-aware plan to validated hover joints')
        hover = station.plan('00_hover_moving_trial', qhover)
        station.move('00_hover_moving_trial', hover)
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
        orange_now = station.live_orange()
        lead = 2.0
        plan = None
        for i in range(3):
            predicted = orange_now.copy()
            predicted[0] += BELT_M_S * lead
            v7.check(predicted[0] <= -0.38, 'Interception target beyond tested conveyor region')
            item = station.world['lower_box']
            item['center_world'] = predicted.copy()
            item['center'] = predicted - np.asarray(v7.SPAWN_WORLD)
            target = t['place_near'].copy()
            target[:2] = predicted[:2]
            q = model.solve_near(station.check_fresh(), target, orient)
            plan = station.plan('05_place_near', q, holding=True)
            duration = float(plan['waypoints'][-1]['seconds'])
            lead = duration + .05  # controller scheduling / endpoint settle allowance
            print('PREDICT:', round(lead, 3), 's to release, orange x:', round(predicted[0], 4))
        v7.check(plan is not None, 'No moving-placement plan')
        predicted_x = float(station.world['lower_box']['center_world'][0])
        print('MOVING INTERCEPT:', 'target x', round(predicted_x, 4),
              ', planned descent', round(duration, 3), 's')
        station.move('05_place_near', plan, carrying=True)

        observed = station.live_orange()[0] - station.belt_x0

        print(
            'MEASURED BELT TRAVEL:',
            round(observed * 1000, 2), 'mm',
            'equivalent moving time:',
            round(observed / BELT_M_S, 3), 's'
        )

        station.vacuum(False)
        station.spin(.55)
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
    ap.add_argument('--hover-plan', type=Path, default=Path('/workspace/data/uf850_hover_plan.json'))
    ap.add_argument('--reports', default='/workspace/data/uf850_stack_moving_v8')
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
