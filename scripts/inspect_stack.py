#!/usr/bin/env python3
"""Report the two boxes' world poses and approximate stacking geometry.

Use after the scripted trial completes. This does NOT assert stable contact;
a physics-based success metric must also examine support and stability.
"""
import argparse
import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

LOWER_H = 0.08
UPPER_H = 0.06
LOWER_XY = 0.12
UPPER_XY = 0.09
EXPECTED_DZ = (LOWER_H + UPPER_H) / 2.0
MAX_CENTER_OFFSET_FOR_FULL_SUPPORT = (LOWER_XY - UPPER_XY) / 2.0


class Inspector(Node):
    def __init__(self):
        super().__init__('inspect_stack')
        self.lower = None
        self.upper = None
        self.joints = None
        self.create_subscription(PoseStamped, '/simulation/lower_box/pose',
                                 lambda m: setattr(self, 'lower', m), 10)
        self.create_subscription(PoseStamped, '/simulation/upper_box/pose',
                                 lambda m: setattr(self, 'upper', m), 10)
        self.create_subscription(JointState, '/joint_states',
                                 lambda m: setattr(self, 'joints', m), 10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=3.0,
                        help='Wall-clock seconds to receive pose data')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds must be positive')
    rclpy.init()
    node = Inspector()
    try:
        deadline = time.monotonic() + args.seconds
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        if node.joints is not None:
            print('Gantry joint positions (m):')
            for name, pos in zip(node.joints.name, node.joints.position):
                print(f'  {name}: {pos:+.4f}')
        else:
            print('WARNING: no /joint_states received')
        if node.lower is None or node.upper is None:
            missing = [label for label, value in
                       [('lower', node.lower), ('upper', node.upper)] if value is None]
            print('MISSING Gazebo pose topic(s): ' + ', '.join(missing))
            print('Check: ros2 topic list | grep /simulation/')
            print('Check: docker compose logs --tail=60 sim')
            return 2
        a, b = node.lower.pose.position, node.upper.pose.position
        dx, dy, dz = b.x-a.x, b.y-a.y, b.z-a.z
        print('\nWorld coordinates (meters):')
        print(f'  lower_box: x={a.x:+.4f} y={a.y:+.4f} z={a.z:+.4f}')
        print(f'  upper_box: x={b.x:+.4f} y={b.y:+.4f} z={b.z:+.4f}')
        print('\nUpper relative to lower (meters):')
        print(f'  dx={dx:+.4f}, dy={dy:+.4f}, dz={dz:+.4f}')
        print(f'  Expected dz when stacked upright: {EXPECTED_DZ:.4f}')
        print(f'  Vertical error: {dz-EXPECTED_DZ:+.4f}')
        print(f'  Full XY footprint margin per axis: '
              f'{MAX_CENTER_OFFSET_FOR_FULL_SUPPORT:.4f}')
        if (abs(dx) <= MAX_CENTER_OFFSET_FOR_FULL_SUPPORT and
                abs(dy) <= MAX_CENTER_OFFSET_FOR_FULL_SUPPORT and
                abs(dz-EXPECTED_DZ) <= 0.015):
            print('GEOMETRY: approximately consistent with centered stack.')
            print('NOT PROVEN: stability, contact, uprightness or detached state.')
        else:
            print('GEOMETRY: not consistent with a centered upright stack.')
            if abs(dx) > MAX_CENTER_OFFSET_FOR_FULL_SUPPORT:
                print('  - X misalignment is too large.')
            if abs(dy) > MAX_CENTER_OFFSET_FOR_FULL_SUPPORT:
                print('  - Y misalignment is too large.')
            if abs(dz-EXPECTED_DZ) > 0.015:
                print('  - Relative height differs from expected stacking height.')
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
