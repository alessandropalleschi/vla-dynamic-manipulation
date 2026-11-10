#!/usr/bin/python3
"""Quick end-to-end ROS data-flow check. Doesn't certify physical stack success."""
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, JointState


class Smoke(Node):
    def __init__(self):
        super().__init__('conveyor_smoke_test')
        self.counts = {'/clock': 0, '/camera/image_raw': 0, '/joint_states': 0}
        self.create_subscription(Clock, '/clock', lambda _: self.mark('/clock'), 10)
        self.create_subscription(Image, '/camera/image_raw', lambda _: self.mark('/camera/image_raw'), qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', lambda _: self.mark('/joint_states'), qos_profile_sensor_data)

    def mark(self, topic):
        self.counts[topic] += 1


def main():
    rclpy.init()
    node = Smoke()
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline and min(node.counts.values()) == 0:
        rclpy.spin_once(node, timeout_sec=0.2)
    print('Counts:', node.counts)
    ok = all(value > 0 for value in node.counts.values())
    print('PASS: camera, clock and gantry joint states detected' if ok else 'FAIL: missing ROS topics; inspect sim logs and gz topics')
    node.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
