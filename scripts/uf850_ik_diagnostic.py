#!/usr/bin/env python3

import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

JOINTS = [f"joint{i}" for i in range(1, 7)]

class IKDiagnostic(Node):
    def __init__(self):
        super().__init__("uf850_ik_diagnostic")
        self.joints = None

        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_callback,
            10
        )

    def joint_callback(self, msg):
        positions = dict(zip(msg.name, msg.position))

        if all(j in positions for j in JOINTS):
            self.joints = np.array(
                [positions[j] for j in JOINTS]
            )

def main():
    rclpy.init()
    node = IKDiagnostic()

    deadline = time.monotonic() + 10

    while node.joints is None and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)

    if node.joints is None:
        print("ERROR: UF850 joint states unavailable")
    else:
        print("\nUF850 current joint configuration:")
        for name, position in zip(JOINTS, node.joints):
            print(f"  {name}: {position:.4f} rad")

        print("\nCurrent configuration acquired successfully.")
        print("Next: solve for the VGC10 pickup pose.")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
