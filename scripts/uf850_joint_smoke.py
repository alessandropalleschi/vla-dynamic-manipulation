#!/usr/bin/env python3
"""Simulation-only six-joint motion test. Never connects to a physical robot."""
import time
import sys
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

ARM = [f'joint{i}' for i in range(1, 7)]


class Tester(Node):
    def __init__(self):
        super().__init__('uf850_sim_smoke')
        self.positions = None
        self.sub = self.create_subscription(JointState, '/joint_states', self.on_js, 10)
        self.client = ActionClient(self, FollowJointTrajectory,
                                   '/uf850_arm_controller/follow_joint_trajectory')

    def on_js(self, msg):
        states = dict(zip(msg.name, msg.position))
        if all(j in states for j in ARM):
            self.positions = [float(states[j]) for j in ARM]

    def spin_until(self, predicate, duration):
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if predicate(): return True
        return False


def main():
    rclpy.init()
    node = Tester()
    try:
        if not node.spin_until(lambda: node.positions is not None, 25):
            raise RuntimeError('No UF850 six-joint /joint_states. Check Gazebo controller startup.')
        print('Initial UF850 joints:', list(round(x, 4) for x in node.positions))
        if not node.client.wait_for_server(timeout_sec=20):
            raise RuntimeError('Missing /uf850_arm_controller/follow_joint_trajectory action')
        initial = list(node.positions)
        target = list(initial)
        target[0] += 0.08  # small joint1 rotation, simulation only
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ARM
        pt = JointTrajectoryPoint()
        pt.positions = target
        pt.time_from_start = Duration(sec=4, nanosec=0)
        goal.trajectory.points = [pt]
        fut = node.client.send_goal_async(goal)
        if not node.spin_until(fut.done, 8):
            raise RuntimeError('Trajectory goal not accepted in time')
        handle = fut.result()
        if not handle or not handle.accepted:
            raise RuntimeError('Trajectory goal rejected')
        result = handle.get_result_async()
        if not node.spin_until(result.done, 12):
            raise RuntimeError('Trajectory action timeout')
        if result.result().result.error_code != 0:
            raise RuntimeError(f'Trajectory action error code {result.result().result.error_code}')
        if not node.spin_until(lambda: abs(node.positions[0] - target[0]) < 0.03, 5):
            raise RuntimeError('Joint1 did not reach the target')
        print('PASS: six-joint UF850 feedback and commanded +0.08 rad joint1 rotation')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        print('FAIL:', exc, file=sys.stderr)
        sys.exit(1)
