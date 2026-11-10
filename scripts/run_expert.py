#!/usr/bin/python3
"""Privileged Gazebo-only stacking expert for the 3-axis demo gantry.

NOT an end-to-end VLA and NOT a robot-safe real-world controller.
The expert uses two Gazebo model pose topics and actual gantry joint states.
It lifts before moving laterally, aligns above the target, then lowers toward a
world-space contact height, and NEVER releases unless pose gates are met.
"""
import argparse
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float64, String

# Configured in simulation/worlds/conveyor_station.sdf
GANTRY_ORIGIN_X = -0.55
GANTRY_ORIGIN_Y = -0.55
LOWER_HEIGHT_M = 0.08
UPPER_HEIGHT_M = 0.06
GANTRY_Z_MAX_M = 0.45
TRAVEL_Z_M = 0.28
# The upper object is released just ABOVE the lower top surface; it falls a few mm.
RELEASE_CLEARANCE_M = 0.006
ALIGN_TOL_M = 0.009
# Must be above the lower box before crossing it laterally.
MIN_TRAVEL_CLEARANCE_M = 0.06
PHASE_TIMEOUT_S = {
    'lift': 6.0, 'align_above_target': 8.0,
    'descend_with_feedback': 8.0, 'release': 2.0, 'retract': 4.0,
}


def clamp(v, low, high):
    return max(low, min(high, v))


def target_z_center(lower_z, clearance=RELEASE_CLEARANCE_M):
    """Desired upper center in world coordinates, not a joint setpoint."""
    return lower_z + (LOWER_HEIGHT_M + UPPER_HEIGHT_M) / 2.0 + clearance


def box_bottom(z):
    return z - UPPER_HEIGHT_M / 2.0


def lower_top(z):
    return z + LOWER_HEIGHT_M / 2.0


class GroundTruthStackExpert(Node):
    def __init__(self, speed, keep_belt_running=False):
        super().__init__('ground_truth_stack_expert')
        # ROS time source may have already declared this parameter.
        # Redeclaring it raises ParameterAlreadyDeclaredException on Jazzy.
        if self.has_parameter('use_sim_time'):
            self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        else:
            self.declare_parameter('use_sim_time', True)
        self.speed = speed
        self.keep_belt_running = keep_belt_running
        self.pub = {
            axis: self.create_publisher(Float64, f'/gantry/{axis}_cmd', 10)
            for axis in ('x', 'y', 'z')
        }
        self.belt = self.create_publisher(Float64, '/conveyor/speed', 10)
        self.detach = self.create_publisher(Empty, '/gripper/detach', 10)
        self.phase_pub = self.create_publisher(String, '/expert/phase', 10)
        self.lower = None
        self.upper = None
        self.joints = {}
        self.attach_state = 'unknown'
        self.last_wall = {}
        self.create_subscription(PoseStamped, '/simulation/lower_box/pose',
                                 lambda msg: self.set_data('lower', msg), 10)
        self.create_subscription(PoseStamped, '/simulation/upper_box/pose',
                                 lambda msg: self.set_data('upper', msg), 10)
        self.create_subscription(JointState, '/joint_states', self.on_joints, 10)
        self.create_subscription(String, '/gripper/attached_state', self.on_gripper, 10)
        self.phase = 'awaiting_ground_truth'
        self.phase_started = None
        self.started_wall = time.monotonic()
        self.running_since = None
        self.ready_ticks = 0
        self.last_log_t = -10.0
        self.finished = False
        self.succeeded = False
        self.timer = self.create_timer(0.05, self.tick)

    def set_data(self, kind, msg):
        setattr(self, kind, msg)
        self.last_wall[kind] = time.monotonic()

    def on_joints(self, msg):
        # Strip optional namespacing; expect x_joint/y_joint/z_joint in gantry world.
        self.joints = {name.split('::')[-1].split('/')[-1]: float(position)
                       for name, position in zip(msg.name, msg.position)}
        self.last_wall['joints'] = time.monotonic()

    def on_gripper(self, msg):
        self.attach_state = msg.data

    @staticmethod
    def send(publisher, value):
        publisher.publish(Float64(data=float(value)))

    def command(self, x, y, z, belt_speed):
        self.send(self.pub['x'], clamp(x, -0.10, 1.20))
        self.send(self.pub['y'], clamp(y, -0.05, 0.65))
        self.send(self.pub['z'], clamp(z, 0.0, GANTRY_Z_MAX_M))
        self.send(self.belt, belt_speed)

    def transition(self, next_phase, now, note=''):
        self.phase = next_phase
        self.phase_started = now
        self.ready_ticks = 0
        self.get_logger().info(f'Phase: {next_phase} (sim t={now:.2f}) {note}')

    def abort(self, now, why, x=0, y=0, z=TRAVEL_Z_M):
        self.get_logger().error(f'ABORT: {why}. No release commanded.')
        self.transition('aborted', now, why)
        # Do not detach. Hold above the region and stop conveyor in simulation.
        self.command(x, y, z, 0.0)
        self.finished = True

    def tick(self):
        now = self.get_clock().now().nanoseconds / 1e9
        if now <= 0:
            return
        has_data = (self.lower is not None and self.upper is not None
                    and all(k in self.joints for k in ('x_joint', 'y_joint', 'z_joint')))
        if not has_data:
            self.send(self.belt, 0.0)
            if time.monotonic() - self.started_wall > 15.0:
                self.abort(now, 'Ground-truth pose or joint topics missing. Check /simulation/*/pose and /joint_states')
            return
        if any(time.monotonic() - self.last_wall.get(k, -1e10) > 2.0
               for k in ('lower', 'upper', 'joints')):
            self.abort(now, 'Stale pose or joint-state data')
            return

        lp = self.lower.pose.position
        up = self.upper.pose.position
        qz = self.joints['z_joint']
        # In this world: joint positions are relative to gantry origin (-.55, -.55).
        x_cmd = lp.x - GANTRY_ORIGIN_X
        y_cmd = lp.y - GANTRY_ORIGIN_Y
        dx = up.x - lp.x
        dy = up.y - lp.y
        xy_error = math.hypot(dx, dy)
        gap = box_bottom(up.z) - lower_top(lp.z)  # positive means safely above target

        if self.phase == 'awaiting_ground_truth':
            self.running_since = now
            self.transition('lift', now, 'Received actual object poses and gantry state')

        elapsed = now - self.phase_started
        if self.phase in PHASE_TIMEOUT_S and elapsed > PHASE_TIMEOUT_S[self.phase]:
            self.abort(now, f'Timed out in {self.phase}; XY error {xy_error:.3f}m, gap {gap:.3f}m',
                       x_cmd, y_cmd, max(qz, TRAVEL_Z_M))
            return

        if self.phase == 'lift':
            # DO NOT move horizontally until the upper box is above the lower top.
            self.command(0, 0, TRAVEL_Z_M, 0.0)
            if qz > TRAVEL_Z_M - 0.02 and gap > MIN_TRAVEL_CLEARANCE_M:
                self.transition('align_above_target', now)

        elif self.phase == 'align_above_target':
            # Start belt only after lifting; use the ACTUAL moving box pose to track.
            self.command(x_cmd, y_cmd, TRAVEL_Z_M, self.speed)
            ok = xy_error < ALIGN_TOL_M and gap > MIN_TRAVEL_CLEARANCE_M
            self.ready_ticks = self.ready_ticks + 1 if ok else 0
            if self.ready_ticks >= 6:
                self.transition('descend_with_feedback', now,
                                f'XY aligned; x err={dx:.3f}, y err={dy:.3f}')

        elif self.phase == 'descend_with_feedback':
            # Recompute world-space Z target from the lower box pose on every tick.
            # Converting target center into a relative gantry joint setpoint uses
            # observed upper center and observed qz, not the assumed initial offset.
            if xy_error > 0.025:
                self.abort(now, f'Lost lateral alignment during descent: {xy_error:.3f} m',
                           x_cmd, y_cmd, qz)
                return
            desired_upper_z = target_z_center(lp.z)
            z_error = desired_upper_z - up.z
            # Limit vertical movement to ~0.06 m/s near the target, to avoid
            # overshooting through contact. Lift back slightly on positive error.
            z_step = clamp(z_error, -0.003, 0.003)
            z_cmd = qz + z_step
            self.command(x_cmd, y_cmd, z_cmd, self.speed)
            if now - self.last_log_t >= 0.5:
                self.last_log_t = now
                self.get_logger().info(
                    f'Placement: xy_err={xy_error:.4f}m, gap={gap:.4f}m, '
                    f'upper_z={up.z:.4f}, desired_z={desired_upper_z:.4f}, qz={qz:.4f}')
            # Require several consecutive valid observations before release.
            ok = (xy_error < ALIGN_TOL_M and 0.002 <= gap <= 0.010
                  and abs(z_error) <= 0.004)
            self.ready_ticks = self.ready_ticks + 1 if ok else 0
            if gap < -0.002:
                self.abort(now, f'Upper box penetrated target surface: gap={gap:.4f}m',
                           x_cmd, y_cmd, qz)
                return
            if self.ready_ticks >= 6:
                self.transition('release', now,
                                f'Safe release gate: xy_err={xy_error:.4f}m gap={gap:.4f}m')

        elif self.phase == 'release':
            self.command(x_cmd, y_cmd, qz, self.speed)
            # Repeated detach commands are harmless and reduce transient message loss.
            if self.attach_state != 'detached':
                self.detach.publish(Empty())
            if self.attach_state == 'detached' and elapsed > 0.25:
                self.transition('retract', now, 'Confirmed detach via /gripper/attached_state')

        elif self.phase == 'retract':
            self.command(x_cmd, y_cmd, TRAVEL_Z_M, self.speed)
            if qz > TRAVEL_Z_M - 0.035 and elapsed > 0.8:
                self.transition('complete', now)

        elif self.phase == 'complete':
            self.command(x_cmd, y_cmd, TRAVEL_Z_M, self.speed if self.keep_belt_running else 0.0)
            if elapsed > 0.4:
                self.get_logger().info(
                    f'Sequence completed. Final pose XY error={xy_error:.4f}m, '
                    f'center height difference={up.z-lp.z:.4f}m. '
                    'Check stable support separately; this is NOT a physics success certificate.')
                self.succeeded = True
                self.finished = True

        self.phase_pub.publish(String(data=self.phase))


def main():
    parser = argparse.ArgumentParser(description='Ground-truth feedback stacking (simulation only)')
    parser.add_argument('--speed', type=float, default=0.0, help='Conveyor speed 0.0..0.10 m/s')
    parser.add_argument('--keep-belt-running', action='store_true',
                        help='Leave conveyor running after expert completes; recorder stops it after evaluation window')
    args = parser.parse_args()
    if not math.isfinite(args.speed) or not 0.0 <= args.speed <= 0.10:
        parser.error('--speed must be between 0.0 and 0.10')
    rclpy.init()
    node = GroundTruthStackExpert(args.speed, args.keep_belt_running)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        node.send(node.belt, 0.0)
    finally:
        if not (node.succeeded and args.keep_belt_running):
            node.send(node.belt, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if node.succeeded else 2


if __name__ == '__main__':
    raise SystemExit(main())
