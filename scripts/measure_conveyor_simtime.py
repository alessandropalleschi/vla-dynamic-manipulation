#!/usr/bin/env python3
"""Measure orange-box velocity using Gazebo simulation time."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

POSE_TOPIC = "/simulation/lower_box/pose"
SPEED_TOPIC = "/conveyor/speed"


def command_speed(speed):
    subprocess.run(
        [
            "gz", "topic",
            "-t", SPEED_TOPIC,
            "-m", "gz.msgs.Double",
            "-p", f"data: {speed:.6f}",
        ],
        check=True,
        timeout=8,
    )


def fit_velocity(samples):
    """Fit X/Y velocity against simulation time."""
    if len(samples) < 8:
        raise RuntimeError(
            f"Insufficient position samples: {len(samples)}"
        )

    times = [s[0] for s in samples]
    mean_t = sum(times) / len(times)

    denominator = sum(
        (t - mean_t) ** 2 for t in times
    )

    if denominator < 0.15:
        raise RuntimeError(
            "Insufficient simulation-time spread"
        )

    velocity = []

    for axis in (1, 2):
        mean_p = (
            sum(s[axis] for s in samples)
            / len(samples)
        )

        velocity.append(
            sum(
                (s[0] - mean_t) *
                (s[axis] - mean_p)
                for s in samples
            ) / denominator
        )

    return velocity


def run(args):
    if os.environ.get("ROS_DOMAIN_ID") != "42":
        raise RuntimeError(
            "Expected isolated simulation ROS_DOMAIN_ID=42"
        )

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rosgraph_msgs.msg import Clock
    from geometry_msgs.msg import Pose
    from std_msgs.msg import Bool

    rclpy.init()

    node = Node("uf850_conveyor_velocity_measurement")

    state = {
        "clock": None,
        "pose": None,
        "attached": None,
        "samples": [],
    }

    def on_clock(msg):
        state["clock"] = (
            msg.clock.sec +
            msg.clock.nanosec * 1e-9
        )

    def on_pose(msg):
        t = state["clock"]

        if t is None:
            return

        p = msg.position

        sample = (
            t,
            float(p.x),
            float(p.y),
            float(p.z),
        )

        state["pose"] = sample
        state["samples"].append(sample)

    def on_attached(msg):
        state["attached"] = bool(msg.data)

    node.create_subscription(
        Clock, "/clock", on_clock, 20
    )

    node.create_subscription(
        Pose,
        POSE_TOPIC,
        on_pose,
        qos_profile_sensor_data,
    )

    node.create_subscription(
        Bool, "/vgc10/attached", on_attached, 10
    )

    bridge = None

    try:
        # Start a temporary pose bridge when necessary.
        if node.count_publishers(POSE_TOPIC) == 0:
            bridge = subprocess.Popen(
                [
                    "ros2", "run",
                    "ros_gz_bridge",
                    "parameter_bridge",
                    POSE_TOPIC +
                    "@geometry_msgs/msg/Pose[gz.msgs.Pose",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            print("Started temporary orange-box pose bridge")

        command_speed(0.0)

        deadline = time.monotonic() + 15

        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.08)

            if bridge and bridge.poll() is not None:
                raise RuntimeError(
                    "ROS–Gazebo pose bridge exited"
                )

            if all(
                state[k] is not None
                for k in ("clock", "pose", "attached")
            ):
                break
        else:
            raise RuntimeError(
                "Missing clock, box pose, or vacuum status"
            )

        if state["attached"]:
            raise RuntimeError(
                "Green box is attached; refusing belt motion"
            )

        initial = state["pose"]

        # Require the box to be near its reset position.
        if not (
            -0.54 < initial[1] < -0.50
            and abs(initial[2]) < 0.015
            and abs(initial[3] - 0.83) < 0.015
        ):
            raise RuntimeError(
                "Orange box is not at its reset pose: "
                f"{initial[1:]}"
            )

        print("Initial orange-box XYZ:", initial[1:])

        # Confirm simulation time advances.
        first_clock = state["clock"]
        deadline = time.monotonic() + 5

        while state["clock"] < first_clock + 0.1:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "Gazebo clock is not advancing"
                )

            rclpy.spin_once(node, timeout_sec=0.05)

        start_sim = state["clock"]

        print(
            f"Running belt at {args.speed:.3f} m/s "
            f"for {args.duration:.1f} simulation seconds"
        )

        command_speed(args.speed)

        deadline = time.monotonic() + 20

        while state["clock"] < start_sim + args.duration:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "Timed out waiting for simulation time"
                )

            rclpy.spin_once(node, timeout_sec=0.05)

        stop_sim = state["clock"]

        command_speed(0.0)

        for _ in range(8):
            rclpy.spin_once(node, timeout_sec=0.05)

        # Exclude startup and stop transients.
        usable = [
            s for s in state["samples"]
            if start_sim + 0.25 <= s[0] <= stop_sim - 0.10
        ]

        vx, vy = fit_velocity(usable)

        endpoint = min(
            state["samples"],
            key=lambda s: abs(s[0] - stop_sim),
        )

        print("\n=== SIMULATION-TIME VELOCITY ===")

        print(
            "Elapsed simulation time:",
            round(stop_sim - start_sim, 4),
            "s",
        )

        print("Samples:", len(usable))

        print(
            "Displacement X:",
            round(endpoint[1] - initial[1], 5),
            "m",
        )

        print("Velocity X:", round(vx, 5), "m/s")
        print("Velocity Y:", round(vy, 5), "m/s")

        # This prediction assumes the belt keeps moving.
        prediction_x = endpoint[1] + vx * 5.0

        print(
            "Predicted X after another 5 simulated seconds "
            "of motion:",
            round(prediction_x, 4),
            "m",
        )

        result = {
            "commanded_speed_m_s": args.speed,
            "sim_time_elapsed_s": stop_sim - start_sim,
            "velocity_x_m_s": vx,
            "velocity_y_m_s": vy,
            "position_at_stop_xyz_m": list(endpoint[1:]),
            "number_of_samples": len(usable),
            "belt_stopped": True,
        }

        output = Path(
            "/workspace/data/conveyor_velocity_simtime.json"
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.write_text(
            json.dumps(result, indent=2) + "\n"
        )

        print("\nSaved:", output)

        if abs(vx - args.speed) < 0.008 and abs(vy) < 0.005:
            print("PASS: Conveyor velocity consistent with command")
        else:
            print("WARNING: Measured velocity differs from command")

    finally:
        # Always attempt to stop the belt.
        try:
            command_speed(0.0)
            print("SAFETY: Conveyor stop command sent")
        except Exception as exc:
            print("WARNING: Could not confirm stop command:", exc)

        if bridge is not None:
            bridge.terminate()

            try:
                bridge.wait(timeout=3)
            except subprocess.TimeoutExpired:
                bridge.kill()

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--speed",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
    )

    args = parser.parse_args()

    if not (0 < args.speed <= 0.04):
        parser.error("Speed must be between 0 and 0.04 m/s")

    try:
        run(args)
    except (RuntimeError, OSError, KeyboardInterrupt) as exc:
        raise SystemExit(f"BLOCKED: {exc}")
