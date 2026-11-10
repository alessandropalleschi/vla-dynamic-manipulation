#!/usr/bin/env python3
"""Test constant-velocity predictions using Gazebo simulation time."""

import argparse
import bisect
import json
import math
import os
import statistics
import re
import subprocess
import time
from pathlib import Path

POSE = "/simulation/lower_box/pose"
SPEED = "/conveyor/speed"


def command_speed(value):
    subprocess.run(
        [
            "gz", "topic",
            "-t", SPEED,
            "-m", "gz.msgs.Double",
            "-p", f"data: {value:.6f}"
        ],
        check=True,
        timeout=8,
        stdout=subprocess.DEVNULL
    )


def linear_velocity(points):
    mean_t = sum(p[0] for p in points) / len(points)

    denominator = sum(
        (p[0] - mean_t) ** 2
        for p in points
    )

    if denominator < 1e-5:
        raise RuntimeError("Insufficient simulation-time spread")

    velocities = []

    for axis in (1, 2):
        mean_p = sum(
            p[axis] for p in points
        ) / len(points)

        velocity = sum(
            (p[0] - mean_t) * (p[axis] - mean_p)
            for p in points
        ) / denominator

        velocities.append(velocity)

    return velocities


def evaluate(samples, start, end):

    samples = sorted(samples)

    clean = []

    for sample in samples:
        if not clean or sample[0] > clean[-1][0] + 1e-5:
            clean.append(sample)

    times = [p[0] for p in clean]

    if len(clean) < 25:
        raise RuntimeError("Insufficient pose samples")

    horizons = (0.5, 1.0)

    errors = {h: [] for h in horizons}
    velocities = []

    last_prediction = -1e9

    for sample in clean:

        t = sample[0]

        if t < start + 1.5:
            continue

        if t > end - max(horizons) - 0.15:
            continue

        if t - last_prediction < 0.2:
            continue

        # Estimate velocity using the preceding second.
        window = [
            p for p in clean
            if t - 1.0 <= p[0] <= t
        ]

        if len(window) < 6:
            continue

        if window[-1][0] - window[0][0] < 0.6:
            continue

        vx, vy = linear_velocity(window)

        velocities.append((vx, vy))

        last_prediction = t

        for horizon in horizons:

            future_t = t + horizon

            index = bisect.bisect_left(
                times, future_t
            )

            if not 0 < index < len(clean):
                continue

            before = clean[index - 1]
            after = clean[index]

            dt = after[0] - before[0]

            if dt > 0.2:
                continue

            fraction = (future_t - before[0]) / dt

            actual_x = (
                before[1]
                + fraction * (after[1] - before[1])
            )

            actual_y = (
                before[2]
                + fraction * (after[2] - before[2])
            )

            predicted_x = sample[1] + vx * horizon
            predicted_y = sample[2] + vy * horizon

            error = math.hypot(
                predicted_x - actual_x,
                predicted_y - actual_y
            )

            errors[horizon].append(error)

    if not velocities:
        raise RuntimeError(
            "No valid velocity estimates"
        )

    results = {}

    for horizon, values in errors.items():

        if len(values) < 3:
            raise RuntimeError(
                f"Insufficient predictions for {horizon}s"
            )

        values.sort()

        p95_index = min(
            len(values) - 1,
            math.ceil(0.95 * len(values)) - 1
        )

        results[str(horizon)] = {
            "predictions": len(values),
            "mean_error_mm": round(
                1000 * statistics.mean(values), 3
            ),
            "p95_error_mm": round(
                1000 * values[p95_index], 3
            ),
            "max_error_mm": round(
                1000 * max(values), 3
            )
        }

    return {
        "simulation_duration_s": round(end - start, 3),
        "samples": len(clean),
        "velocity_x_m_s": round(
            statistics.median(v[0] for v in velocities), 6
        ),
        "velocity_y_m_s": round(
            statistics.median(v[1] for v in velocities), 6
        ),
        "prediction": results
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--speed", type=float, default=0.02
    )

    parser.add_argument(
        "--duration", type=float, default=6.0
    )

    args = parser.parse_args()

    if not 0 < args.speed <= 0.04:
        parser.error("Speed must be 0–0.04 m/s")

    if not 5 <= args.duration <= 12:
        parser.error("Duration must be 5–12 seconds")

    if os.environ.get("ROS_DOMAIN_ID") != "42":
        parser.error("Expected isolated ROS_DOMAIN_ID=42")

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rosgraph_msgs.msg import Clock
    from geometry_msgs.msg import Pose
    from std_msgs.msg import Bool, String

    rclpy.init()

    node = Node("conveyor_prediction_validation")

    state = {
        "time": None,
        "pose": None,
        "attached": None,
        "diag": None,
        "samples": []
    }

    def clock_callback(msg):
        state["time"] = (
            msg.clock.sec
            + msg.clock.nanosec * 1e-9
        )

    def pose_callback(msg):

        if state["time"] is None:
            return

        p = msg.position

        sample = (
            state["time"],
            float(p.x),
            float(p.y),
            float(p.z)
        )

        state["pose"] = sample
        state["samples"].append(sample)

    def attached_callback(msg):
        state["attached"] = bool(msg.data)

    def diagnostics_callback(msg):
        state["diag"] = str(msg.data)

    node.create_subscription(
        Clock, "/clock", clock_callback, 20
    )
    # PATCH_V2_WORLD_CLOCK
    node.create_subscription(
        Clock,
        "/world/conveyor_station/clock",
        clock_callback,
        qos_profile_sensor_data
    )

    node.create_subscription(
        Pose,
        POSE,
        pose_callback,
        qos_profile_sensor_data
    )

    node.create_subscription(
        Bool,
        "/vgc10/attached",
        attached_callback,
        10
    )

    node.create_subscription(
        String,
        "/vgc10/diagnostics",
        diagnostics_callback,
        10
    )

    bridge = None

    try:
        command_speed(0.0)

        # Bridge orange-box pose if necessary.
        if (
            node.count_publishers(POSE) == 0
            or node.count_publishers(
                "/world/conveyor_station/clock"
            ) == 0
        ):

            bridge = subprocess.Popen(
                [
                    "ros2", "run",
                    "ros_gz_bridge",
                    "parameter_bridge",
                    POSE
                    + "@geometry_msgs/msg/Pose[gz.msgs.Pose",
                    "/world/conveyor_station/clock"
                    "@rosgraph_msgs/msg/Clock[gz.msgs.Clock"
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

        deadline = time.monotonic() + 20

        while time.monotonic() < deadline:

            rclpy.spin_once(
                node, timeout_sec=0.08
            )

            if bridge and bridge.poll() is not None:
                raise RuntimeError("Pose bridge exited")

            if (
                state["time"] is not None
                and state["pose"] is not None
                and (
                    state["attached"] is not None
                    or state["diag"] is not None
                )
            ):
                break
        else:
            missing = [
                name for name, value in (
                    ("simulation clock", state["time"]),
                    ("orange-box ROS pose", state["pose"]),
                    ("VGC10 status",
                     state["attached"] if state["attached"] is not None
                     else state["diag"]),
                ) if value is None
            ]
            raise RuntimeError(
                "Missing ROS input(s): " + ", ".join(missing)
            )

        if state["attached"] is True:
            raise RuntimeError(
                "Green box attached; refusing belt motion"
            )

        if state["diag"] is not None:
            diag = state["diag"]
            for channel in ("a", "b"):
                pattern = rf"(?:^|[\s,]){channel}\s*=\s*0(?:$|[\s,])"
                if not re.search(pattern, diag):
                    raise RuntimeError(
                        "Vacuum channel is not confirmed OFF: " + diag
                    )

        initial = state["pose"]

        if not (
            -0.54 < initial[1] < -0.50
            and abs(initial[2]) < 0.015
        ):
            raise RuntimeError(
                "Orange box is not at its initial position"
            )

        print("Initial orange-box pose:", initial[1:])

        # Verify clock is advancing.
        initial_time = state["time"]

        deadline = time.monotonic() + 8

        while state["time"] < initial_time + 0.2:

            if time.monotonic() > deadline:
                raise RuntimeError(
                    "Simulation clock is not advancing"
                )

            rclpy.spin_once(
                node, timeout_sec=0.05
            )

        command_speed(args.speed)

        start = state["time"]

        print(
            f"Conveyor running at {args.speed:.3f} m/s"
        )

        deadline = time.monotonic() + max(
            30, args.duration * 5
        )

        while state["time"] < start + args.duration:

            if time.monotonic() > deadline:
                raise RuntimeError(
                    "Timed out during conveyor trial"
                )

            rclpy.spin_once(
                node, timeout_sec=0.05
            )

        end = state["time"]

        command_speed(0.0)

        print("Conveyor stop command sent")

        report = evaluate(
            state["samples"], start, end
        )

        report["commanded_speed_m_s"] = args.speed

        output = Path(
            "/workspace/data/conveyor_prediction_validation.json"
        )

        output.parent.mkdir(
            parents=True, exist_ok=True
        )

        output.write_text(
            json.dumps(report, indent=2) + "\n"
        )

        print("\n=== PREDICTION RESULTS ===")
        print(json.dumps(report, indent=2))

        print("\nSaved:", output)
        print("PASS: No robot commands sent")

    finally:

        try:
            command_speed(0.0)
            print("SAFETY: Conveyor stop command sent")
        except Exception as exc:
            print("WARNING: Could not stop belt:", exc)

        if bridge is not None:

            bridge.terminate()

            try:
                bridge.wait(timeout=3)
            except subprocess.TimeoutExpired:
                bridge.kill()

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
