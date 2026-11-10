#!/usr/bin/env python3

import re
import subprocess
import time
import math

POSE_TOPIC = "/simulation/lower_box/pose"
SPEED_TOPIC = "/conveyor/speed"


def command_speed(speed):
    subprocess.run(
        [
            "gz", "topic",
            "-t", SPEED_TOPIC,
            "-m", "gz.msgs.Double",
            "-p", f"data: {speed}"
        ],
        check=True,
        timeout=10
    )


def read_box_pose():
    command = [
        "stdbuf", "-oL",
        "gz", "topic",
        "-e", "-t", POSE_TOPIC
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=1.5
        )
        output = result.stdout
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or b""

    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")

    match = re.search(
        r"\bposition\s*\{([^{}]*)\}",
        output,
        re.DOTALL
    )

    if not match:
        raise RuntimeError(
            "No orange-box position received. "
            "Check Gazebo and the pose topic."
        )

    body = match.group(1)

    xyz = {}

    for axis in ("x", "y", "z"):
        value = re.search(
            rf"\b{axis}\s*:\s*"
            r"([-+]?(?:\d+\.?\d*|\.\d+)"
            r"(?:[eE][-+]?\d+)?)",
            body
        )

        # Protobuf text omits coordinates equal to zero.
        xyz[axis] = (
            float(value.group(1))
            if value else 0.0
        )

    return xyz


def main():

    # Begin from a stopped conveyor.
    command_speed(0.0)

    before = read_box_pose()

    print("Initial orange-box pose:", before)

    speed = 0.02
    duration = 2.0

    print(
        f"\nCommanding conveyor at "
        f"{speed:.3f} m/s..."
    )

    start = None
    end = None

    try:
        command_speed(speed)
        start = time.monotonic()

        time.sleep(duration)

    finally:
        # Stop even if the test is interrupted.
        command_speed(0.0)
        end = time.monotonic()

    after = read_box_pose()

    dt = end - start

    dx = after["x"] - before["x"]
    dy = after["y"] - before["y"]

    distance = math.hypot(dx, dy)

    print("\nFinal orange-box pose:", after)

    print("\n=== Conveyor motion measurement ===")

    print(f"Wall time: {dt:.3f} s")
    print(f"Delta X: {dx:+.5f} m")
    print(f"Delta Y: {dy:+.5f} m")

    print(f"Horizontal displacement: {distance:.5f} m")

    print(
        "Approximate speed per wall second: "
        f"{distance / dt:.5f} m/s"
    )

    print("\nPASS: Conveyor stop command sent")

    if distance < 0.002:
        print(
            "WARNING: Orange box barely moved. "
            "Check conveyor actuation."
        )
    else:
        print(
            "PASS: Orange-box movement detected"
        )

    print(
        "\nNOTE: This is a wall-time estimate. "
        "We will use Gazebo simulation timestamps "
        "for precise interception."
    )


if __name__ == "__main__":
    main()
