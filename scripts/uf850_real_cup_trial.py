#!/usr/bin/env python3
"""
UF850 + real VGC10 suction cups.

PLAN-HOVER:
    Fresh 6D IK and collision-aware MoveIt planning.
    No robot movement or vacuum commands.

MOVE-HOVER:
    Replan from live joint states, then execute in Gazebo.
    Vacuum remains OFF.

Uses the existing v7 RobotKinematics and SimStation classes.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from uf850_stationary_stack_v7 import (
    Blocked,
    RobotKinematics,
    SimStation,
    check,
    rot_rpy,
    rotvec,
    verify_pose_snapshot,
    world_objects,
)

from uf850_moveit_common import URDF_PATH, WORLD_PATH


TOOL_OFFSET = 0.189395921
BODY_LIP_Z = -0.174395921

PITCH = 0.044
RADIUS = 0.0202

CUPS = {
    "a1": (-1,  1),
    "a2": ( 1,  1),
    "b1": (-1, -1),
    "b2": ( 1, -1),
}


def validate_urdf(path):
    robot = ET.parse(path).getroot()

    tcp = robot.find(
        "./joint[@name='vgc10_tcp_fixed']/origin"
    )

    mount = robot.find(
        "./joint[@name='uf850_to_vgc10']/origin"
    )

    check(
        tcp is not None and mount is not None,
        "Missing VGC10 mounting or TCP joint"
    )

    tcp_z = float(tcp.get("xyz").split()[2])
    roll = float(mount.get("rpy").split()[0])

    check(
        abs(tcp_z - BODY_LIP_Z) < 0.0005,
        "URDF still contains the old TCP position"
    )

    check(
        abs(roll - math.pi) < 0.001,
        "VGC10 mounting orientation is incorrect"
    )

    body = robot.find("./link[@name='vgc10_body']")

    check(body is not None, "Missing VGC10 body")

    for name, (sx, sy) in CUPS.items():

        joint = robot.find(
            f"./joint[@name='vgc10_cup_{name}_fixed']/origin"
        )

        cup = robot.find(
            f"./link[@name='vgc10_cup_{name}']"
        )

        visuals = [
            v for v in body.findall("visual")
            if v.get("name") == f"vgc10_body_cup_{name}"
        ]

        check(
            joint is not None
            and cup is not None
            and len(visuals) == 1,
            f"Missing or duplicated cup {name}"
        )

        x, y, z = map(
            float,
            joint.get("xyz").split()
        )

        check(
            abs(x - sx * PITCH / 2) < 0.0001
            and abs(y - sy * PITCH / 2) < 0.0001
            and abs(z - (BODY_LIP_Z + 0.012)) < 0.0005,
            f"Incorrect cup position: {name}"
        )

        cylinder = cup.find(
            "collision/geometry/cylinder"
        )

        check(
            cylinder is not None,
            f"Missing collision for {name}"
        )

        check(
            abs(float(cylinder.get("radius")) - RADIUS)
            < 0.0005,
            f"Incorrect cup radius: {name}"
        )

        mesh = visuals[0].find("geometry/mesh")

        check(
            mesh is not None
            and mesh.get("filename", "").endswith(
                "suction_cup_1cup.stl"
            ),
            f"Real STL missing from cup {name}"
        )

        origin = visuals[0].find("origin")

        check(
            origin is not None
            and abs(
                float(origin.get("xyz").split()[2])
                + 0.0901
            ) < 0.001,
            f"Incorrect STL origin for cup {name}"
        )

    print("PASS: Four real suction-cup meshes")
    print("PASS: Updated TCP and cup collision geometry")


def validate_world(path):
    tree = ET.parse(path)

    plugins = [
        p for p in tree.findall(".//plugin")
        if p.find("tool_offset") is not None
        and p.find("cup_pitch_x") is not None
    ]

    check(
        len(plugins) == 1,
        "Expected exactly one VGC10 vacuum plugin"
    )

    plugin = plugins[0]

    offset = np.array([
        float(x)
        for x in plugin.findtext("tool_offset").split()
    ])

    pitch_x = float(
        plugin.findtext("cup_pitch_x")
    )

    pitch_y = float(
        plugin.findtext("cup_pitch_y")
    )

    axis = np.array([
        float(x)
        for x in plugin.findtext("suction_axis").split()
    ])

    check(
        np.linalg.norm(
            offset - [0, 0, TOOL_OFFSET]
        ) < 0.0001,
        "Gazebo vacuum TCP is outdated"
    )

    check(
        abs(pitch_x - PITCH) < 0.0001
        and abs(pitch_y - PITCH) < 0.0001,
        "Gazebo cup spacing is outdated"
    )

    check(
        np.linalg.norm(axis - [0, 0, 1]) < 0.0001,
        "Incorrect suction direction"
    )

    print("PASS: Gazebo vacuum configuration")


def validate_moveit_description(node, urdf_path):
    """
    Ensure MoveIt did not retain the old 127 mm tool.
    """

    import rclpy
    from rcl_interfaces.srv import GetParameters

    client = node.create_client(
        GetParameters,
        "/move_group/get_parameters"
    )

    check(
        client.wait_for_service(timeout_sec=6.0),
        "Cannot query MoveIt robot_description"
    )

    request = GetParameters.Request()
    request.names = ["robot_description"]

    future = client.call_async(request)

    rclpy.spin_until_future_complete(
        node,
        future,
        timeout_sec=8.0
    )

    response = (
        future.result()
        if future.done()
        else None
    )

    check(
        response is not None
        and len(response.values) == 1,
        "MoveIt robot_description unavailable"
    )

    loaded = response.values[0].string_value.strip()
    expected = Path(urdf_path).read_text().strip()

    check(
        loaded == expected,
        "MoveIt loaded a different URDF. "
        "Restart MoveIt with the new gripper description."
    )

    print("PASS: MoveIt uses the updated VGC10 geometry")


def four_cups_fit(T, green):
    """
    Check full cup disks against the 90x90 mm box.

    T is the TCP pose in world coordinates.
    """

    flange_R = (
        T[:3, :3]
        @ rot_rpy([math.pi, 0, 0]).T
    )

    box_xy = np.asarray(
        green["center_world"][:2],
        dtype=float
    )

    half_xy = np.asarray(
        green["size"][:2],
        dtype=float
    ) / 2

    tcp_xy = np.asarray(
        T[:2, 3],
        dtype=float
    )

    for sx, sy in CUPS.values():

        offset = np.array([
            sx * PITCH / 2,
            sy * PITCH / 2,
            0
        ])

        cup_xy = (
            tcp_xy
            + (flange_R @ offset)[:2]
        )

        outside = (
            np.abs(cup_xy - box_xy)
            + RADIUS
            > half_xy + 1e-7
        )

        if np.any(outside):
            return False

    return True


def find_hover_ik(
    model,
    current,
    target,
    green,
    seeds
):
    """
    Search for full-pose IK with axis-aligned cups.

    Unlike the old probe, this constrains all three
    orientation axes, not just downward direction.
    """

    rng = np.random.default_rng(77)

    previous_candidate = np.array([
        -3.000785307,
        -0.4529,
        -0.9352,
         0.1693,
        -0.4516,
        -0.746485307
    ])

    starts = [
        np.asarray(current),
        np.clip(
            previous_candidate,
            model.lo_vec,
            model.hi_vec
        )
    ]

    additional = max(0, seeds // 4 - 2)

    for _ in range(additional):
        starts.append(
            rng.uniform(
                model.lo_vec + 0.002,
                model.hi_vec - 0.002
            )
        )

    candidates = []

    for yaw in (
        0,
        math.pi / 2,
        -math.pi / 2,
        math.pi
    ):

        desired_R = rot_rpy([
            0,
            0,
            yaw
        ])

        found = 0

        for seed in starts:

            try:
                q = model.solve_near(
                    seed,
                    target,
                    desired_R,
                    tol_pos=0.0015,
                    tol_angle=0.017
                )

                T = model.T(q)

                position_error = float(
                    np.linalg.norm(
                        T[:3, 3] - target
                    )
                )

                orientation_error = float(
                    np.linalg.norm(
                        rotvec(
                            desired_R
                            @ T[:3, :3].T
                        )
                    )
                )

                if position_error >= 0.0015:
                    continue

                if orientation_error >= 0.017:
                    continue

                if T[2, 2] < 0.999:
                    continue

                if not four_cups_fit(T, green):
                    continue

                score = (
                    float(np.linalg.norm(q - current))
                    + 10 * position_error
                    + 0.5 * orientation_error
                )

                candidates.append(
                    (
                        score,
                        q.copy(),
                        yaw,
                        position_error,
                        orientation_error
                    )
                )

                found += 1

                if found >= 4:
                    break

            except (
                Blocked,
                ValueError,
                ArithmeticError,
                np.linalg.LinAlgError
            ):
                continue

        print(
            f"Yaw {math.degrees(yaw):+4.0f}°: "
            f"{found} precise IK candidates"
        )

    candidates.sort(
        key=lambda item: item[0]
    )

    check(
        bool(candidates),
        "No full-pose IK solution with four cups "
        "fitting on the green box"
    )

    return candidates


def run(args):

    check(
        os.getenv("ROS_DOMAIN_ID") == "42",
        "Only isolated ROS_DOMAIN_ID=42 is allowed"
    )

    validate_urdf(args.urdf)
    validate_world(args.world)

    model = RobotKinematics(args.urdf)
    world = world_objects(args.world)

    green = world["upper_box"]

    green_top = (
        green["center_world"][2]
        + green["size"][2] / 2
    )

    hover_target = np.array([
        green["center_world"][0],
        green["center_world"][1],
        green_top + 0.12
    ])

    print(
        "Hover target:",
        np.round(
            hover_target,
            4
        ).tolist()
    )

    check(
        min(green["size"][:2])
        > PITCH + 2 * RADIUS,
        "Green box too small for four cup disks"
    )

    if args.mode == "move-hover":

        check(
            args.ack_simulation_only
            and args.ack_conveyor_stopped
            and args.ack_scene_verified,
            "Execution requires all three safety flags"
        )

    verify_pose_snapshot(
        args.pose_snapshot,
        args.world
    )

    station = SimStation(
        model,
        world,
        args.reports
    )

    try:

        station.ready()

        validate_moveit_description(
            station.node,
            args.urdf
        )

        check(
            station.state["attached"] is False,
            "Green box is already attached"
        )

        diagnostics = (
            station.state["diag"]
            .replace(" ", "")
        )

        check(
            "a=0,b=0" in diagnostics,
            "Vacuum channels must start OFF"
        )

        current = station.check_fresh()

        print(
            "Current joints:",
            np.round(
                current,
                4
            ).tolist()
        )

        candidates = find_hover_ik(
            model,
            current,
            hover_target,
            green,
            args.seeds
        )

        plan = None

        for index, candidate in enumerate(
            candidates[:args.max_plans],
            start=1
        ):

            _, q, yaw, ep, eo = candidate

            print(
                f"Planning candidate {index}: "
                f"yaw={math.degrees(yaw):.0f}°, "
                f"position error={ep*1000:.2f} mm"
            )

            try:

                plan = station.plan(
                    "00_hover",
                    q,
                    holding=False
                )

                break

            except (
                Blocked,
                RuntimeError
            ) as exc:

                print(
                    "MoveIt rejected candidate:",
                    exc
                )

        check(
            plan is not None,
            "MoveIt found no acceptable collision-aware path"
        )

        output = args.hover_plan

        output.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        output.write_text(
            json.dumps(
                plan,
                indent=2
            )
        )

        print()
        print("PASS: Fresh hover trajectory generated")
        print("Saved:", output)

        if args.mode == "plan-hover":

            print(
                "PLAN-ONLY: No movement or vacuum commands."
            )

            return

        print()
        print(
            "EXECUTING fresh trajectory "
            "in Gazebo simulation..."
        )

        station.move(
            "00_hover",
            plan,
            carrying=False
        )

        actual = station.check_fresh()
        T = model.T(actual)

        error = float(
            np.linalg.norm(
                T[:3, 3] - hover_target
            )
        )

        check(
            error < 0.007
            and four_cups_fit(T, green)
            and T[2, 2] > 0.995,
            f"Hover verification failed: {error:.4f} m"
        )

        print()
        print("PASS: UF850 reached hover")
        print("PASS: Four-cup footprint fits the box")
        print("PASS: Vacuum remains OFF")

    finally:
        station.close()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=(
            "plan-hover",
            "move-hover"
        ),
        default="plan-hover"
    )

    parser.add_argument(
        "--urdf",
        type=Path,
        default=URDF_PATH
    )

    parser.add_argument(
        "--world",
        type=Path,
        default=WORLD_PATH
    )

    parser.add_argument(
        "--pose-snapshot",
        type=Path,
        default=Path(
            "/workspace/data/uf850_gz_pose_snapshot.json"
        )
    )

    parser.add_argument(
        "--reports",
        type=Path,
        default=Path(
            "/workspace/data/uf850_stack_v8"
        )
    )

    parser.add_argument(
        "--hover-plan",
        type=Path,
        default=Path(
            "/workspace/data/uf850_hover_plan.json"
        )
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=48
    )

    parser.add_argument(
        "--max-plans",
        type=int,
        default=5
    )

    parser.add_argument(
        "--ack-simulation-only",
        action="store_true"
    )

    parser.add_argument(
        "--ack-conveyor-stopped",
        action="store_true"
    )

    parser.add_argument(
        "--ack-scene-verified",
        action="store_true"
    )

    args = parser.parse_args()

    try:
        run(args)

    except (
        Blocked,
        RuntimeError,
        ValueError,
        OSError,
        ArithmeticError,
        KeyboardInterrupt
    ) as exc:

        print(
            "BLOCKED:",
            exc,
            file=sys.stderr
        )

        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
