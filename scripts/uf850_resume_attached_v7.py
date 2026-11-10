#!/usr/bin/env python3
"""Resume simulated stacking after a successful VGC10 attachment."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from uf850_stationary_stack_v7 import (
    Blocked,
    RobotKinematics,
    SimStation,
    check,
    go_stage,
    make_box_pose,
    stacking_targets,
    verify_pose_snapshot,
    world_objects,
)

from uf850_moveit_common import URDF_PATH, WORLD_PATH


def run(args):
    check(
        os.getenv("ROS_DOMAIN_ID") == "42",
        "Only isolated simulation domain 42 is allowed"
    )

    if args.mode == "execute-rest":
        check(
            args.ack_simulation_only
            and args.ack_conveyor_stopped
            and args.ack_scene_verified,
            "Missing required execution acknowledgements"
        )

    verify_pose_snapshot(args.pose_snapshot, args.world)

    snapshot = json.loads(args.pose_snapshot.read_text())
    green_data = snapshot["boxes"]["upper_box"]

    check(
        green_data["ok"]
        and green_data["rot_error_rad"] < 0.01,
        "Green box is no longer at its verified seal position"
    )

    green_center = np.array(
        green_data["actual"], dtype=float
    )

    model = RobotKinematics(args.urdf)
    world = world_objects(args.world)
    targets = stacking_targets(world)

    station = SimStation(model, world, args.reports)

    try:
        station.ready()

        joints = station.check_fresh()
        diag = station.parse_diag()

        check(
            station.state["attached"] is True,
            "Gazebo does not report an attached box"
        )

        check(
            diag.get("a") == 1
            and diag.get("b") == 1,
            "Both vacuum channels must already be ON"
        )

        check(
            diag.get("down_dot", -1) > 0.99
            and abs(diag.get("height_gap_m", 99)) < 0.007
            and diag.get("xy_error_m", 99) < 0.006,
            f"Robot is no longer at a valid seal: {diag}"
        )

        check(
            station.all_contacts(),
            "Four-cup geometric contacts are no longer valid"
        )

        contact_T = model.T(joints)

        check(
            np.linalg.norm(
                contact_T[:2, 3]
                - targets["pickup_seal"][:2]
            ) < 0.01
            and abs(
                contact_T[2, 3]
                - targets["pickup_seal"][2]
            ) < 0.012,
            "Robot is not at the expected seal pose"
        )

        # Recover attachment transform from live Gazebo pose.
        green_T = make_box_pose(green_center)

        station.attach_tf = (
            np.linalg.inv(contact_T) @ green_T
        )

        # Both channels have been verified ON.
        # Do not toggle vacuum during recovery.
        station.vac_energized = True

        print("PASS: Green box attached in Gazebo")
        print("PASS: Vacuum channels A and B ON")
        print("PASS: Current pose matches seal configuration")
        print("PASS: Attachment transform recovered")

        lift_target = (
            contact_T[:3, 3]
            + np.array([0.0, 0.0, 0.15])
        )

        if args.mode == "plan-lift":
            go_stage(
                station,
                model,
                "03_lift",
                lift_target,
                contact_T[:3, :3],
                True,
                False,
                support_contact=True
            )

            print("PASS: Collision-aware lift plan generated")
            print("PLAN ONLY: Robot has not moved")
            print("Vacuum remains ON")
            return

        # Stage 3: lift with attached green box.
        go_stage(
            station,
            model,
            "03_lift",
            lift_target,
            contact_T[:3, :3],
            True,
            True,
            support_contact=True
        )

        # Stage 4: transfer above orange box.
        current_T = model.T(station.check_fresh())

        go_stage(
            station,
            model,
            "04_transfer",
            targets["place_hover"],
            current_T[:3, :3],
            True,
            True
        )

        # Stage 5: approach stationary orange box.
        current_T = model.T(station.check_fresh())

        go_stage(
            station,
            model,
            "05_place_near",
            targets["place_near"],
            current_T[:3, :3],
            True,
            True
        )

        # Check the green box's actual predicted placement.
        release_tcp = model.T(station.check_fresh())
        box_T = release_tcp @ station.attach_tf

        green_size = np.array(
            world["upper_box"]["size"], dtype=float
        )

        half_height = float(
            np.sum(
                np.abs(box_T[2, :3])
                * green_size / 2
            )
        )

        green_bottom = box_T[2, 3] - half_height

        gap = float(
            green_bottom - targets["orange_top"]
        )

        xy_error = float(
            np.linalg.norm(
                box_T[:2, 3]
                - world["lower_box"]["center_world"][:2]
            )
        )

        check(
            0.004 <= gap <= 0.025
            and xy_error < 0.018
            and station.state["attached"] is True,
            f"Unsafe release pose: "
            f"gap={gap:.4f}m, XY={xy_error:.4f}m"
        )

        print(
            f"PASS: Green box aligned above orange "
            f"(gap={gap:.4f}m, XY={xy_error:.4f}m)"
        )

        # Release only at the validated placement pose.
        station.vacuum(False, seconds=0.5)
        station.spin(0.6)

        check(
            station.state["attached"] is False,
            "Gazebo did not detach after release"
        )

        print("PASS: Vacuum released")

        # Stage 6: retreat without the box.
        placed_center = box_T[:3, 3]

        retreat_target = (
            release_tcp[:3, 3]
            + np.array([0.0, 0.0, 0.15])
        )

        go_stage(
            station,
            model,
            "06_retreat",
            retreat_target,
            release_tcp[:3, :3],
            False,
            True,
            released_position=placed_center
        )

        station.attach_tf = None

        print()
        print("PASS: Stationary stack motion sequence complete")
        print("NEXT: Verify final box poses in Gazebo")

    finally:
        station.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["plan-lift", "execute-rest"],
        default="plan-lift"
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
            "/workspace/data/uf850_stack_v7"
        )
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
        KeyError,
        OSError,
        ArithmeticError,
        KeyboardInterrupt
    ) as exc:
        print("BLOCKED:", exc, file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
