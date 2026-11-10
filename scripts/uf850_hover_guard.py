#!/usr/bin/env python3
"""Guarded UF850 hover-trajectory inspection and OPTIONAL Gazebo-only action replay.

Default mode never publishes a robot command.
Execution requires a freshly generated MoveIt plan plus explicit acknowledgements.
Does NOT independently validate the current Gazebo collision scene or prove ROS domain
is isolated from real hardware. Never run against a hardware-connected ROS graph.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

NAMES = [f"joint{i}" for i in range(1, 7)]
ACTION = "/uf850_arm_controller/follow_joint_trajectory"
DEFAULT_PLAN = "/workspace/data/uf850_hover_plan.json"
DEFAULT_URDF = "/workspace/simulation/generated/uf850_vgc10.urdf"
DEFAULT_PLANNER = "/workspace/scripts/plan_uf850_pickup.py"


class GuardError(RuntimeError):
    pass


def require(ok, msg):
    if not ok:
        raise GuardError(msg)


def validate_file(path: Path, urdf_path=None):
    d = json.loads(path.read_text())
    require(d.get("success") is True and d.get("plan_only") is True,
            "Expected successful plan-only result")
    require(d.get("joint_names") == NAMES, "Wrong joint names/order")
    w = d.get("waypoints")
    require(isinstance(w, list) and len(w) >= 2, "Missing trajectory points")
    require(d.get("obstacle_count", 0) >= 5, "Fewer than five obstacles reported")
    require(d.get("planning_frame") == "world", "Unexpected planning frame")
    starts = d.get("start_joints")
    goals = d.get("target_joints")
    require(isinstance(starts, list) and len(starts) == 6, "Malformed start state")
    require(isinstance(goals, list) and len(goals) == 6, "Malformed goal state")
    last_t = -1.0
    positions = []
    ts = []
    for i, p in enumerate(w):
        t = p.get("seconds")
        q = p.get("positions")
        require(isinstance(t, (int, float)) and math.isfinite(t), f"Bad time at point {i}")
        require(isinstance(q, list) and len(q) == 6 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in q),
                f"Bad positions at point {i}")
        require(t > last_t, f"Non-increasing time at point {i}")
        require(t >= 0, f"Negative time at point {i}")
        ts.append(t)
        positions.append(q)
        last_t = t
    require(abs(ts[0]) < 1e-6, "Trajectory does not begin at t=0")
    require(ts[-1] <= 120.0, "Duration exceeds safety threshold")
    require(max(abs(a-b) for a, b in zip(positions[0], starts)) < 1e-4,
            "First waypoint differs from declared start")
    goal_deviation = [abs(a-b) for a,b in zip(positions[-1], goals)]
    require(max(goal_deviation) <= 0.02, "Final waypoint is >0.02 rad from planned IK target")

    max_speed = [0.0] * 6
    max_accel = [0.0] * 6
    prev_velocities = None
    previous_dt = None
    for i in range(1, len(w)):
        dt = ts[i] - ts[i-1]
        velocity = [(positions[i][k]-positions[i-1][k])/dt for k in range(6)]
        for k in range(6):
            max_speed[k] = max(max_speed[k], abs(velocity[k]))
            require(abs(positions[i][k]-positions[i-1][k]) <= 0.02001,
                    f"Excessive joint jump at {i}: {NAMES[k]}")
        if prev_velocities is not None:
            mid_dt = (dt + previous_dt)/2
            for k in range(6):
                accel = abs(velocity[k]-prev_velocities[k])/mid_dt
                max_accel[k] = max(max_accel[k], accel)
        prev_velocities = velocity
        previous_dt = dt
    require(max(max_speed) < 0.21, "Planned speed exceeds 0.21 rad/s")
    require(max(max_accel) < 0.26, "Planned acceleration exceeds 0.26 rad/s²")

    if urdf_path is not None:
        require(urdf_path.is_file(), f"URDF not found: {urdf_path}")
        root = ET.parse(urdf_path).getroot()
        for k, name in enumerate(NAMES):
            joint = root.find(f"./joint[@name='{name}']")
            require(joint is not None, f"Missing URDF joint {name}")
            typ = joint.get("type")
            if typ == "continuous":
                continue
            limit = joint.find("limit")
            require(limit is not None, f"Missing limits for {name}")
            lo = float(limit.get("lower"))
            hi = float(limit.get("upper"))
            require(all(lo - 1e-6 <= p[k] <= hi + 1e-6 for p in positions),
                    f"Trajectory exceeds URDF limits at {name}: [{lo},{hi}]")

    # How much does the trajectory deviate from a straight line in joint space?
    q0, q1 = positions[0], positions[-1]
    main_axis = max(range(6), key=lambda k: abs(q1[k]-q0[k]))
    biggest = 0.0
    for q in positions:
        fraction = (q[main_axis] - q0[main_axis]) / (q1[main_axis]-q0[main_axis])
        for k in range(6):
            biggest = max(biggest, abs(q[k] - (q0[k] + fraction*(q1[k]-q0[k]))))

    print(f"PASS: {len(w)} waypoints, {ts[-1]:.3f} sec, {d['obstacle_count']} configured obstacles")
    print("Planning frame:", d['planning_frame'])
    print("Start:", [round(x,4) for x in positions[0]])
    print("Goal:", [round(x,4) for x in goals])
    print("Endpoint:", [round(x,4) for x in positions[-1]])
    print("Goal error (rad):", [round(x,5) for x in goal_deviation])
    print("Max velocity (rad/s):", [round(x,4) for x in max_speed])
    print("Max acceleration (rad/s²):", [round(x,4) for x in max_accel])
    print(f"Joint-line deviation: {biggest:.2e} rad")
    print("URDF limits:", "checked" if urdf_path is not None else "NOT checked (provide --urdf)")
    print("WARNING: JSON cannot prove live obstacle poses, link collisions, or TCP clearance.")
    return d


def get_ros_checks(plan, execute):
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Bool, String
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration

    rclpy.init()
    node = Node("uf850_hover_guard")
    state = {"joints": None, "clock": None, "attached": None, "diagnostics": None,
             "time_received": 0.0}

    def joints_cb(msg):
        if len(set(NAMES) & set(msg.name)) != len(NAMES):
            return
        index = {n:i for i,n in enumerate(msg.name)}
        if not all(index[n] < len(msg.position) for n in NAMES):
            return
        state["joints"] = [float(msg.position[index[n]]) for n in NAMES]
        state["time_received"] = time.monotonic()

    node.create_subscription(JointState, "/joint_states", joints_cb, 20)
    node.create_subscription(Clock, "/clock", lambda msg: state.update(clock=msg.clock.sec + msg.clock.nanosec * 1e-9), 10)
    node.create_subscription(Bool, "/vgc10/attached", lambda msg: state.update(attached=msg.data), 10)
    node.create_subscription(String, "/vgc10/diagnostics", lambda msg: state.update(diagnostics=msg.data), 10)
    client = ActionClient(node, FollowJointTrajectory, ACTION)

    try:
        deadline = time.monotonic() + 10.0
        first_clock = None
        clock_progress = False
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if state["clock"] is not None:
                if first_clock is None:
                    first_clock = state["clock"]
                if state["clock"] > first_clock + 0.05:
                    clock_progress = True
            if all([state["joints"] is not None, state["attached"] is not None,
                    state["diagnostics"] is not None, clock_progress]):
                break
        require(clock_progress, "No advancing /clock: Gazebo not running or simulation paused")
        require(state["joints"] is not None, "No six-axis /joint_states feedback")
        require(state["attached"] is False, "VGC10 is attached: not safe for hover test")
        require(state["diagnostics"] is not None, "No vacuum diagnostics")
        require("a=0,b=0" in state["diagnostics"].replace(" ", ""),
                "Vacuum must be OFF on both channels")
        publishers = node.get_publishers_info_by_topic("/joint_states")
        require(len(publishers) == 1,
                f"Expected one joint-state publisher, found {len(publishers)}")
        deviation = [abs(a-b) for a,b in zip(state["joints"], plan["waypoints"][0]["positions"])]
        require(max(deviation) < 0.03,
                f"Live /joint_states differs from planned start; max error {max(deviation):.4f} rad")
        require(client.wait_for_server(timeout_sec=5.0), "UF850 follow_joint_trajectory action unavailable")
        print("PASS: advancing Gazebo /clock, one /joint_states publisher, vacuum OFF")
        print("PASS: live start matches fresh planned start (max error: %.4f rad)" % max(deviation))
        print("PASS: controller action available:", ACTION)
        if not execute:
            print("READ-ONLY LIVE CHECK COMPLETE — no action was sent")
            return

        # Guard against motion between state check and action dispatch.
        require(time.monotonic() - state["time_received"] < 1.0,
                "Joint feedback is stale; not executing")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(NAMES)
        # The trajectory header has the default zero stamp: start as soon as accepted.
        for p in plan["waypoints"]:
            point = JointTrajectoryPoint()
            point.positions = [float(x) for x in p["positions"]]
            secs = float(p["seconds"])
            whole = int(secs)
            nanos = round((secs-whole)*1_000_000_000)
            if nanos >= 1_000_000_000:
                whole += 1; nanos -= 1_000_000_000
            point.time_from_start = Duration(sec=whole, nanosec=nanos)
            goal.trajectory.points.append(point)
        print("SENDING simulator-only FollowJointTrajectory goal", flush=True)
        sending = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, sending, timeout_sec=10.0)
        require(sending.done(), "No reply to trajectory goal")
        handle = sending.result()
        require(handle is not None and handle.accepted, "Controller rejected goal")
        print("Goal accepted; monitoring completion.", flush=True)
        result_future = handle.get_result_async()
        try:
            rclpy.spin_until_future_complete(node, result_future,
                                             timeout_sec=float(plan["waypoints"][-1]["seconds"]) + 15.0)
        except KeyboardInterrupt:
            print("Interrupted; requesting trajectory cancellation")
            cancellation = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(node, cancellation, timeout_sec=5.0)
            raise GuardError("Execution cancelled by user")
        if not result_future.done():
            cancellation = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(node, cancellation, timeout_sec=5.0)
            raise GuardError("Controller action timed out; cancellation requested")
        outcome = result_future.result()
        require(outcome is not None, "No action result")
        require(outcome.result.error_code == 0,
                f"Controller returned error {outcome.result.error_code}: {outcome.result.error_string}")
        # One more check of actual final joint state.
        final_deadline = time.monotonic()+3
        while time.monotonic() < final_deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        actual = state["joints"]
        final = plan["waypoints"][-1]["positions"]
        err = max(abs(a-b) for a,b in zip(actual,final))
        require(err < 0.03, f"Final measured joint error {err:.4f} rad exceeds 0.03 rad")
        print(f"PASS: simulator arm reached hover goal, maximum joint error {err:.4f} rad")
        print("VACUUM STAYS OFF. Verify gripper pose in both cameras before descent.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan-file", type=Path, default=Path(DEFAULT_PLAN))
    ap.add_argument("--urdf", type=Path, default=None,
                    help="Path to model URDF; check all waypoints against joint limits")
    ap.add_argument("--live-check", action="store_true", help="Read ROS/Gazebo state, DO NOT command")
    ap.add_argument("--execute", action="store_true", help="Send action to simulator (requires all safeguards)")
    ap.add_argument("--fresh-plan", action="store_true", help="Run the MoveIt planner immediately before validation")
    ap.add_argument("--planner", default=DEFAULT_PLANNER)
    ap.add_argument("--ack-simulation-only", action="store_true")
    ap.add_argument("--ack-conveyor-stopped", action="store_true")
    ap.add_argument("--ack-scene-verified", action="store_true")
    args = ap.parse_args()
    try:
        if args.execute:
            require(args.fresh_plan, "Execution requires --fresh-plan (never replay a stored plan blindly)")
            require(args.urdf is not None, "Execution requires --urdf for joint limits")
            require(args.ack_simulation_only and args.ack_conveyor_stopped and args.ack_scene_verified,
                    "Execution requires --ack-simulation-only --ack-conveyor-stopped --ack-scene-verified")
        if args.fresh_plan:
            require(Path(args.planner).is_file(), f"Planner not found: {args.planner}")
            print("Generating a fresh MoveIt trajectory (plan-only)...", flush=True)
            result = subprocess.run([sys.executable, args.planner, "--plan"],
                                    capture_output=True, text=True, timeout=120)
            print(result.stdout[-5000:])
            require(result.returncode == 0, f"Planner failed: {result.stderr[-2000:]}")
            require(args.plan_file.is_file(), "Planner did not produce expected JSON")
            require(time.time()-args.plan_file.stat().st_mtime < 120,
                    "Planner JSON not fresh")
        require(args.plan_file.is_file(), f"Trajectory JSON not found: {args.plan_file}")
        data = validate_file(args.plan_file, args.urdf)
        if args.live_check or args.execute:
            get_ros_checks(data, args.execute)
        else:
            print("OFFLINE CHECK COMPLETE. No ROS commands sent.")
    except (GuardError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as ex:
        print(f"BLOCKED: {ex}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
