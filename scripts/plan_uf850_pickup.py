#!/usr/bin/env python3
"""Plan-only UF850+VGC10 joint-space hover path with MoveIt collision objects.

No motion commands, no FollowJointTrajectory calls, no vacuum activation.
This DOES NOT plan a Cartesian descent or validate four-cup contact.
"""
import argparse
from pathlib import Path
import json
import math
import sys
from uf850_moveit_common import (JOINT_NAMES, DEFAULT_CANDIDATE, URDF_PATH, WORLD_PATH,
    parse_robot, normalized_candidate, parse_sdf_obstacles)


def preflight(args):
    root, planning_frame, joints, limits = parse_robot(args.urdf)
    chosen, errors = normalized_candidate(args.candidate, limits)
    obstacles = parse_sdf_obstacles(args.world)
    print('UF850 / VGC10 -- SIMULATION ONLY, NO MOTION COMMANDS')
    print('Robot root planning frame:', planning_frame)
    print('Gazebo spawn offset is accounted for in obstacle placement.')
    print('VGC10 TCP is part of URDF:', root.find("link[@name='vgc10_tcp']") is not None)
    for name, value in zip(JOINT_NAMES,args.candidate):
        lo,hi,vel = limits[name]
        print(f' {name}: IK={value:+.4f}, URDF=[{lo:+.4f}, {hi:+.4f}], v_max={vel:.3f}')
    if errors:
        for error in errors: print('REJECTED:',error)
    else:
        print('Equivalent joint target within URDF limits:', [round(x,4) for x in chosen])
    print('World collision boxes staged:',len(obstacles))
    for obs in obstacles:
        print('  ',obs['name'], 'center=', [round(x,3) for x in obs['center']], 'size=',obs['size'])
    print('NOTE: obstacles are an SDF snapshot, not tracked dynamically. Conveyor MUST be stopped.')
    return chosen, planning_frame, obstacles


def plan_moveit(args, chosen, planning_frame, obstacles):
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from moveit_msgs.action import MoveGroup
    from moveit_msgs.srv import ApplyPlanningScene
    from moveit_msgs.msg import CollisionObject, Constraints, JointConstraint, MoveItErrorCodes
    from shape_msgs.msg import SolidPrimitive
    from geometry_msgs.msg import Pose
    from sensor_msgs.msg import JointState

    rclpy.init()
    node=Node('uf850_plan_only_probe', parameter_overrides=[])
    try:
        # For move_group, the entire robot model remains in ROS state monitor;
        # only an allowed equivalent joint target gets sent, never an execute goal.
        current=[None]
        def joints_cb(msg):
            d=dict(zip(msg.name,msg.position))
            if all(n in d for n in JOINT_NAMES):
                current[0]=[d[n] for n in JOINT_NAMES]
        qos=QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.VOLATILE)
        sub=node.create_subscription(JointState,'/joint_states',joints_cb,qos)
        import time
        end=time.monotonic()+15.0
        while current[0] is None and time.monotonic()<end:
            rclpy.spin_once(node,timeout_sec=.2)
        if current[0] is None:
            raise RuntimeError('UF850 /joint_states unavailable')
        print('Live UF850 joint positions:',[round(x,4) for x in current[0]])

        # Apply ONLY the named primitive collisions to the MoveIt world. No scene
        # broad 'clear' or blanket disable-collision operations are permitted.
        scene=ApplyPlanningScene.Request()
        scene.scene.is_diff=True
        scene.scene.robot_state.is_diff=True
        for obs in obstacles:
            co=CollisionObject()
            co.id='v6_'+obs['name']
            co.header.frame_id=planning_frame
            co.operation=CollisionObject.ADD
            prim=SolidPrimitive()
            prim.type=SolidPrimitive.BOX
            prim.dimensions=list(obs['size'])
            pose=Pose(); pose.orientation.w=1.0
            pose.position.x,pose.position.y,pose.position.z=obs['center']
            co.primitives=[prim];co.primitive_poses=[pose]
            scene.scene.world.collision_objects.append(co)
        client=node.create_client(ApplyPlanningScene,'/apply_planning_scene')
        if not client.wait_for_service(timeout_sec=25.0):
            raise RuntimeError('MoveIt /apply_planning_scene service unavailable; inspect moveit logs')
        future=client.call_async(scene)
        rclpy.spin_until_future_complete(node,future,timeout_sec=15.0)
        result=future.result()
        if result is None or not result.success:
            raise RuntimeError('MoveIt rejected planning scene collision objects')
        print('Applied',len(obstacles),'SDF collision boxes to the MoveIt planning scene.')

        action=ActionClient(node,MoveGroup,'/move_action')
        if not action.wait_for_server(timeout_sec=25.0):
            raise RuntimeError('MoveIt /move_action server unavailable; inspect moveit logs')
        goal=MoveGroup.Goal()
        req=goal.request
        req.group_name='arm'
        req.pipeline_id='ompl'
        req.planner_id='RRTConnectkConfigDefault'
        req.num_planning_attempts=5
        req.allowed_planning_time=12.0
        req.max_velocity_scaling_factor=0.60
        req.max_acceleration_scaling_factor=0.60
        constraints=Constraints()
        for name,position in zip(JOINT_NAMES,chosen):
            c=JointConstraint()
            c.joint_name=name;c.position=float(position)
            c.tolerance_above=0.015;c.tolerance_below=0.015
            c.weight=1.0
            constraints.joint_constraints.append(c)
        req.goal_constraints=[constraints]
        req.start_state.is_diff=True  # use MoveIt's current state monitor
        goal.planning_options.plan_only=True
        goal.planning_options.replan=False
        goal.planning_options.look_around=False
        goal.planning_options.planning_scene_diff.is_diff=True
        goal.planning_options.planning_scene_diff.robot_state.is_diff=True
        sent=action.send_goal_async(goal)
        rclpy.spin_until_future_complete(node,sent,timeout_sec=10.0)
        handle=sent.result()
        if handle is None or not handle.accepted:
            raise RuntimeError('MoveGroup plan-only goal rejected')
        future=handle.get_result_async()
        rclpy.spin_until_future_complete(node,future,timeout_sec=30.0)
        wrapped=future.result()
        if wrapped is None:
            raise RuntimeError('MoveIt planner timed out')
        response=wrapped.result
        code=response.error_code.val
        print('MoveIt result code:',code,'(SUCCESS=1)')
        trajectory=response.planned_trajectory.joint_trajectory
        print('Trajectory waypoints:',len(trajectory.points))
        if code != MoveItErrorCodes.SUCCESS or not trajectory.points:
            raise RuntimeError('NO SAFE PLAN produced; do not command robot')
        report={'success':True, 'plan_only':True,'joint_names':list(trajectory.joint_names),
                'start_joints':current[0], 'target_joints':chosen,
                'waypoints':[{'positions':list(pt.positions),
                              'seconds':pt.time_from_start.sec+pt.time_from_start.nanosec*1e-9}
                             for pt in trajectory.points],
                'obstacle_count':len(obstacles), 'planning_frame':planning_frame,
                'planning_time_s':float(response.planning_time)}
        out=Path(args.output)
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(report,indent=2))
        print('PASS: MoveIt returned a PLAN-ONLY collision-aware trajectory')
        print('Saved:',out)
        print('NO EXECUTION performed. Review scene geometry and path before moving anything.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--urdf',type=Path,default=URDF_PATH)
    p.add_argument('--world',type=Path,default=WORLD_PATH)
    p.add_argument('--candidate',type=float,nargs=6,default=DEFAULT_CANDIDATE)
    p.add_argument('--plan',action='store_true',help='Call MoveIt plan-only action; no execution')
    p.add_argument('--output',default='/workspace/data/uf850_hover_plan.json')
    args=p.parse_args()
    chosen,frame,obstacles=preflight(args)
    if chosen is None:
        raise SystemExit('REJECTED: at least one joint is outside manufacturer URDF limits')
    if args.plan:
        plan_moveit(args,chosen,frame,obstacles)
    else:
        print('PREFLIGHT PASSED. Run with --plan for MoveIt collision planning (still no execution).')

if __name__=='__main__':
    main()
