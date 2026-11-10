#!/usr/bin/env python3
"""UF850+VGC10: guarded, Gazebo-only stationary pick/lift/stack experiment.

Reads the corrected URDF, the existing SDF station, and live ROS joint/vacuum
feedback. Uses MoveIt PLAN-ONLY for *each* joint path, then (if explicitly
armed) submits it to the Gazebo ros2_control action. Does not move any robot by
simply importing this module.

IMPORTANT: Geometric box/cup proxies, static SDF obstacles, assumed Gazebo
model spawn. NOT FOR HARDWARE OR MOVING CONVEYOR. Never run on a ROS domain that
can reach real robot hardware. Abort leaves vacuum ON when carrying a box.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

try:
    from uf850_moveit_common import JOINT_NAMES, SPAWN_WORLD, URDF_PATH, WORLD_PATH, parse_robot, parse_sdf_obstacles
except ImportError:
    # Allow offline unit tests from a neighboring tests/ directory.
    from scripts.uf850_moveit_common import JOINT_NAMES, SPAWN_WORLD, URDF_PATH, WORLD_PATH, parse_robot, parse_sdf_obstacles

NAMES = JOINT_NAMES
ROBOT_ACTION = '/uf850_arm_controller/follow_joint_trajectory'
MOVE_GROUP_ACTION = '/move_action'
OBJECT_NAME_PREFIX = 'v6_'
MAX_PLAN_SPEED = 0.30  # rad/s, simulated actuator
MAX_PLAN_ACCEL = 0.60  # rad/s^2


class Blocked(RuntimeError):
    pass


def check(ok, why):
    if not ok:
        raise Blocked(why)


def xyz(s, default='0 0 0'):
    return np.array([float(a) for a in (s or default).split()], dtype=float)


def rot_rpy(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
        [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
        [-sp, cp*sr, cp*cr]
    ])


def origin_T(e):
    T = np.eye(4)
    if e is not None:
        T[:3, :3] = rot_rpy(xyz(e.get('rpy')))
        T[:3, 3] = xyz(e.get('xyz'))
    return T


def axis_angle(u, angle):
    u = np.asarray(u, dtype=float)
    u /= np.linalg.norm(u)
    x, y, z = u
    c, s = math.cos(angle), math.sin(angle)
    return c*np.eye(3) + (1-c)*np.outer(u,u) + s*np.array([[0,-z,y],[z,0,-x],[-y,x,0]])


def rotvec(R):
    """SO(3) logarithm, adequate for small IK corrections; near-pi guarded."""
    c = max(-1.0, min(1.0, (np.trace(R)-1)/2))
    angle = math.acos(c)
    if angle < 1e-8:
        return np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / 2
    if angle > math.pi-1e-3:
        raise Blocked('IK rotation error too close to 180 deg; refuse unstable rotation log')
    return angle/(2*math.sin(angle)) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])


def quat_xyzw(R):
    # Numerically stable matrix -> xyzw; source matrix must be rotation.
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(1+t)*2
        return ((R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s,
                (R[1,0]-R[0,1])/s, .25*s)
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s=math.sqrt(max(0,1+R[0,0]-R[1,1]-R[2,2]))*2
        return (.25*s,(R[0,1]+R[1,0])/s,(R[0,2]+R[2,0])/s,(R[2,1]-R[1,2])/s)
    if i == 1:
        s=math.sqrt(max(0,1+R[1,1]-R[0,0]-R[2,2]))*2
        return ((R[0,1]+R[1,0])/s,.25*s,(R[1,2]+R[2,1])/s,(R[0,2]-R[2,0])/s)
    s=math.sqrt(max(0,1+R[2,2]-R[0,0]-R[1,1]))*2
    return ((R[0,2]+R[2,0])/s,(R[1,2]+R[2,1])/s,.25*s,(R[1,0]-R[0,1])/s)


class RobotKinematics:
    def __init__(self, urdf, spawn=SPAWN_WORLD):
        root = ET.parse(urdf).getroot()
        by_child = {j.find('child').get('link'): j for j in root.findall('joint')
                    if j.find('child') is not None}
        c = 'vgc10_tcp'
        chain=[]
        while c in by_child:
            j=by_child[c]
            chain.append(j)
            c=j.find('parent').get('link')
        check(chain, 'Missing URDF vgc10_tcp transform')
        chain.reverse()
        self.root_frame=c
        self.chain=[]
        self.lo={};self.hi={}
        for j in chain:
            n=j.get('name');typ=j.get('type')
            ax=j.find('axis')
            self.chain.append((n,typ,origin_T(j.find('origin')),
                               xyz(ax.get('xyz') if ax is not None else '1 0 0')))
            if n in NAMES:
                lim=j.find('limit')
                check(lim is not None and lim.get('lower') and lim.get('upper'), f'Joint {n} lacks explicit limits')
                self.lo[n]=float(lim.get('lower'))
                self.hi[n]=float(lim.get('upper'))
        check(set(NAMES) == set(self.lo), 'URDF TCP chain does not contain all six UF850 joints')
        # The user's measured TF is link6 -> TCP: +0.189396m, roll=pi.
        # If this changes, the plugin's suction-axis convention needs rework.
        tool=np.eye(4)
        after_link6=False
        for j in chain:
            if j.find('parent').get('link')=='link6':
                after_link6=True
            if after_link6:
                check(j.get('type')=='fixed', 'Moving joint detected between link6 and TCP')
                tool=tool@origin_T(j.find('origin'))
        check(after_link6, 'URDF chain does not contain link6 -> TCP')
        check(np.linalg.norm(tool[:3,3]-[0,0,.189395921])<.003 and
              np.linalg.norm(tool[:3,:3]-rot_rpy([math.pi,0,0]))<.03,
              'Corrected VGC10 link6 -> TCP transform does not match measured 180deg / +0.189396m')
        self.lo_vec=np.array([self.lo[n] for n in NAMES]);self.hi_vec=np.array([self.hi[n] for n in NAMES])
        self.T_spawn=np.eye(4)
        self.T_spawn[:3,3]=spawn
        check(self.root_frame == 'world', f'Expected URDF root "world", got {self.root_frame!r}')

    def T(self, q):
        joints = dict(zip(NAMES, q))
        T=self.T_spawn.copy()
        for name,typ,origin,axis in self.chain:
            T=T@origin
            if typ in ('revolute','continuous'):
                a=np.eye(4)
                a[:3,:3]=axis_angle(axis,joints.get(name,0.0))
                T=T@a
            elif typ=='prismatic':
                a=np.eye(4)
                a[:3,3]=axis*joints.get(name,0.0)
                T=T@a
        return T

    def solve_near(self, start, target_world, desired_R, tol_pos=.002, tol_angle=.025):
        """Damped least-squares near the *current* joint branch, full TCP pose."""
        x=np.array(start,dtype=float)
        target=np.array(target_world,dtype=float)
        check(np.all(np.isfinite(x)) and np.all(np.isfinite(target)), 'IK non-finite input')
        check(np.all(x>=self.lo_vec-.01) and np.all(x<=self.hi_vec+.01),'IK seed out of limits')
        best=(float('inf'),None,None)
        for _ in range(250):
            T=self.T(x)
            ep=target-T[:3,3]
            eo=rotvec(desired_R@T[:3,:3].T)
            score=np.linalg.norm(ep)+0.12*np.linalg.norm(eo)
            if score < best[0]:best=(score,x.copy(),(np.linalg.norm(ep),np.linalg.norm(eo)))
            if np.linalg.norm(ep)<tol_pos and np.linalg.norm(eo)<tol_angle:
                return x
            J=np.zeros((6,6))
            step=1e-5
            for j in range(6):
                xx=x.copy();xx[j]+=step
                other=self.T(xx)
                J[:3,j]=(other[:3,3]-T[:3,3])/step
                J[3:,j]=rotvec(other[:3,:3]@T[:3,:3].T)/step
            e=np.r_[ep,eo]
            dq=J.T@np.linalg.solve(J@J.T + (0.015**2)*np.eye(6), e)
            dq=np.clip(dq,-.10,.10)
            improved=False
            for f in (1.0, 0.5, 0.25, 0.125):
                trial=np.clip(x+f*dq, self.lo_vec+1e-5, self.hi_vec-1e-5)
                TT=self.T(trial)
                val=np.linalg.norm(target-TT[:3,3])+0.12*np.linalg.norm(rotvec(desired_R@TT[:3,:3].T))
                if val<score-1e-7:
                    x=trial;improved=True;break
            if not improved:break
        raise Blocked(f'IK cannot achieve TCP target within tolerance; best position/angle errors: {best[2]}')


def world_objects(sdf):
    boxes=parse_sdf_obstacles(sdf)
    result={}
    for d in boxes:
        name=d['name'].split('_')[0]
        # Names starting lower_box and upper_box cannot be split on underscore.
        for candidate in ('upper_box','lower_box','pickup_table','conveyor','uf850_pedestal'):
            if d['name'].startswith(candidate+'_'):
                name=candidate
                break
        if name in result:
            raise Blocked(f'World has more than one SDF collision for {name}: update this script')
        result[name]={**d,'center_world':np.array(d['center'])+np.array(SPAWN_WORLD)}
    for name in ('upper_box','lower_box','pickup_table','conveyor','uf850_pedestal'):
        check(name in result,f'Missing required SDF obstacle {name}')
    return result


def planned_limits_ok(data, start, end, model):
    """Offline sanity checks on one MoveIt-generated trajectory."""
    check(data.get('joint_names')==NAMES,'MoveIt returned unexpected joint names or order')
    pts=data.get('waypoints',[])
    check(len(pts)>=2,'Empty MoveIt path')
    check(data.get('success') is True and data.get('plan_only') is True, 'Not a successful plan-only MoveIt result')
    t_last=-1
    last=None
    last_v=None
    prev_dt=None
    max_v=0
    max_a=0
    for i,p in enumerate(pts):
        t=float(p['seconds']);q=np.asarray(p['positions'],dtype=float)
        check(len(q)==6 and np.all(np.isfinite(q)) and math.isfinite(t) and t>t_last, 'Invalid trajectory samples or times')
        check(np.all(q>=model.lo_vec-1e-5) and np.all(q<=model.hi_vec+1e-5), 'Trajectory exceeds URDF limits')
        if last is not None:
            dt=t-t_last
            vel=(q-last)/dt
            max_v=max(max_v,float(np.max(np.abs(vel))))
            if last_v is not None:
                max_a=max(max_a,float(np.max(np.abs(vel-last_v)/((prev_dt+dt)/2))))
            last_v=vel;prev_dt=dt
        last=q;t_last=t
    check(abs(pts[0]['seconds'])<=1e-6,'Plan must start at t=0')
    check(t_last<=120.,'Path takes >120 s')
    check(max_v<MAX_PLAN_SPEED and max_a<MAX_PLAN_ACCEL,
          f'Motion exceeds conservative speed/accel caps {max_v:.3f}/{max_a:.3f}')
    check(max(abs(a-b) for a,b in zip(pts[0]['positions'],start))<.03,'Plan starts away from live joints')
    check(max(abs(a-b) for a,b in zip(pts[-1]['positions'],end))<.003,
          'Final MoveIt waypoint differs >0.003 rad from target; tighten tolerance')
    return max_v,max_a


def change_acm_pair(matrix, a, b, allowed, remembered_default=None):
    """Change exactly one symmetric ACM pair; preserve all other exclusions.

    `matrix` is a moveit_msgs/AllowedCollisionMatrix-like object. Entry type
    comes from the caller's message class; this function accepts its factory.
    Returns (new entry names, new rows, old_setting) for explicit application.
    """
    names=list(matrix.entry_names)
    rows=[list(e.enabled) for e in matrix.entry_values]
    check(len(names)==len(rows) and all(len(row)==len(names) for row in rows),
          'MoveIt returned a malformed allowed collision matrix')
    if remembered_default is None:
        old_setting=bool(rows[names.index(a)][names.index(b)]) if a in names and b in names else False
    else:
        old_setting=remembered_default
    for name in (a,b):
        if name not in names:
            for row in rows:row.append(False)
            rows.append([False]*(len(names)+1))
            names.append(name)
    ia,ib=names.index(a),names.index(b)
    rows[ia][ib]=rows[ib][ia]=(allowed or old_setting)
    return names,rows,old_setting


class SimStation:
    def __init__(self, model, world, report_dir):
        import rclpy
        from rclpy.node import Node
        from rclpy.action import ActionClient
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool, String
        from control_msgs.action import FollowJointTrajectory
        from moveit_msgs.action import MoveGroup
        from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene

        rclpy.init()
        self.rclpy=rclpy
        self.node=Node('uf850_stationary_stack_v7')
        self.model=model;self.world=world;self.report_dir=report_dir
        self.state={'joints':None,'joint_t':0.,'attached':None,'attached_t':0.,'diag':None,
                    'diag_t':0.,'cups':None,'cups_t':0.,'clock':None,'clock_t':0.}
        def j_cb(m):
            d=dict(zip(m.name,m.position))
            if all(n in d for n in NAMES):
                self.state['joints']=[float(d[n]) for n in NAMES]
                self.state['joint_t']=time.monotonic()
        def clock_cb(m):
            self.state['clock']=m.clock.sec+m.clock.nanosec*1e-9
            self.state['clock_t']=time.monotonic()
        def attach_cb(m):
            self.state['attached']=bool(m.data);self.state['attached_t']=time.monotonic()
        def diag_cb(m):
            self.state['diag']=str(m.data);self.state['diag_t']=time.monotonic()
        def cups_cb(m):
            self.state['cups']=str(m.data);self.state['cups_t']=time.monotonic()
        self.node.create_subscription(JointState,'/joint_states',j_cb,20)
        self.node.create_subscription(Clock,'/clock',clock_cb,10)
        self.node.create_subscription(Bool,'/vgc10/attached',attach_cb,10)
        self.node.create_subscription(String,'/vgc10/diagnostics',diag_cb,10)
        self.node.create_subscription(String,'/vgc10/cup_contacts',cups_cb,10)
        self.pa=self.node.create_publisher(Bool,'/vgc10/channel_a/enable',10)
        self.pb=self.node.create_publisher(Bool,'/vgc10/channel_b/enable',10)
        self.vacbool=Bool
        self.controller=ActionClient(self.node,FollowJointTrajectory,ROBOT_ACTION)
        self.planner=ActionClient(self.node,MoveGroup,MOVE_GROUP_ACTION)
        self.scene_client=self.node.create_client(ApplyPlanningScene,'/apply_planning_scene')
        self.scene_get_client=self.node.create_client(GetPlanningScene,'/get_planning_scene')
        self.m_arm=FollowJointTrajectory
        self.m_move=MoveGroup
        self.m_scene=ApplyPlanningScene
        self.m_getscene=GetPlanningScene
        self.last_plan=None
        self.attach_tf=None
        self.vac_energized=False
        self.phase='START'
        self.trace=[]
        self._last_trace=0.0

    def spin(self,duration):
        until=time.monotonic()+duration
        while time.monotonic()<until:
            self.rclpy.spin_once(self.node,timeout_sec=min(.06,max(0.,until-time.monotonic())))
            if self.state['joints'] is not None and time.monotonic()-self._last_trace>=.1:
                self._last_trace=time.monotonic()
                self.trace.append({'wall_time':time.time(),'sim_time':self.state['clock'],
                                   'phase':self.phase,'joints':list(self.state['joints']),
                                   'attached':self.state['attached'],
                                   'vacuum_command_on':self.vac_energized,
                                   'diagnostics':self.state['diag'],'cups':self.state['cups']})

    def ready(self):
        start=time.monotonic()
        firstclock=None
        advanced=False
        while time.monotonic()-start<15.:
            self.spin(.12)
            if self.state['clock'] is not None:
                if firstclock is None:firstclock=self.state['clock']
                if self.state['clock']>firstclock+.06:advanced=True
            if advanced and all(self.state[k] is not None for k in ('joints','attached','diag','cups')):
                break
        check(advanced,'Gazebo clock is not advancing')
        check(self.state['joints'] is not None,'Missing 6 UF850 joints')
        check(self.state['attached'] is not None,'Missing VGC10 attachment topic')
        check(self.state['diag'] is not None,'Missing VGC10 diagnostics topic')
        check(self.state['cups'] is not None,'Missing VGC10 contacts topic')
        check(len(self.node.get_publishers_info_by_topic('/joint_states'))==1,
              'Multiple /joint_states publishers; stop duplicate bridge')
        check(self.controller.wait_for_server(timeout_sec=10.),'UF850 trajectory controller unavailable')
        check(self.planner.wait_for_server(timeout_sec=15.),'MoveIt move_action unavailable')
        check(self.scene_client.wait_for_service(timeout_sec=15.),'MoveIt apply_planning_scene unavailable')
        check(self.scene_get_client.wait_for_service(timeout_sec=15.),'MoveIt get_planning_scene unavailable')
        print('PASS: Gazebo, state feedback, vacuum bridge and MoveIt available')

    def check_fresh(self):
        self.spin(.12)
        now=time.monotonic()
        for name,t in (('joint states',self.state['joint_t']),('clock',self.state['clock_t']),
                       ('vacuum diagnostics',self.state['diag_t']),('attachment',self.state['attached_t'])):
            check(now-t<1.0,f'Stale {name} telemetry')
        return np.array(self.state['joints'])

    def vacuum(self,on,seconds=.7):
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            m=self.vacbool();m.data=bool(on)
            self.pa.publish(m);self.pb.publish(m)
            self.spin(.08)
        self.vac_energized=bool(on)

    def parse_diag(self):
        result={}
        for key,value in re.findall(r'([a-z_]+)=([-+]?\d*\.?\d+)',self.state['diag'] or ''):
            result[key]=float(value)
        return result

    def all_contacts(self):
        txt=(self.state['cups'] or '').replace(' ','')
        return all(f'{n}=1' in txt.split(',') for n in ('a1','a2','b1','b2'))

    def check_seal(self):
        self.check_fresh()
        d=self.parse_diag()
        check(d.get('down_dot',-1)>.99, f'Suction orientation off: {d}')
        check(abs(d.get('height_gap_m',99))<.007, f'Suction height not within 7mm: {d}')
        check(d.get('xy_error_m',99)<.006, f'Suction center >6mm from target: {d}')
        check(self.all_contacts(), f'Not all four cup centers in seal geometry: {self.state["cups"]}')
        check(self.state['attached'] is False, 'Unexpected attached before switching vacuum on')
        print('PASS: four geometric cup contacts, near-vertical suction, correct height')

    def apply_scene(self, holding=False, released_position=None, support_contact=False):
        from moveit_msgs.msg import CollisionObject, AttachedCollisionObject
        from shape_msgs.msg import SolidPrimitive
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import PlanningSceneComponents, AllowedCollisionEntry
        scene=self.m_scene.Request()
        scene.scene.is_diff=True
        scene.scene.robot_state.is_diff=True
        items=self.world
        upper=items['upper_box']
        for name,item in items.items():
            co=CollisionObject()
            co.id=OBJECT_NAME_PREFIX+item['name']
            co.header.frame_id=self.model.root_frame
            co.operation=CollisionObject.ADD
            shape=SolidPrimitive()
            shape.type=SolidPrimitive.BOX
            shape.dimensions=list(item['size'])
            p=Pose();p.orientation.w=1.
            local_center=np.array(item['center'])
            if name=='upper_box' and released_position is not None:
                local_center=np.array(released_position)-np.array(SPAWN_WORLD)
            p.position.x,p.position.y,p.position.z=map(float,local_center)
            co.primitives=[shape]
            co.primitive_poses=[p]
            if name!='upper_box' or not holding:
                scene.scene.world.collision_objects.append(co)
        if holding:
            check(self.attach_tf is not None, 'Missing calibrated green box attachment transform')
            # MoveIt automatically removes the matching world object
            # when attaching it. Do not send a second world REMOVE:
            # that REMOVE returns false and rejects the scene update.
            attached=AttachedCollisionObject()
            attached.link_name='vgc10_tcp'
            attached.touch_links=['vgc10_tcp','vgc10_body','link6']+[f'vgc10_cup_{x}' for x in ('a1','a2','b1','b2')]
            attached.object.id=OBJECT_NAME_PREFIX+upper['name']
            attached.object.header.frame_id='vgc10_tcp'
            attached.object.operation=CollisionObject.ADD
            shape=SolidPrimitive();shape.type=SolidPrimitive.BOX;shape.dimensions=list(upper['size'])
            attached.object.primitives=[shape]
            p=Pose()
            p.position.x,p.position.y,p.position.z=map(float,self.attach_tf[:3,3])
            p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w=quat_xyzw(self.attach_tf[:3,:3])
            attached.object.primitive_poses=[p]
            scene.scene.robot_state.attached_collision_objects.append(attached)
        elif self.attach_tf is not None:
            # Remove stale attached shape on transition to released world object.
            attached=AttachedCollisionObject()
            attached.link_name='vgc10_tcp';attached.object.id=OBJECT_NAME_PREFIX+upper['name']
            attached.object.operation=CollisionObject.REMOVE
            scene.scene.robot_state.attached_collision_objects.append(attached)
        # At grasp the green box is resting on its table. For the *first upward
        # lift only* permit that single, intentional object/table contact. All
        # robot-vs-table and box-vs-other-fixture collisions stay enabled.
        # Each new plan restores the contact pair's previous setting.
        req=self.m_getscene.Request()
        req.components.components=PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        rf=self.scene_get_client.call_async(req)
        self.rclpy.spin_until_future_complete(self.node,rf,timeout_sec=10.)
        check(rf.done() and rf.result() is not None,'Unable to inspect existing collision matrix')
        acm=rf.result().scene.allowed_collision_matrix
        a=OBJECT_NAME_PREFIX+upper['name']
        b=OBJECT_NAME_PREFIX+items['pickup_table']['name']
        # Do not replace the matrix with a tiny one: preserve all SRDF
        # adjacent-link exclusions exactly as MoveIt reports them.
        # Force this exception OFF outside lift, even after an interrupted prior run.
        names,rows,_=change_acm_pair(acm,a,b,support_contact,False)
        acm.entry_names=names
        acm.entry_values=[]
        for row in rows:
            e=AllowedCollisionEntry();e.enabled=row;acm.entry_values.append(e)
        scene.scene.allowed_collision_matrix=acm
        if support_contact:
            print('NOTE: permitting ONLY carried green-box / pickup-table contact during vertical lift')
        f=self.scene_client.call_async(scene)
        self.rclpy.spin_until_future_complete(self.node,f,timeout_sec=12.)
        check(f.done() and f.result() is not None and f.result().success,
              'MoveIt rejected a planning-scene update')

    def plan(self,name,desired,holding=False,released_position=None,support_contact=False):
        """Refresh static collision scene; MoveIt 2 PLAN ONLY, never execute."""
        from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
        self.check_fresh()
        start=self.state['joints']
        self.apply_scene(holding=holding,released_position=released_position,support_contact=support_contact)
        goal=self.m_move.Goal()
        req=goal.request
        req.group_name='arm';req.pipeline_id='ompl';req.planner_id='RRTConnectkConfigDefault'
        req.num_planning_attempts=6;req.allowed_planning_time=15.
        req.max_velocity_scaling_factor=.60
        req.max_acceleration_scaling_factor=.60
        req.start_state.is_diff=True
        constraints=Constraints()
        for n,q in zip(NAMES,desired):
            c=JointConstraint();c.joint_name=n;c.position=float(q)
            c.tolerance_above=.0015;c.tolerance_below=.0015;c.weight=1.
            constraints.joint_constraints.append(c)
        req.goal_constraints=[constraints]
        goal.planning_options.plan_only=True
        goal.planning_options.replan=False
        goal.planning_options.look_around=False
        goal.planning_options.planning_scene_diff.is_diff=True
        goal.planning_options.planning_scene_diff.robot_state.is_diff=True
        fut=self.planner.send_goal_async(goal)
        self.rclpy.spin_until_future_complete(self.node,fut,timeout_sec=12.)
        handle=fut.result() if fut.done() else None
        check(handle is not None and handle.accepted,f'[{name}] MoveGroup goal rejected')
        result=handle.get_result_async()
        self.rclpy.spin_until_future_complete(self.node,result,timeout_sec=40.)
        wrapped=result.result() if result.done() else None
        check(wrapped is not None,f'[{name}] MoveGroup timed out')
        res=wrapped.result
        check(res.error_code.val==MoveItErrorCodes.SUCCESS,f'[{name}] MoveIt plan failed: {res.error_code.val}')
        traj=res.planned_trajectory.joint_trajectory
        data={'success':True,'plan_only':True,'joint_names':list(traj.joint_names),
              'start_joints':start,'target_joints':list(map(float,desired)),
              'waypoints':[{'positions':list(pt.positions),
                            'seconds':float(pt.time_from_start.sec)+float(pt.time_from_start.nanosec)*1e-9}
                           for pt in traj.points],
              'planning_frame':self.model.root_frame,'obstacle_count':len(self.world),
              'holding_box':holding,'stage':name}
        vmax,amax=planned_limits_ok(data,start,desired,self.model)
        self.check_geometric_clearances(data,name)
        self.report_dir.mkdir(parents=True,exist_ok=True)
        out=self.report_dir/f'{name}.json'
        out.write_text(json.dumps(data,indent=2))
        print(f'PLAN {name}: {len(data["waypoints"])} points, '
              f'{data["waypoints"][-1]["seconds"]:.2f}s, vmax={vmax:.3f}, amax={amax:.3f}; saved {out}')
        self.last_plan=data
        return data

    def check_geometric_clearances(self,data,stage):
        # FK checks are supplementary to MoveIt's collision model. In
        # particular, they prevent a detouring planner from pushing the box
        # through the table when a *single* intended support-contact pair is
        # permitted during the first lift.
        green=self.world['upper_box'];orange=self.world['lower_box']
        green_top=float(green['center_world'][2]+green['size'][2]/2)
        orange_top=float(orange['center_world'][2]+orange['size'][2]/2)
        T0=self.model.T(data['waypoints'][0]['positions'])
        start_xyz=T0[:3,3]
        for pt in data['waypoints']:
            T=self.model.T(pt['positions'])
            xyz=T[:3,3]
            check(T[2,2]>.975, f'{stage}: suction cup tilts excessively along plan')
            if stage=='01_precontact':
                check(xyz[2]>green_top+.021,f'{stage}: TCP descended into contact zone')
            if stage=='02_seal':
                check(xyz[2]>green_top-.001,f'{stage}: TCP below top surface')
            if stage=='03_lift':
                check(xyz[2]>=start_xyz[2]-.004,
                      'Lift plan descends into pickup table while support contact allowed')
                check(np.linalg.norm(xyz[:2]-start_xyz[:2])<.025,
                      'Lift plan strays sideways while support contact allowed')
            if stage in ('04_transfer','05_place_near'):
                check(self.attach_tf is not None,'Cannot check attached box without grasp transform')
                box_T=T@self.attach_tf
                box_center=box_T[:3,3]
                half_height_world=float(np.sum(np.abs(box_T[2,:3])*np.array(green['size'])/2))
                check(box_center[2]-half_height_world>=orange_top+.005,
                      f'{stage}: carried box bottom approaches orange box closer than 5mm')

    def move(self,stage,plan,carrying=False):
        """Execute a *just-generated* plan via ros2_control; monitor live feedback."""
        from trajectory_msgs.msg import JointTrajectoryPoint
        from builtin_interfaces.msg import Duration
        check(plan is self.last_plan, 'Cannot replay nonfresh trajectory')
        check(time.monotonic()-self.state['joint_t']<.8,'No recent live feedback')
        check(carrying==bool(self.state['attached']), f'{stage}: unexpected attached state')
        check(self.vac_energized==carrying, f'{stage}: unexpected vacuum command state')
        current=self.check_fresh()
        require_start=np.array(plan['waypoints'][0]['positions'])
        check(float(np.max(np.abs(current-require_start)))<.02,f'{stage}: controller start mismatch')
        goal=self.m_arm.Goal()
        goal.trajectory.joint_names=list(NAMES)
        for p in plan['waypoints']:
            pp=JointTrajectoryPoint();pp.positions=list(map(float,p['positions']))
            nanos=round(p['seconds']*1e9)
            pp.time_from_start=Duration(sec=int(nanos//1_000_000_000),nanosec=int(nanos%1_000_000_000))
            goal.trajectory.points.append(pp)
        self.phase=stage
        print(f'EXECUTE {stage}: sending sim-only {len(goal.trajectory.points)}-point trajectory',flush=True)
        sent=self.controller.send_goal_async(goal)
        self.rclpy.spin_until_future_complete(self.node,sent,timeout_sec=10.)
        handle=sent.result() if sent.done() else None
        check(handle is not None and handle.accepted,f'{stage}: controller rejected trajectory')
        done=handle.get_result_async()
        deadline=time.monotonic()+plan['waypoints'][-1]['seconds']+18.
        problem=None
        try:
            while time.monotonic()<deadline and not done.done():
                self.spin(.1)
                try:
                    self.check_fresh()
                    if carrying and self.state['attached'] is not True:
                        problem='Unexpected drop of green box while arm in motion'
                        break
                except Blocked as exc:
                    problem=str(exc)
                    break
        except KeyboardInterrupt:
            problem='operator interrupted motion'
        if problem or not done.done():
            cancel=handle.cancel_goal_async()
            self.rclpy.spin_until_future_complete(self.node,cancel,timeout_sec=5.)
            raise Blocked(f'{stage}: cancelled trajectory: {problem or "timeout"}')
        result=done.result()
        check(result is not None and result.result.error_code==0,
              f'{stage}: controller error: {None if result is None else result.result.error_code}')
        self.spin(.35)
        current=self.check_fresh()
        err=float(np.max(np.abs(current-np.array(plan['waypoints'][-1]['positions']))))
        check(err<.02,f'{stage}: measured terminal error {err:.4f}rad >0.02rad')
        print(f'PASS {stage}: executed, controller endpoint tracking error={err:.4f} rad')

    def close(self):
        if self.trace:
            self.report_dir.mkdir(parents=True,exist_ok=True)
            logfile=self.report_dir/'run_trace.jsonl'
            logfile.write_text(''.join(json.dumps(row)+'\n' for row in self.trace))
            print('Saved ROS state/diagnostic trace:',logfile)
        if self.vac_energized:
            print('SAFETY: vacuum was ON at process exit; it is LEFT ON to avoid dropping a carried box.')
            print('Ensure box has a supported landing before manually disabling the vacuum.')
        self.node.destroy_node()
        self.rclpy.shutdown()


def make_box_pose(center):
    T=np.eye(4)
    T[:3,3]=center
    return T


def stacking_targets(world):
    green=world['upper_box'];orange=world['lower_box']
    # Collision primitive boxes are assumed unrotated, matching parser's SDF check.
    gtop=green['center_world'][2]+green['size'][2]/2
    otop=orange['center_world'][2]+orange['size'][2]/2
    gheight=green['size'][2]
    return {
        'pickup_pre':np.array([*green['center_world'][:2],gtop+.035]),
        'pickup_seal':np.array([*green['center_world'][:2],gtop+.004]),
        'place_hover':np.array([*orange['center_world'][:2],otop+gheight+.004+.150]),
        'place_near':np.array([*orange['center_world'][:2],otop+gheight+.004+.010]),
        'green_center':green['center_world'],
        'orange_top':otop,
    }


def go_stage(station, model, stage, target, orient, holding, execute, released_position=None, support_contact=False):
    start=station.check_fresh()
    q=model.solve_near(start,target,orient)
    tip=model.T(q)[:3,3]
    down=float(model.T(q)[2,2])  # TCP local -Z points downward; TCP local +Z points upward
    check(down>.99,f'{stage}: suction no longer vertical ({down:.4f})')
    print(f'IK {stage}: target={np.round(target,4)}, error={np.linalg.norm(tip-target):.4f} m, down={down:.4f}')
    plan=station.plan(stage,q,holding=holding,released_position=released_position,support_contact=support_contact)
    if execute:station.move(stage,plan,carrying=holding)
    return q


def verify_pose_snapshot(path, world_path):
    check(path.is_file(), f'Missing fresh Gazebo pose check: {path}. Run check_gz_box_pose_v7.py in sim container')
    report=json.loads(path.read_text())
    check(report.get('all_ok') is True,'Gazebo box-pose check failed')
    check(0 <= time.time()-report['timestamp_wall_s'] < 60,
          'Gazebo pose snapshot is older than 60 seconds; rerun the check')
    check(report['world_sha256']==hashlib.sha256(Path(world_path).read_bytes()).hexdigest(),
          'SDF changed since the Gazebo pose check')
    check(all(report['boxes'][name]['ok'] is True for name in ('upper_box','lower_box')),
          'Gazebo pose check incomplete or invalid')
    print('PASS: fresh Gazebo green/orange box positions verified against SDF snapshot')


def run(args):
    model=RobotKinematics(args.urdf)
    world=world_objects(args.world)
    targets=stacking_targets(world)
    print('Gazebo-only UF850 station, root frame:',model.root_frame)
    print('Green box top / orange box top (world z):',
          round(targets['pickup_seal'][2]-.004,4),round(targets['orange_top'],4))
    print('Planned seal point XYZ:',np.round(targets['pickup_seal'],4).tolist())
    print('Planned place target XYZ:',np.round(targets['place_near'],4).tolist())
    if args.mode=='inspect':
        print('READ-ONLY INSPECTION COMPLETE: no ROS commands or plans submitted')
        return
    check(os.getenv('ROS_DOMAIN_ID')=='42','Refusing to use a ROS graph other than isolated sim domain 42')
    if args.mode=='execute-stack':
        check(args.ack_simulation_only and args.ack_conveyor_stopped and args.ack_scene_verified
              and args.ack_robot_hover_verified,
              'Execution requires all four explicit --ack-* flags')
        verify_pose_snapshot(args.pose_snapshot,args.world)
    station=SimStation(model,world,Path(args.reports))
    try:
        station.ready()
        check(station.state['attached'] is False,'Upper box is already attached')
        check('a=0,b=0' in station.state['diag'].replace(' ',''),'Vacuum must start OFF')
        check(not station.vac_energized,'Internal vacuum state inconsistent')
        hover=json.loads(args.hover_plan.read_text())
        expected=np.array(hover['waypoints'][-1]['positions'])
        actual=station.check_fresh()
        check(np.max(np.abs(actual-expected))<.04,
              'Robot must first be at *executed* hover endpoint (<0.04rad); run the hover guard')
        actual_T=model.T(actual)
        hover_xyz=actual_T[:3,3]
        check(hover_xyz[2]>(world['upper_box']['center_world'][2]+world['upper_box']['size'][2]/2)+.07,
              'Hover is not at least 7cm above box top')
        check(float(np.linalg.norm(hover_xyz[:2]-targets['pickup_seal'][:2]))<.04,
              f'Hover TCP is not centered above green box (delta={hover_xyz[:2]-targets["pickup_seal"][:2]})')
        orient=actual_T[:3,:3]
        check(orient[2,2]>.99,'Hover suction must point down')
        check(float(np.max(np.abs(actual_T[:3,3]-np.array([-.55,-.55,.97]))))<.04,
              'Robot FK vs expected hover world pose differs >4cm; check spawn / TCP assumptions')
        print('PASS: robot is at verified hover pose in world XYZ:',np.round(hover_xyz,4).tolist())
        execute=(args.mode=='execute-stack')
        go_stage(station,model,'01_precontact',targets['pickup_pre'],orient,False,execute)
        if not execute:
            print('PLAN-ONLY CHECK COMPLETE. No robot action or vacuum command sent.')
            print('The subsequent contact, lift, transfer and release plans require actual updated joint states; '
                  'run --mode execute-stack only after checking the first plan and sim-only safeguards.')
            return
        go_stage(station,model,'02_seal',targets['pickup_seal'],orient,False,True)
        station.spin(.3)
        station.check_seal()
        contact_T=model.T(station.state['joints'])
        green_pose=make_box_pose(world['upper_box']['center_world'])
        station.attach_tf=np.linalg.inv(contact_T)@green_pose
        station.vacuum(True,seconds=.75)
        until=time.monotonic()+4
        while station.state['attached'] is not True and time.monotonic()<until:
            station.vacuum(True,seconds=.2)
        check(station.state['attached'] is True,'No Gazebo attachment; abort before lifting')
        print('PASS: Gazebo reports VGC10 fixed-joint attachment')
        # Recalculate desired *world* TCP from the actual contact pose, not nominal IK target.
        current_T=model.T(station.check_fresh())
        go_stage(station,model,'03_lift',current_T[:3,3]+np.array([0,0,.15]),
                 current_T[:3,:3],True,True,support_contact=True)
        current_T=model.T(station.check_fresh())
        go_stage(station,model,'04_transfer',targets['place_hover'],current_T[:3,:3],True,True)
        current_T=model.T(station.check_fresh())
        go_stage(station,model,'05_place_near',targets['place_near'],current_T[:3,:3],True,True)
        # The orange box must remain stationary at its initial SDF pose; outside
        # this explicitly stationary trial that assumption is unsafe.
        station.vacuum(False,seconds=.5)
        station.spin(.6)
        check(station.state['attached'] is False,'Object has not detached after vacuum OFF')
        print('PASS: vacuum OFF; Gazebo plugin reports green box detached')
        # Approximate released collision pose until the box settles. No claims of
        # stable stack without Gazebo world-pose measurement or camera inspection.
        release_tcp=model.T(station.check_fresh())
        placed_center=(release_tcp@station.attach_tf)[:3,3]
        retreat=release_tcp[:3,3]+np.array([0,0,.15])
        go_stage(station,model,'06_retreat',retreat,release_tcp[:3,:3],False,True,
                 released_position=placed_center)
        station.attach_tf=None
        print('STATIONARY STACK MOTION SEQUENCE COMPLETED (SIMULATION).')
        print('VERIFY PHYSICS: inspect /simulation/upper_box/pose and camera image to ensure stable stacking.')
        print('Moving conveyor, visual occlusion, and belief-VLA policy are NOT tested by this sequence.')
    finally:
        station.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=('inspect','plan-contact','execute-stack'),default='inspect')
    ap.add_argument('--urdf',type=Path,default=URDF_PATH)
    ap.add_argument('--world',type=Path,default=WORLD_PATH)
    ap.add_argument('--hover-plan',type=Path,default=Path('/workspace/data/uf850_hover_plan.json'))
    ap.add_argument('--pose-snapshot',type=Path,default=Path('/workspace/data/uf850_gz_pose_snapshot.json'))
    ap.add_argument('--reports',default='/workspace/data/uf850_stack_v7')
    ap.add_argument('--ack-simulation-only',action='store_true')
    ap.add_argument('--ack-conveyor-stopped',action='store_true')
    ap.add_argument('--ack-scene-verified',action='store_true')
    ap.add_argument('--ack-robot-hover-verified',action='store_true')
    args=ap.parse_args()
    try:
        run(args)
    except (Blocked,ValueError,OSError,KeyError,ArithmeticError,KeyboardInterrupt) as exc:
        print('BLOCKED:',exc,file=sys.stderr)
        return 2
    return 0


if __name__=='__main__':
    sys.exit(main())
