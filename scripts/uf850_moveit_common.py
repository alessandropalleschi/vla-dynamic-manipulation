#!/usr/bin/env python3
"""Small, conservative MoveIt config helpers for a *simulated* UF850+VGC10.

No real-robot connections or motion commands. Generated configuration is a
starting point, not a manufacturer-validated SRDF/self-collision matrix.
"""
from pathlib import Path
import math
import xml.etree.ElementTree as ET

JOINT_NAMES = [f'joint{i}' for i in range(1, 7)]
DEFAULT_CANDIDATE = [3.2824, -0.4529, -0.9352, 0.1693, -0.4516, 5.5367]
SPAWN_WORLD = (-0.05, -0.47, 0.78)
URDF_PATH = Path('/workspace/simulation/generated/uf850_vgc10.urdf')
WORLD_PATH = Path('/workspace/simulation/worlds/conveyor_station_uf850.sdf')


def parse_robot(path):
    path = Path(path)
    root = ET.parse(path).getroot()
    if root.tag != 'robot':
        raise ValueError(f'{path} is not a URDF <robot>')
    links = {link.get('name') for link in root.findall('link')}
    joints = {joint.get('name'): joint for joint in root.findall('joint')}
    absent = [n for n in JOINT_NAMES if n not in joints]
    if absent:
        raise ValueError(f'Missing arm joints {absent}')
    if 'vgc10_tcp' not in links:
        raise ValueError('Missing vgc10_tcp link; regenerate your corrected VGC10 URDF')
    children = {j.find('child').get('link') for j in joints.values() if j.find('child') is not None}
    roots = sorted(links - children)
    if len(roots) != 1:
        raise ValueError(f'URDF does not have one root link: {roots}')
    limits = {}
    for name in JOINT_NAMES:
        joint = joints[name]
        node = joint.find('limit')
        if joint.get('type') == 'continuous':
            # A continuous revolute joint can be represented periodically;
            # collision and cable-wrap constraints still need checking.
            limits[name] = (-math.inf, math.inf, float(node.get('velocity', '1')) if node is not None else 1.0)
        elif node is not None and node.get('lower') is not None and node.get('upper') is not None:
            limits[name] = (float(node.get('lower')), float(node.get('upper')), float(node.get('velocity', '1')))
        else:
            raise ValueError(f'{name} lacks explicit limits')
    return root, roots[0], joints, limits


def feasible_equivalent(angle, lo, hi, reference=0.0):
    """Find an *equivalent* angle in URDF limits; do not clamp a goal silently."""
    candidates = [angle + 2 * math.pi * k for k in range(-3, 4)]
    candidates = [x for x in candidates if lo - 1e-8 <= x <= hi + 1e-8]
    return min(candidates, key=lambda x: abs(x - reference)) if candidates else None


def normalized_candidate(raw, limits, reference=None):
    ref = reference or [0.0] * 6
    chosen = []
    errors = []
    for name, value, around in zip(JOINT_NAMES, raw, ref):
        lo, hi, _ = limits[name]
        equiv = feasible_equivalent(value, lo, hi, around)
        if equiv is None:
            errors.append(f'{name}: {value:+.4f} outside [{lo:+.4f}, {hi:+.4f}] with no allowed 2pi equivalent')
        else:
            chosen.append(equiv)
    return (chosen if not errors else None), errors


def make_srdf(urdf_root, joints):
    srdf = ET.Element('robot', {'name': urdf_root.get('name', 'uf850')})
    arm = ET.SubElement(srdf, 'group', {'name': 'arm'})
    first_parent = joints['joint1'].find('parent').get('link')
    # A chain is required by the KDL plugin; it includes the fixed VGC10 TCP.
    ET.SubElement(arm, 'chain', {'base_link': first_parent, 'tip_link': 'vgc10_tcp'})
    # Only neighboring links are disabled by default. Do NOT disable nonadjacent
    # collisions, and review any false positives using MoveIt Setup Assistant.
    seen = set()
    for joint in joints.values():
        p, c = joint.find('parent'), joint.find('child')
        if p is None or c is None:
            continue
        a, b = p.get('link'), c.get('link')
        key = tuple(sorted((a, b)))
        if a != b and key not in seen:
            seen.add(key)
            ET.SubElement(srdf, 'disable_collisions', {'link1': a, 'link2': b, 'reason': 'Adjacent'})
    ET.indent(srdf, space='  ')
    return ET.tostring(srdf, encoding='unicode')


def make_moveit_parameters(urdf_text, srdf_text, limits):
    joint_limits = {}
    for name in JOINT_NAMES:
        lo, hi, velocity = limits[name]
        joint_limits[name] = {
            'has_velocity_limits': True,
            'max_velocity': min(max(velocity, 0.05), 1.0),  # deliberately conservative, sim only
            'has_acceleration_limits': True,
            'max_acceleration': 1.0,
        }
    return {
        'use_sim_time': True,
        'robot_description': urdf_text,
        'robot_description_semantic': srdf_text,
        'robot_description_kinematics': {
            'arm': {
                'kinematics_solver': 'kdl_kinematics_plugin/KDLKinematicsPlugin',
                'kinematics_solver_search_resolution': 0.005,
                'kinematics_solver_timeout': 0.2,
            }
        },
        'robot_description_planning': {'joint_limits': joint_limits},
        'planning_pipelines': ['ompl'],
        'default_planning_pipeline': 'ompl',
        'ompl': {
            'planning_plugins': ['ompl_interface/OMPLPlanner'],
            'request_adapters': [
                'default_planning_request_adapters/ResolveConstraintFrames',
                'default_planning_request_adapters/ValidateWorkspaceBounds',
                'default_planning_request_adapters/CheckStartStateBounds',
                'default_planning_request_adapters/CheckStartStateCollision',
            ],
            'response_adapters': [
                'default_planning_response_adapters/AddTimeOptimalParameterization',
                'default_planning_response_adapters/ValidateSolution',
                'default_planning_response_adapters/DisplayMotionPath',
            ],
            'planner_configs': {
                'RRTConnectkConfigDefault': {'type': 'geometric::RRTConnect', 'range': 0.0},
            },
            'arm': {'planner_configs': ['RRTConnectkConfigDefault']},
        },
        'allow_trajectory_execution': False,
        'moveit_manage_controllers': False,
        'planning_scene_monitor': {
            'publish_planning_scene': True,
            'publish_geometry_updates': True,
            'publish_state_updates': True,
            'publish_transforms_updates': True,
        },
    }


def parse_sdf_obstacles(path, spawn=SPAWN_WORLD):
    """Static snapshot from SDF; positions expressed in the robot's local planning root.

    Current scene uses axis-aligned boxes with model/link/collision zero RPY.
    Fail closed on rotations so we don't invent a rotated obstacle.
    """
    root = ET.parse(path).getroot()
    world = root.find('world')
    wanted = {'conveyor', 'pickup_table', 'uf850_pedestal', 'lower_box', 'upper_box'}
    found = []
    def pose6(element):
        p = element.find('pose')
        vals = [float(i) for i in p.text.split()] if p is not None and p.text else [0.0]*6
        if len(vals) != 6:
            raise ValueError('Unsupported SDF pose syntax: expected 6 floats')
        if any(abs(v) > 1e-6 for v in vals[3:]):
            raise ValueError('Obstacle rotated: implement full SDF pose composition before planning')
        return vals[:3]
    for model in world.findall('model'):
        name = model.get('name')
        if name not in wanted:
            continue
        pos_model = pose6(model)
        for link in model.findall('link'):
            pos_link = pose6(link)
            for collision in link.findall('collision'):
                box = collision.find('./geometry/box/size')
                if box is None:
                    continue
                pos_coll = pose6(collision)
                xyz = [a+b+c-s for a,b,c,s in zip(pos_model, pos_link, pos_coll, spawn)]
                size = [float(x) for x in box.text.split()]
                if len(size)!=3 or min(size)<=0:
                    raise ValueError(f'Invalid collision box in {name}')
                found.append(dict(name=f'{name}_{link.get("name")}_{collision.get("name")}', center=xyz, size=size))
    for required in ('conveyor', 'pickup_table', 'uf850_pedestal'):
        if not any(o['name'].startswith(required+'_') for o in found):
            raise ValueError(f'Missing essential planning obstacle: {required}')
    return found
