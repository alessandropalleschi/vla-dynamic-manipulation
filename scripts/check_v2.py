#!/usr/bin/env python3
"""Sanity-check world, bridge and patch before running a costly batch."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET

base = Path(__file__).resolve().parent.parent
world = base/'simulation/worlds/conveyor_station.sdf'
root = ET.parse(world).getroot()
zj = root.find(".//model[@name='gantry']/joint[@name='z_joint']/axis/limit/lower")
if zj is None:
    raise SystemExit('FAIL: z_joint not found in Gazebo world')
if float(zj.text) >= 0:
    raise SystemExit('FAIL: z_joint lower limit is still >= 0. This caused the attachment-related lift lock. Apply the verified -0.03 fix before collecting.')
cam = root.find(".//model[@name='external_camera']/link/sensor[@name='rgb']")
if cam is None:
    raise SystemExit('FAIL: external RGB sensor missing')
bridge = (base/'config/bridge.yaml').read_text()
for topic in ('/simulation/lower_box/pose','/simulation/upper_box/pose', '/joint_states'):
    if topic not in bridge:
        raise SystemExit(f'FAIL: {topic} missing from bridge.yaml')
for path in ('collect_episode.py','evaluate_episode.py','encode_frames.py','run_expert.py'):
    if not (base/'scripts'/path).exists():
        raise SystemExit(f'FAIL: scripts/{path} not installed')
print(f'PASS: world present, z_joint lower limit {zj.text}, camera and bridge configured')
print('NOTE: Static check only; physical Gazebo behavior must be validated on your machine.')
