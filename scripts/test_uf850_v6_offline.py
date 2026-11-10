#!/usr/bin/env python3
"""Run without ROS or Gazebo: verify v6 config logic on synthetic test fixtures."""
from pathlib import Path
import tempfile
import math
import xml.etree.ElementTree as ET
from uf850_moveit_common import (parse_robot, make_srdf, make_moveit_parameters,
    normalized_candidate, DEFAULT_CANDIDATE, parse_sdf_obstacles)


def main():
    root = ET.Element('robot', name='toy_uf850')
    ET.SubElement(root, 'link', name='world')
    parent='world'
    for i in range(1,7):
        child=f'link{i}'
        ET.SubElement(root, 'link', name=child)
        joint=ET.SubElement(root, 'joint', name=f'joint{i}', type='revolute')
        ET.SubElement(joint, 'parent', link=parent)
        ET.SubElement(joint, 'child', link=child)
        ET.SubElement(joint, 'origin', xyz='0 0 0',rpy='0 0 0')
        ET.SubElement(joint, 'axis', xyz='0 0 1')
        ET.SubElement(joint, 'limit', lower='-3.2',upper='3.2',velocity='1',effort='50')
        parent=child
    ET.SubElement(root,'link',name='vgc10_tcp')
    fixed=ET.SubElement(root,'joint',name='vgc10_tcp_fixed',type='fixed')
    ET.SubElement(fixed,'parent',link='link6')
    ET.SubElement(fixed,'child',link='vgc10_tcp')
    ET.SubElement(fixed,'origin',xyz='0 0 0.127',rpy=f'{math.pi} 0 0')
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'robot.urdf'
        ET.ElementTree(root).write(path)
        parsed,base,joints,limits=parse_robot(path)
        assert base=='world'
        srdf=make_srdf(parsed,joints)
        assert 'tip_link="vgc10_tcp"' in srdf
        assert srdf.count('<disable_collisions')==7
        config=make_moveit_parameters(path.read_text(),srdf,limits)
        assert config['allow_trajectory_execution'] is False
        goal,errors=normalized_candidate(DEFAULT_CANDIDATE,limits)
        assert not errors and len(goal)==6
        assert abs(goal[0]-(DEFAULT_CANDIDATE[0]-2*math.pi))<1e-8
        assert abs(goal[5]-(DEFAULT_CANDIDATE[5]-2*math.pi))<1e-8
        limits['joint1']=(-2.8,2.8,1)
        goal,errors=normalized_candidate(DEFAULT_CANDIDATE,limits)
        assert goal is None and errors
        # Minimal SDF fixture: do not depend on any downloaded ZIPs.
        world=ET.fromstring('''<sdf version="1.9"><world name="conveyor_station">
          <model name="conveyor"><pose>0 0 0.75 0 0 0</pose><link name="belt_link">
            <collision name="belt_collision"><geometry><box><size>1.65 0.34 0.08</size>
            </box></geometry></collision></link></model>
          <model name="pickup_table"><pose>-0.55 -0.55 0.73 0 0 0</pose>
            <link name="table_link"><collision name="table_collision"><geometry><box>
            <size>0.38 0.36 0.12</size></box></geometry></collision></link></model>
          <model name="uf850_pedestal"><pose>-0.05 -0.47 0.39 0 0 0</pose>
            <link name="pedestal_link"><collision name="pedestal_collision"><geometry>
            <box><size>0.22 0.22 0.78</size></box></geometry></collision></link></model>
          <model name="lower_box"><pose>-0.52 0 0.83 0 0 0</pose><link name="box_link">
            <collision name="collision"><geometry><box><size>0.12 0.12 0.08</size>
            </box></geometry></collision></link></model>
          <model name="upper_box"><pose>-0.55 -0.55 0.82 0 0 0</pose><link name="box_link">
            <collision name="collision"><geometry><box><size>0.09 0.09 0.06</size>
            </box></geometry></collision></link></model>
        </world></sdf>''')
        world_file=Path(tmp)/'uf850.sdf'
        ET.ElementTree(world).write(world_file)
        obstacles=parse_sdf_obstacles(world_file)
        assert len(obstacles)==5
        table=next(x for x in obstacles if x['name'].startswith('pickup_table'))
        assert abs(table['center'][0]+.5)<1e-9
        assert abs(table['center'][1]+.08)<1e-9
    print('PASS v6 offline: limits, equivalent angles, semantic chain, plan-only policy, SDF obstacles')

if __name__=='__main__':
    main()
