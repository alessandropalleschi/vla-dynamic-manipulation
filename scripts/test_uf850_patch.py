#!/usr/bin/env python3
"""Offline static checks — do not claim this simulates actual robot physics."""
from pathlib import Path
import tempfile
import sys
import xml.etree.ElementTree as ET

from prepare_uf850_model import configure
from make_uf850_world import generate


def test():
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        arm = ET.Element('robot', name='uf850_test')
        ET.SubElement(arm, 'link', name='base_link')
        for i in range(1, 7):
            ET.SubElement(arm, 'link', name=f'link{i}')
            joint = ET.SubElement(arm, 'joint', name=f'joint{i}', type='revolute')
            ET.SubElement(joint, 'parent', link='base_link' if i == 1 else f'link{i-1}')
            ET.SubElement(joint, 'child', link=f'link{i}')
        old_gazebo = ET.SubElement(arm, 'gazebo')
        ET.SubElement(old_gazebo, 'plugin', filename='libgazebo_ros2_control.so', name='classic')
        old_hardware = ET.SubElement(arm, 'ros2_control', name='RealHardware', type='system')
        ET.SubElement(ET.SubElement(old_hardware, 'hardware'), 'plugin').text = 'uf_robot_hardware/UFRobotSystemHardware'
        src = temp/'raw.urdf'
        dest = temp/'out.urdf'
        ET.ElementTree(arm).write(src, encoding='unicode')
        configure(src, dest)
        root = ET.parse(dest).getroot()
        assert len(root.findall('link')) == 1 + 6 + 6  # base + 6 arm + body, four cups, tcp
        assert len(root.findall('gazebo/plugin')) == 1
        assert root.find('gazebo/plugin').get('filename') == 'libgz_ros2_control-system.so'
        assert root.find('ros2_control/hardware/plugin').text == 'gz_ros2_control/GazeboSimSystem'
        assert set(f'vgc10_cup_{i}' for i in ['a1','a2','b1','b2']).issubset(
            {el.get('name') for el in root.findall('link')})
        cup_positions = [el.find('origin').get('xyz') for el in root.findall('joint') if (el.get('name') or '').startswith('vgc10_cup_')]
        assert len(set(cup_positions)) == 4
        print('PASS: UF850 + VGC10 generated URDF composition (synthetic input)')

    # World generation with a minimal, constructed source
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        sdf = ET.Element('sdf', version='1.9')
        world = ET.SubElement(sdf, 'world', name='conveyor_station')
        ET.SubElement(world, 'model', name='gantry')
        ET.SubElement(world, 'model', name='conveyor')
        ET.SubElement(world, 'model', name='external_camera')
        src = temp/'world.sdf'
        dst = temp/'robot.sdf'
        ET.ElementTree(sdf).write(src)
        generate(src,dst)
        new_world = ET.parse(dst).getroot()
        assert new_world.find(".//model[@name='gantry']") is None
        for name in ['conveyor','external_camera','uf850_pedestal']:
            assert new_world.find(f".//model[@name='{name}']") is not None
        print('PASS: UF850 world derived without overwriting baseline (synthetic input)')


if __name__ == '__main__':
    try:
        test()
    except Exception as e:
        print(f'FAIL: {e}', file=sys.stderr)
        raise
