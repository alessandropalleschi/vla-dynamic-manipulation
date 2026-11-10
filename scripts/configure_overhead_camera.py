#!/usr/bin/env python3
"""Change only camera fields in the existing Gazebo world; keep robot/joint edits.

Gazebo camera +X is forward and +Z points upward. Pitch +pi/2 makes it
point down. The projected image vertical direction is parallel to belt +X.
"""
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def configure(src, backup=True):
    tree = ET.parse(src)
    root = tree.getroot()
    model = root.find(".//world/model[@name='external_camera']")
    if model is None:
        raise ValueError('external_camera model not found; refusing to change world')
    pose = model.find('pose')
    if pose is None:
        raise ValueError('camera world pose missing')
    # Centered over pickup (y=-.55) and belt (y=0); +X follows conveyor.
    pose.text = '-0.12 -0.25 2.20 0 0 0'
    sensor = model.find('.//sensor[@name="rgb"]')
    if sensor is None:
        raise ValueError('rgb sensor not found')
    sp = sensor.find('pose')
    if sp is None:
        sp = ET.SubElement(sensor, 'pose')
    sp.text = '0 0 0 0 1.57079632679 0'
    # Wide enough to see full working areas, but retain current resolution/ROS topic.
    fov = sensor.find('./camera/horizontal_fov')
    if fov is None:
        raise ValueError('camera horizontal_fov not found')
    fov.text = '1.48'
    if backup:
        dest = src.with_suffix(src.suffix + '.pre_overhead.bak')
        if not dest.exists():
            shutil.copy2(src, dest)
            print(f'Backup: {dest}')
    ET.indent(tree, space='  ')
    tree.write(src, encoding='utf-8', xml_declaration=True)
    print('Updated camera to nadir view, oriented along belt +X:', src)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--world', type=Path, default=Path('simulation/worlds/conveyor_station.sdf'))
    args = p.parse_args()
    configure(args.world)


if __name__ == '__main__':
    main()
