#!/usr/bin/env python3
"""Generate a *separate* UF850 conveyor station world from the working scene.

Never edits the existing baseline SDF. UF850 is spawned separately from a URDF.
"""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def generate(src: Path, dst: Path):
    if src.resolve() == dst.resolve():
        raise ValueError('Input/output must differ')
    tree = ET.parse(src)
    world = tree.getroot().find('world')
    if world is None:
        raise ValueError('No SDF world element')
    gantry = world.find("model[@name='gantry']")
    if gantry is None:
        raise ValueError('Gantry absent from source world: refusing to generate')
    world.remove(gantry)
    # Accessible from both the pickup table and moving conveyor (approx. 0.5-0.7m planar range).
    pedestal = ET.fromstring('''<model name="uf850_pedestal">
      <static>true</static><pose>-0.05 -0.47 0.39 0 0 0</pose>
      <link name="pedestal_link">
       <collision name="pedestal_collision"><geometry><box><size>0.22 0.22 0.78</size></box></geometry></collision>
       <visual name="pedestal_visual"><geometry><box><size>0.22 0.22 0.78</size></box></geometry>
        <material><ambient>0.35 0.38 0.42 1</ambient><diffuse>0.35 0.38 0.42 1</diffuse></material>
       </visual>
      </link>
    </model>''')
    world.append(pedestal)
    ET.indent(tree, space='  ')
    dst.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst, encoding='utf-8', xml_declaration=True)
    print('Generated UF850 world:', dst)
    print('Robot base spawn: x=-0.05 y=-0.47 z=0.78; camera+conveyor retained')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path('simulation/worlds/conveyor_station.sdf'))
    p.add_argument('--output', type=Path, default=Path('simulation/worlds/conveyor_station_uf850.sdf'))
    args = p.parse_args()
    generate(args.input, args.output)


if __name__ == '__main__':
    main()
