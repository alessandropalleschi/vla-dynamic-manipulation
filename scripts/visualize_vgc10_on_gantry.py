#!/usr/bin/env python3
"""Replace old two-cup gantry visuals with a four-cup 2x2 VGC10 proxy.

VISUAL ONLY. Keep the original working gantry's collision, suction attachment,
robot joint constraints, controllers and physical contact unchanged.
Use this camera-data surrogate while the real UF850 model is being integrated.
"""
import argparse
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def visual(name, kind, size, pose, color):
    obj = ET.Element('visual', name=name)
    ET.SubElement(obj, 'pose').text = pose
    ge = ET.SubElement(obj, 'geometry')
    if kind == 'box':
        ET.SubElement(ET.SubElement(ge, 'box'), 'size').text = size
    elif kind == 'cylinder':
        radius, length = size
        cyl = ET.SubElement(ge, 'cylinder')
        ET.SubElement(cyl, 'radius').text = f'{radius:.5f}'
        ET.SubElement(cyl, 'length').text = f'{length:.4f}'
    mat = ET.SubElement(obj, 'material')
    ET.SubElement(mat, 'ambient').text = color
    ET.SubElement(mat, 'diffuse').text = color
    return obj


def transform(world_file, cup_mm=30, pitch_x_mm=50, pitch_y_mm=50,
              channel_axis='y'):
    if cup_mm not in (15, 30, 40):
        raise ValueError('Cup size must be 15, 30, or 40 mm')
    if pitch_x_mm < cup_mm or pitch_y_mm < cup_mm:
        raise ValueError('Cup pitch must be >= cup diameter')
    if channel_axis not in ('x', 'y'):
        raise ValueError('channel_axis must be x or y')
    tree = ET.parse(world_file)
    root = tree.getroot()
    link = root.find(".//model[@name='gantry']/link[@name='z_carriage']")
    if link is None:
        raise ValueError('Gantry z_carriage not found; run only on baseline world')
    # Remove both old two-cup and any prior four-cup proxy visuals; idempotent.
    for v in list(link.findall('visual')):
        name = v.get('name', '')
        if name in ('column', 'suction_tip', 'vgc10_body_visual') or name.startswith('vgc10_cup_'):
            link.remove(v)
    link.append(visual('vgc10_body_visual', 'box', '0.101 0.100 0.100',
                       '0 0 -0.119 0 0 0', '0.17 0.20 0.23 1'))
    px = pitch_x_mm / 2000
    py = pitch_y_mm / 2000
    for x_sign in (-1, 1):
        for y_sign in (-1, 1):
            if channel_axis == 'y':
                channel = 'a' if y_sign > 0 else 'b'
                col = 1 if x_sign < 0 else 2
            else:
                channel = 'a' if x_sign < 0 else 'b'
                col = 1 if y_sign > 0 else 2
            cid = f'{channel}{col}'
            link.append(visual(f'vgc10_cup_{cid}_visual', 'cylinder',
                               (cup_mm/2000, 0.012),
                               f'{x_sign*px:.6f} {y_sign*py:.6f} -0.175 0 0 0',
                               '0.06 0.06 0.065 1' if channel == 'a' else '0.12 0.12 0.13 1'))
    ET.indent(tree, space='  ')
    tree.write(world_file, encoding='utf-8', xml_declaration=True)
    print(f'Updated 4-cup 2x2 VGC10 visual proxy: {world_file}')
    print('Physics/controllers unchanged. This is still a gantry, NOT the UF850.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--world', type=Path, default=Path('simulation/worlds/conveyor_station.sdf'))
    p.add_argument('--cup-mm', type=int, choices=(15, 30, 40), default=30)
    p.add_argument('--pitch-x-mm', type=float, default=50)
    p.add_argument('--pitch-y-mm', type=float, default=50)
    p.add_argument('--channel-axis', choices=('x', 'y'), default='y')
    a = p.parse_args()
    backup = a.world.with_suffix(a.world.suffix+'.pre_vgc10_4cup_visual.bak')
    if not backup.exists():
        shutil.copy2(a.world, backup)
        print(f'Backup: {backup}')
    transform(a.world, a.cup_mm, a.pitch_x_mm, a.pitch_y_mm, a.channel_axis)

if __name__ == '__main__':
    main()
