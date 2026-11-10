#!/usr/bin/env python3
"""Attach a visual/collision VGC10 4-cup 2x2 proxy to a *generated* UF850 URDF.

Two independently addressable channels (A/B) are labels only: this script does NOT
simulate vacuum pressure, seals, attachment, air routing or a real gripper driver.
The cup pitch, diameter, robot flange transform and channel routing must be checked
against the physical VGC10 before any real-robot work.
"""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

BODY_X, BODY_Y, BODY_Z = 0.101, 0.100, 0.100
NOMINAL_TOTAL_MASS_KG = 0.814
CUP_PROXY_MASS_KG = 0.010  # per cup; invented proxy, not manufacturer's spec
CUP_LENGTH_M = 0.012


def element(parent, tag, **attrs):
    return ET.SubElement(parent, tag, attrs)


def add_joint(robot, name, parent, child, xyz, rpy='0 0 0'):
    joint = element(robot, 'joint', name=name, type='fixed')
    element(joint, 'parent', link=parent)
    element(joint, 'child', link=child)
    element(joint, 'origin', xyz=xyz, rpy=rpy)
    return joint


def geometry(link, tag, shape, size, xyz='0 0 0', rgba=None):
    geom_part = element(link, tag)
    element(geom_part, 'origin', xyz=xyz, rpy='0 0 0')
    shape_el = element(geom_part, 'geometry')
    if shape == 'box':
        element(shape_el, 'box', size=size)
    elif shape == 'cylinder':
        radius, length = size.split()
        element(shape_el, 'cylinder', radius=radius, length=length)
    else:
        raise ValueError(f'Unsupported shape: {shape}')
    if rgba and tag == 'visual':
        mat = element(geom_part, 'material', name='vgc10_' + rgba.replace(' ', '_'))
        element(mat, 'color', rgba=rgba)


def inertial_box(link, mass, sx, sy, sz, xyz):
    i = element(link, 'inertial')
    element(i, 'origin', xyz=xyz, rpy='0 0 0')
    element(i, 'mass', value=f'{mass:.6f}')
    element(i, 'inertia',
            ixx=f'{mass*(sy**2+sz**2)/12:.9f}', ixy='0', ixz='0',
            iyy=f'{mass*(sx**2+sz**2)/12:.9f}', iyz='0',
            izz=f'{mass*(sx**2+sy**2)/12:.9f}')


def inertial_cylinder(link, mass, radius, length):
    i = element(link, 'inertial')
    element(i, 'origin', xyz='0 0 -0.006', rpy='0 0 0')
    element(i, 'mass', value=f'{mass:.6f}')
    ixy = mass * (3*radius*radius+length*length) / 12
    izz = mass * radius*radius / 2
    element(i, 'inertia', ixx=f'{ixy:.9f}', ixy='0', ixz='0',
            iyy=f'{ixy:.9f}', iyz='0', izz=f'{izz:.9f}')


def cup_locations(pitch_x_mm=44, pitch_y_mm=44, channel_axis='y'):
    """Return (cup_id, x, y, channel); defaults to two cups per Y row."""
    if channel_axis not in ('x', 'y'):
        raise ValueError('channel_axis must be x or y')
    px = pitch_x_mm / 2000
    py = pitch_y_mm / 2000
    if px <= 0 or py <= 0:
        raise ValueError('Cup pitch must be positive')
    if channel_axis == 'y':
        return [('a1', -px, +py, 'A'), ('a2', +px, +py, 'A'),
                ('b1', -px, -py, 'B'), ('b2', +px, -py, 'B')]
    return [('a1', -px, +py, 'A'), ('a2', -px, -py, 'A'),
            ('b1', +px, +py, 'B'), ('b2', +px, -py, 'B')]


def augment(src: Path, dest: Path, flange='link6', mount_xyz='0 0 0.015',
            mount_rpy='3.141592653589793 0 0', cups_mm=40, pitch_x_mm=44, pitch_y_mm=44,
            channel_axis='y'):
    if cups_mm not in (15, 30, 40):
        raise ValueError('Cup diameter must be 15, 30 or 40 mm')
    locations = cup_locations(pitch_x_mm, pitch_y_mm, channel_axis)
    if pitch_x_mm < cups_mm or pitch_y_mm < cups_mm:
        raise ValueError('Cup centers would overlap: pitch must be >= cup diameter')
    src, dest = Path(src), Path(dest)
    if src.resolve() == dest.resolve():
        raise ValueError('Input and output must be different files')
    tree = ET.parse(src)
    robot = tree.getroot()
    if robot.tag != 'robot':
        raise ValueError('Input must be a generated URDF <robot>, not a xacro macro')
    names = {link.get('name') for link in robot.findall('link')}
    if flange not in names:
        raise ValueError(f'Flange {flange!r} not in source robot. Available: {sorted(names)}')
    if 'vgc10_body' in names:
        raise ValueError('VGC10 appears already present; refusing duplicate links')

    # Partition the nominal total mass among the approximate body and cup proxies.
    body = element(robot, 'link', name='vgc10_body')
    body_mass = NOMINAL_TOTAL_MASS_KG - 4*CUP_PROXY_MASS_KG
    inertial_box(body, body_mass, BODY_X, BODY_Y, BODY_Z, '0 0 -0.050')
    geometry(body, 'visual', 'box', '0.101 0.100 0.100', '0 0 -0.050', '.15 .18 .20 1')
    geometry(body, 'collision', 'box', '0.101 0.100 0.100', '0 0 -0.050')
    add_joint(robot, 'uf850_to_vgc10', flange, 'vgc10_body', mount_xyz, mount_rpy)

    radius_m = 0.0202 if cups_mm == 40 else cups_mm/2000
    for cup_id, x, y, channel in locations:
        cup = element(robot, 'link', name=f'vgc10_cup_{cup_id}')
        inertial_cylinder(cup, CUP_PROXY_MASS_KG, radius_m, CUP_LENGTH_M)
        geometry(cup, 'visual', 'cylinder', f'{radius_m:.6f} {CUP_LENGTH_M:.3f}',
                 '0 0 -0.006', '.08 .08 .09 1' if channel == 'A' else '.12 .12 .13 1')
        geometry(cup, 'collision', 'cylinder', f'{radius_m:.6f} {CUP_LENGTH_M:.3f}',
                 '0 0 -0.006')
        add_joint(robot, f'vgc10_cup_{cup_id}_fixed', 'vgc10_body',
                  f'vgc10_cup_{cup_id}', f'{x:.6f} {y:.6f} -0.162395921')

    element(robot, 'link', name='vgc10_tcp')
    add_joint(robot, 'vgc10_tcp_fixed', 'vgc10_body', 'vgc10_tcp', '0 0 -0.174395921')
    ET.indent(tree, space='  ')
    dest.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dest, encoding='utf-8', xml_declaration=True)
    print(f'Generated: {dest} (VGC10 proxy, 4x{cups_mm} mm cups, '
          f'pitch={pitch_x_mm}x{pitch_y_mm} mm, channels split by {channel_axis.upper()})')
    print('CAUTION: Nominal mass is distributed across proxy components. '
          'Tool mount, TCP, pitch and vacuum behavior are NOT validated.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('uf850_urdf', type=Path)
    p.add_argument('output_urdf', type=Path)
    p.add_argument('--flange', default='link6')
    p.add_argument('--mount-xyz', default='0 0 0.015')
    p.add_argument('--mount-rpy', default='3.141592653589793 0 0')
    p.add_argument('--cup-mm', type=int, choices=(15, 30, 40), default=40)
    p.add_argument('--pitch-x-mm', type=float, default=44)
    p.add_argument('--pitch-y-mm', type=float, default=44)
    p.add_argument('--channel-axis', choices=('x', 'y'), default='y')
    args = p.parse_args()
    augment(args.uf850_urdf, args.output_urdf, args.flange, args.mount_xyz,
            args.mount_rpy, args.cup_mm, args.pitch_x_mm, args.pitch_y_mm,
            args.channel_axis)

if __name__ == '__main__':
    main()
