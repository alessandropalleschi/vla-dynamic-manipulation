#!/usr/bin/env python3
from pathlib import Path
from shutil import copy2
from xml.etree import ElementTree as ET
import math
import ast

root = Path(__file__).resolve().parent.parent

LIP_Z = -0.174395921
JOINT_Z = LIP_Z + 0.012
LINK6_TCP_Z = 0.015 - LIP_Z

PITCH = 0.044
RADIUS = 0.0202

CUPS = {
    'a1': (-1,  1),
    'a2': ( 1,  1),
    'b1': (-1, -1),
    'b2': ( 1, -1),
}

def backup(path):
    dest = Path(str(path) + '.before_real_cup_sync')
    if not dest.exists():
        copy2(path, dest)

def patch_source(relative, changes):
    path = root / relative
    if not path.exists():
        raise RuntimeError(f'Missing source: {relative}')

    src = path.read_text()

    for old, new in changes:
        if old in src:
            count = src.count(old)

            expected = (
                2 if relative ==
                'scripts/uf850_stationary_stack_v7.py'
                and old == '+0.127m'
                else 1
            )

            if count != expected:
                raise RuntimeError(
                    f'{relative}: unexpected count for {old}: {count}'
                )

            src = src.replace(old, new)

        elif new not in src:
            raise RuntimeError(
                f'{relative}: unexpected source structure: {old}'
            )

    if path.suffix == '.py':
        ast.parse(src)

    backup(path)
    path.write_text(src)
    print('PATCHED:', relative)

# Update the persistent URDF generator.
patch_source('scripts/uf850_tool_urdf.py', [
    (
        "f'{x:.6f} {y:.6f} -0.100'",
        "f'{x:.6f} {y:.6f} -0.162395921'"
    ),
    (
        "'vgc10_tcp', '0 0 -0.112'",
        "'vgc10_tcp', '0 0 -0.174395921'"
    ),
    (
        'cups_mm=30, pitch_x_mm=50, pitch_y_mm=50',
        'cups_mm=40, pitch_x_mm=44, pitch_y_mm=44'
    ),
    (
        'cup_locations(pitch_x_mm=50, pitch_y_mm=50',
        'cup_locations(pitch_x_mm=44, pitch_y_mm=44'
    ),
    (
        "p.add_argument('--cup-mm', type=int, choices=(15, 30, 40), default=30)",
        "p.add_argument('--cup-mm', type=int, choices=(15, 30, 40), default=40)"
    ),
    (
        "p.add_argument('--pitch-x-mm', type=float, default=50)",
        "p.add_argument('--pitch-x-mm', type=float, default=44)"
    ),
    (
        "p.add_argument('--pitch-y-mm', type=float, default=50)",
        "p.add_argument('--pitch-y-mm', type=float, default=44)"
    ),
    (
        'radius_m = cups_mm/2000',
        'radius_m = 0.0202 if cups_mm == 40 else cups_mm/2000'
    ),
    (
        "mount_rpy='0 0 0', cups_mm",
        "mount_rpy='3.141592653589793 0 0', cups_mm"
    ),
    (
        "p.add_argument('--mount-rpy', default='0 0 0')",
        "p.add_argument('--mount-rpy', default='3.141592653589793 0 0')"
    ),
])

patch_source('scripts/prepare_uf850_model.py', [
    ('cup_mm: int = 30,', 'cup_mm: int = 40,'),
    (
        'pitch_x_mm: float = 50, pitch_y_mm: float = 50',
        'pitch_x_mm: float = 44, pitch_y_mm: float = 44'
    ),
    (
        "p.add_argument('--cup-mm', type=int, choices=[15, 30, 40], default=30)",
        "p.add_argument('--cup-mm', type=int, choices=[15, 30, 40], default=40)"
    ),
    (
        "p.add_argument('--pitch-x-mm', type=float, default=50)",
        "p.add_argument('--pitch-x-mm', type=float, default=44)"
    ),
    (
        "p.add_argument('--pitch-y-mm', type=float, default=50)",
        "p.add_argument('--pitch-y-mm', type=float, default=44)"
    ),
])

# Update the v7 kinematics validation.
patch_source('scripts/uf850_stationary_stack_v7.py', [
    ('[0,0,.127]', '[0,0,.189395921]'),
    ('+0.127m', '+0.189396m'),
])

patch_source('scripts/test_uf850_v7_offline.py', [
    ("'0 0 -.112'", "'0 0 -.174395921'"),
    ('+0.127m', '+0.189396m'),
])

# Preserve the correct parameters on future vacuum installation.
patch_source('scripts/install_uf850_vacuum.py', [
    (
        "'tool_offset':'0 0 0.127'",
        "'tool_offset':'0 0 0.189395921'"
    ),
    ("'cup_pitch_x':'0.05'", "'cup_pitch_x':'0.044'"),
    ("'cup_pitch_y':'0.05'", "'cup_pitch_y':'0.044'"),
])

# Match vacuum seal geometry to the actual STL lip.
patch_source('simulation/plugins/vgc10/Vgc10Vacuum.cc', [
    (
        'const double cupRadius = 0.015; // provisional 30-mm cup diameter',
        'const double cupRadius = 0.0202; // measured STL sealing-lip radius'
    ),
])

# Keep the isolated vacuum fixture compatible.
fixture = root / 'simulation/tests/vgc10_fixture.sdf'

if fixture.exists():
    tree = ET.parse(fixture)
    plugin = tree.getroot().find(
        ".//plugin[@name='conveyor::Vgc10Vacuum']"
    )

    if plugin is None:
        raise RuntimeError('Vacuum fixture plugin missing')

    for tag in ('cup_pitch_x', 'cup_pitch_y'):
        plugin.find(tag).text = '0.044'

    backup(fixture)
    ET.indent(tree, space='  ')
    tree.write(fixture, encoding='utf-8', xml_declaration=True)

# Update the live Gazebo station.
world = root / 'simulation/worlds/conveyor_station_uf850.sdf'

tree = ET.parse(world)
plugin = tree.getroot().find(
    ".//plugin[@name='conveyor::Vgc10Vacuum']"
)

if plugin is None:
    raise RuntimeError('Gazebo VGC10 plugin missing')

for tag, value in (
    ('tool_offset', f'0 0 {LINK6_TCP_Z:.9f}'),
    ('cup_pitch_x', f'{PITCH:.3f}'),
    ('cup_pitch_y', f'{PITCH:.3f}'),
):
    element = plugin.find(tag)
    if element is None:
        raise RuntimeError(f'Missing SDF parameter: {tag}')
    element.text = value

backup(world)
ET.indent(tree, space='  ')
tree.write(world, encoding='utf-8', xml_declaration=True)

# Modify the CURRENT robot while preserving the real STL visuals.
urdf = root / 'simulation/generated/uf850_vgc10.urdf'

tree = ET.parse(urdf)
robot = tree.getroot()

mount = robot.find("./joint[@name='uf850_to_vgc10']/origin")

if mount is None:
    raise RuntimeError('VGC10 mounting joint missing')

if abs(float(mount.get('xyz').split()[2]) - 0.015) > 1e-6:
    raise RuntimeError('Unexpected VGC10 mounting translation')

if abs(float(mount.get('rpy').split()[0]) - math.pi) > .001:
    raise RuntimeError('Unexpected VGC10 mounting rotation')

body = robot.find("./link[@name='vgc10_body']")

if body is None:
    raise RuntimeError('VGC10 body missing')

tcp = robot.find("./joint[@name='vgc10_tcp_fixed']/origin")

if tcp is None:
    raise RuntimeError('VGC10 TCP joint missing')

tcp.set('xyz', f'0 0 {LIP_Z:.9f}')

for cid, (sx, sy) in CUPS.items():
    x = sx * PITCH / 2
    y = sy * PITCH / 2

    matches = [
        v for v in body.findall('visual')
        if v.get('name') == f'vgc10_body_cup_{cid}'
    ]

    if len(matches) != 1:
        raise RuntimeError(f'Expected one real STL visual: {cid}')

    visual = matches[0]
    mesh = visual.find('geometry/mesh')

    if mesh is None or not mesh.get('filename', '').endswith(
        'suction_cup_1cup.stl'
    ):
        raise RuntimeError(f'Unexpected cup mesh: {cid}')

    origin = visual.find('origin')
    xyz = [float(v) for v in origin.get('xyz').split()]

    if abs(xyz[2] + .0901) > .001:
        raise RuntimeError(f'Unexpected mesh height for {cid}')

    origin.set(
        'xyz',
        f'{x:.6f} {y:.6f} {xyz[2]:.7f}'
    )

    joint = robot.find(
        f"./joint[@name='vgc10_cup_{cid}_fixed']/origin"
    )

    if joint is None:
        raise RuntimeError(f'Missing cup joint: {cid}')

    joint.set(
        'xyz',
        f'{x:.6f} {y:.6f} {JOINT_Z:.9f}'
    )

    cup = robot.find(
        f"./link[@name='vgc10_cup_{cid}']"
    )

    if cup is None:
        raise RuntimeError(f'Missing cup link: {cid}')

    # Approximate sealing rim collision.
    cap = cup.find('collision/geometry/cylinder')

    if cap is None:
        raise RuntimeError(f'Missing cup collision: {cid}')

    cap.set('radius', f'{RADIUS:.5f}')

    # Add an approximate stem collision for the extended cup.
    if not any(
        c.get('name') == f'vgc10_shaft_{cid}'
        for c in cup.findall('collision')
    ):
        shaft = ET.SubElement(
            cup, 'collision',
            name=f'vgc10_shaft_{cid}'
        )

        ET.SubElement(
            shaft,
            'origin',
            xyz='0 0 0.0344',
            rpy='0 0 0'
        )

        geometry = ET.SubElement(shaft, 'geometry')

        ET.SubElement(
            geometry,
            'cylinder',
            radius='0.011',
            length='0.076'
        )

    # The four real STL visuals are already attached to vgc10_body.
    for old_visual in list(cup.findall('visual')):
        cup.remove(old_visual)

backup(urdf)

ET.indent(tree, space='  ')
tree.write(urdf, encoding='utf-8', xml_declaration=True)

# Prevent accidental replay of the old short-tool trajectory.
plan = root / 'data/uf850_hover_plan.json'

if plan.exists():
    old = plan.with_name(
        'uf850_hover_plan.OLD_127mm_DO_NOT_EXECUTE.json'
    )

    if not old.exists():
        plan.rename(old)
        print('ARCHIVED OLD TRAJECTORY:', old)
    else:
        print('Old plan already archived; current file left untouched.')

print()
print('PASS: Real VGC10 cup geometry synchronized')
print(f'Cup lip Z in body: {LIP_Z:.6f} m')
print(f'TCP Z from link6: +{LINK6_TCP_Z:.6f} m')
print(f'Cup pitch: {PITCH*1000:.1f} mm')
print(f'Cup lip diameter: {RADIUS*2000:.1f} mm')
print()
print('IMPORTANT: Rebuild the plugin and regenerate MoveIt plans.')
print('DO NOT execute old trajectories.')
