#!/usr/bin/env python3
"""Visual-only VGC10 4-cup STL replacement. Leaves physical proxies untouched.

Source: UOsaka-Harada-Laboratory/onrobot (MIT-licensed ROS description).
The upstream xacro attaches base_link_4cups.stl with identity visual transform.
The chosen UF850 mounting transform is preserved and must be checked in Gazebo.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

MODEL_URI = ('/workspace/external/onrobot_ros_assets/'
             'onrobot_vg_description/meshes/vgc10/visual/base_link_4cups.stl')
LOCAL_REL = ('external/onrobot_ros_assets/'
             'onrobot_vg_description/meshes/vgc10/visual/base_link_4cups.stl')


def apply_real_visual(robot, mesh_path=None, xyz='0 0 0', rpy='0 0 0'):
    mesh_path = Path(mesh_path) if mesh_path else Path(MODEL_URI)
    if not mesh_path.is_file() or mesh_path.stat().st_size < 84:
        raise FileNotFoundError(f'VGC10 four-cup STL missing or empty: {mesh_path}')
    body = robot.find("./link[@name='vgc10_body']")
    if body is None:
        raise ValueError('No vgc10_body link in URDF; run UF850 model generation first')
    for cid in ('a1', 'a2', 'b1', 'b2'):
        if robot.find(f"./link[@name='vgc10_cup_{cid}']") is None:
            raise ValueError(f'Missing cup frame: vgc10_cup_{cid}')
    for old_visual in list(body.findall('visual')):
        body.remove(old_visual)
    visual = ET.SubElement(body, 'visual', name='vgc10_manufacturer_style_4cup_stl')
    ET.SubElement(visual, 'origin', xyz=xyz, rpy=rpy)
    geometry = ET.SubElement(visual, 'geometry')
    ET.SubElement(geometry, 'mesh', filename=MODEL_URI, scale='1 1 1')
    material = ET.SubElement(visual, 'material', name='vgc10_dark_gray')
    ET.SubElement(material, 'color', rgba='0.24 0.26 0.29 1')
    # Preserve all four individual cup visuals.
    return visual


def patch_generated_urdf(urdf: Path, xyz='0 0 0', rpy='0 0 0'):
    urdf = Path(urdf)
    tree = ET.parse(urdf)
    root = tree.getroot()
    if root.tag != 'robot':
        raise ValueError('Expected URDF <robot> root')
    mesh_file = urdf.parent.parent.parent / LOCAL_REL
    # This filename's absolute path in the container is independent of the host.
    apply_real_visual(root, mesh_path=mesh_file, xyz=xyz, rpy=rpy)
    # Do not change mounting joint, TCP, inertials, collisions, control or plugins.
    ET.indent(tree, space='  ')
    tree.write(urdf, encoding='utf-8', xml_declaration=True)
    return urdf
