#!/usr/bin/env python3
"""Convert an official UF850 expanded URDF to Gazebo Harmonic + VGC10.

Requires an expanded URDF from xArm-Developer/xarm_ros2 (Jazzy). The upstream
xacro can include Gazebo Classic plugins; strip those and install gz_ros2_control.
The robot is never connected to real hardware.
"""
from pathlib import Path
import argparse
import sys
import xml.etree.ElementTree as ET

from uf850_tool_urdf import augment

JOINTS = [f'joint{i}' for i in range(1, 7)]
CONTROLLERS_YAML = '/workspace/config/uf850_controllers.yaml'


def configure(raw_urdf: Path, output: Path, flange: str = 'link6', cup_mm: int = 40,
              pitch_x_mm: float = 44, pitch_y_mm: float = 44, channel_axis: str = 'y',
              model_dir: str = '/workspace/external/xarm_ws/src/xarm_ros2/xarm_description'):
    raw_urdf, output = Path(raw_urdf), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Tool geometry from the user's chosen 4-cup layout; mass and dimensions approximate.
    augment(raw_urdf, output, flange=flange, mount_rpy='3.141592653589793 0 0', cups_mm=cup_mm,
            pitch_x_mm=pitch_x_mm, pitch_y_mm=pitch_y_mm, channel_axis=channel_axis)
    tree = ET.parse(output)
    robot = tree.getroot()
    # VGC10_REAL_STL_VISUAL_AUTOGEN
    from vgc10_real_visual import apply_real_visual, LOCAL_REL
    _asset = Path(__file__).resolve().parent.parent / LOCAL_REL
    if _asset.is_file():
        apply_real_visual(robot, mesh_path=_asset)
    link_names = {x.get('name') for x in robot.findall('link')}
    if not all(f'vgc10_cup_{i}' in link_names for i in ('a1', 'a2', 'b1', 'b2')):
        raise RuntimeError('Missing VGC10 2x2 cup links')
    joint_names = {x.get('name') for x in robot.findall('joint')}
    if not set(JOINTS).issubset(joint_names):
        raise RuntimeError(f'Expected UF850 arm joints {JOINTS}; found {sorted(joint_names)}')

    # Keep official link dynamics; use a dedicated, simulation-only arm control interface.
    # Don't use the vendor's real robot hardware plugin and don't load Gazebo Classic.
    for control in list(robot.findall('ros2_control')):
        robot.remove(control)
    control = ET.SubElement(robot, 'ros2_control', name='UF850GazeboSystem', type='system')
    hardware = ET.SubElement(control, 'hardware')
    ET.SubElement(hardware, 'plugin').text = 'gz_ros2_control/GazeboSimSystem'
    for name in JOINTS:
        joint = ET.SubElement(control, 'joint', name=name)
        ET.SubElement(joint, 'command_interface', name='position')
        ET.SubElement(joint, 'state_interface', name='position')
        ET.SubElement(joint, 'state_interface', name='velocity')

    removed_plugins = 0
    for gazebo in list(robot.findall('gazebo')):
        for plugin in list(gazebo.findall('plugin')):
            gazebo.remove(plugin)
            removed_plugins += 1
        if len(gazebo) == 0:
            robot.remove(gazebo)
    gazebo = ET.SubElement(robot, 'gazebo')
    plugin = ET.SubElement(gazebo, 'plugin',
                           filename='libgz_ros2_control-system.so',
                           name='gz_ros2_control::GazeboSimROS2ControlPlugin')
    ET.SubElement(plugin, 'parameters').text = CONTROLLERS_YAML
    ET.SubElement(plugin, 'controller_manager_name').text = 'controller_manager'

    # Absolute paths accessible in our bind-mounted Docker container. Gazebo otherwise
    # cannot always resolve package://xarm_description in its resource resolver.
    for mesh in robot.findall('.//mesh'):
        uri = mesh.get('filename', '')
        for prefix in ('package://xarm_description/', 'model://xarm_description/'):
            if uri.startswith(prefix):
                mesh.set('filename', model_dir.rstrip('/') + '/' + uri[len(prefix):])
                break

    ET.indent(tree, space='  ')
    tree.write(output, encoding='utf-8', xml_declaration=True)
    # Structural validation, including no real-hardware or Gazebo Classic plugins.
    check = ET.parse(output).getroot()
    assert len(check.findall('ros2_control')) == 1
    assert check.find('ros2_control/hardware/plugin').text == 'gz_ros2_control/GazeboSimSystem'
    plugins = check.findall('gazebo/plugin')
    assert len(plugins) == 1 and plugins[0].get('filename') == 'libgz_ros2_control-system.so'
    assert 'UFRobotSystemHardware' not in output.read_text()
    print(f'PASS: generated robot {output} with 6 controlled joints, 4 suction cups, gz_ros2_control')
    print(f'Removed {removed_plugins} original Gazebo plugin(s). Check model placement and mesh loading in Gazebo.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('raw_urdf', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--flange', default='link6')
    p.add_argument('--cup-mm', type=int, choices=[15, 30, 40], default=40)
    p.add_argument('--pitch-x-mm', type=float, default=44)
    p.add_argument('--pitch-y-mm', type=float, default=44)
    p.add_argument('--channel-axis', choices=['x', 'y'], default='y')
    args = p.parse_args()
    configure(args.raw_urdf, args.output, args.flange,
              args.cup_mm, args.pitch_x_mm, args.pitch_y_mm, args.channel_axis)


if __name__ == '__main__':
    main()
