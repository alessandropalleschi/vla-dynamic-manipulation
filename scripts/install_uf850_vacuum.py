#!/usr/bin/env python3
"""Install the vacuum plugin into the GENERATED UF850 world + dedicated bridge.

Idempotent, backs up user files, never modifies the working gantry world.
Does not build or launch Gazebo.
"""
import argparse
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import yaml

TOPICS = [
    ('/vgc10/channel_a/enable', 'std_msgs/msg/Bool', 'gz.msgs.Boolean', 'ROS_TO_GZ'),
    ('/vgc10/channel_b/enable', 'std_msgs/msg/Bool', 'gz.msgs.Boolean', 'ROS_TO_GZ'),
    ('/vgc10/attached', 'std_msgs/msg/Bool', 'gz.msgs.Boolean', 'GZ_TO_ROS'),
    ('/vgc10/cup_contacts', 'std_msgs/msg/String', 'gz.msgs.StringMsg', 'GZ_TO_ROS'),
    ('/vgc10/grasp_quality', 'std_msgs/msg/Float64', 'gz.msgs.Double', 'GZ_TO_ROS'),
    ('/vgc10/status', 'std_msgs/msg/String', 'gz.msgs.StringMsg', 'GZ_TO_ROS'),
    ('/vgc10/diagnostics', 'std_msgs/msg/String', 'gz.msgs.StringMsg', 'GZ_TO_ROS'),
]


def backup_once(p):
    dst = p.with_name(p.name + '.pre_vacuum_backup')
    if not dst.exists(): shutil.copy2(p, dst)


def install(root: Path):
    world_path = root/'simulation/worlds/conveyor_station_uf850.sdf'
    if not world_path.exists():
        raise RuntimeError(f'{world_path} not found: run UF850 generation first')
    tree = ET.parse(world_path)
    world = tree.getroot().find('world')
    assert world is not None and world.get('name') == 'conveyor_station'
    if world.find(".//model[@name='gantry']") is not None:
        raise RuntimeError('Refusing to change a world containing the gantry')
    existing = [p for p in world.findall('plugin') if p.get('name') == 'conveyor::Vgc10Vacuum']
    if not existing:
        backup_once(world_path)
        plugin = ET.SubElement(world, 'plugin', filename='libVgc10Vacuum.so', name='conveyor::Vgc10Vacuum')
        for k, value in {
            'robot_model':'uf850', 'parent_link':'link6',
            'object_model':'upper_box', 'object_link':'box_link',
            'tool_offset':'0 0 0.189395921', 'suction_axis':'0 0 1', 'cup_pitch_x':'0.044', 'cup_pitch_y':'0.044',
            'box_size_x':'0.09','box_size_y':'0.09','box_size_z':'0.06',
            'contact_tolerance':'0.007', 'stable_contact_s':'0.15',
        }.items(): ET.SubElement(plugin, k).text = value
        ET.indent(tree, space='  ')
        tree.write(world_path, encoding='utf-8', xml_declaration=True)
    startup = root/'scripts/start_uf850_sim.sh'
    source = startup.read_text()
    # Use the active UF850-specific config: do not add to gantry bridge.
    import re
    matches = re.findall(r'config_file:=/workspace/(config/[A-Za-z0-9_.-]+\.yaml)', source)
    if len(set(matches)) != 1:
        raise RuntimeError('Cannot identify a unique active ROS bridge YAML in UF850 startup: '+str(matches))
    bridge_path = root/matches[0]
    if bridge_path.name == 'bridge.yaml':
        raise RuntimeError('UF850 still uses the shared gantry bridge.yaml; apply the dedicated-bridge fix first')
    records = yaml.safe_load(bridge_path.read_text())
    if not isinstance(records, list): raise RuntimeError('Unexpected bridge format')
    by_name = {r.get('ros_topic_name'): r for r in records}
    for topic, ros_type, gz_type, direction in TOPICS:
        wanted = dict(ros_topic_name=topic, gz_topic_name=topic,
                      ros_type_name=ros_type, gz_type_name=gz_type, direction=direction)
        if topic in by_name:
            if by_name[topic] != wanted:
                raise RuntimeError(f'Incompatible existing bridge entry for {topic}')
        else:
            records.append(wanted)
    backup_once(bridge_path)
    bridge_path.write_text(yaml.safe_dump(records, sort_keys=False))
    compose = root/'compose.uf850.vacuum.yaml'
    compose.write_text('''# Add to -f compose.yaml -f compose.uf850.yaml
services:
  sim:
    environment:
      GZ_SIM_SYSTEM_PLUGIN_PATH: /workspace/simulation/plugins/vgc10/build
''')
    print(f'PASS: UF850 world plugin installed, {len(TOPICS)} ROS topics bridged, Compose overlay generated.')
    print('VGC10 starts DETACHED; tool geometry and channel routing are provisional.')
    print('Next: bash scripts/build_uf850_vacuum.sh')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    a = p.parse_args()
    install(a.root)
