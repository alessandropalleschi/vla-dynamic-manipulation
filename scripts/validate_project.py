#!/usr/bin/env python3
"""Static structural checks available without ROS/Docker/Gazebo installed."""
import pathlib
import sys
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[1]


def verify():
    world = ET.parse(ROOT / 'simulation/worlds/conveyor_station.sdf').getroot()
    assert world.tag == 'sdf'
    w = world.find('world')
    assert w is not None and w.attrib['name'] == 'conveyor_station'
    names = {m.get('name') for m in w.findall('model')}
    assert {'conveyor', 'pickup_table', 'lower_box', 'upper_box', 'gantry', 'external_camera'} <= names
    plugins = [p.get('name') for p in w.iter('plugin')]
    for required in ['gz::sim::systems::TrackController', 'gz::sim::systems::DetachableJoint',
                     'gz::sim::systems::JointPositionController', 'gz::sim::systems::Sensors']:
        assert required in plugins, required
    topic_tags = [e.text for e in w.iter('topic') if e.text]
    assert '/station/camera/image' in topic_tags
    assert '/conveyor/speed' in [e.text for e in w.iter('velocity_topic')]
    assert '/gripper/attach' in [e.text for e in w.iter('attach_topic')]
    assert '/gripper/detach' in [e.text for e in w.iter('detach_topic')]
    gantry = next(m for m in w.findall('model') if m.get('name') == 'gantry')
    assert {j.get('name') for j in gantry.findall('joint')} >= {'x_joint', 'y_joint', 'z_joint'}
    try:
        import yaml
    except ImportError:
        yaml = None
    if yaml:
        with (ROOT / 'config/bridge.yaml').open() as f:
            bridge = yaml.safe_load(f)
        assert len(bridge) == 9
        topics = {x['ros_topic_name'] for x in bridge}
        assert {'/clock', '/joint_states', '/conveyor/speed', '/gantry/x_cmd',
                '/gantry/y_cmd', '/gantry/z_cmd', '/gripper/attach', '/gripper/detach'} <= topics
    # Guard against ROS setup scripts failing when sourced under Bash nounset.
    for script in ('start_sim.sh', 'record.sh'):
        shell = (ROOT / 'scripts' / script).read_text()
        assert shell.index('source /opt/ros/jazzy/setup.bash') < shell.index('set -u\n'), script
    dockerfile = (ROOT / 'docker/Dockerfile').read_text()
    assert 'ros-jazzy-rclpy' in dockerfile
    assert 'PYTHONPATH="/opt/ros/jazzy/lib/python3.12/site-packages"' in dockerfile
    assert 'PASS: direct ROS Python imports' in dockerfile
    assert 'command: ["sleep", "infinity"]' in (ROOT / 'compose.yaml').read_text()
    # Detect any accidental host-only paths in Docker scripts.
    assert 'FROM ros:jazzy-ros-base-noble' in (ROOT / 'docker/Dockerfile').read_text()
    print('PASS: world XML, physics plugins, topics and basic project structure')


if __name__ == '__main__':
    try:
        verify()
    except Exception as e:
        print('FAIL:', e, file=sys.stderr)
        raise
