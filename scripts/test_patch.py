#!/usr/bin/env python3
"""Standalone static/unit checks; requires no ROS or Gazebo."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

from configure_overhead_camera import configure
from uf850_tool_urdf import augment
from evaluate_episode import evaluate


def main():
    source_world = Path(__file__).resolve().parent.parent/'simulation/worlds/conveyor_station.sdf'
    if not source_world.exists():
        source_world = Path('/mnt/data/conveyor_work/conveyor-belief-vla/simulation/worlds/conveyor_station.sdf')
    # Test camera change against a temporary copy, preserving any existing joint changes.
    with TemporaryDirectory() as temp:
        root = Path(temp)
        if source_world.exists():
            world = root/'world.sdf'
            world.write_bytes(source_world.read_bytes())
            before_z = ET.parse(world).getroot().find(".//joint[@name='z_joint']/axis/limit/lower").text
            configure(world, backup=False)
            after = ET.parse(world).getroot()
            assert after.find(".//joint[@name='z_joint']/axis/limit/lower").text == before_z
            assert after.find(".//model[@name='external_camera']/pose").text.startswith('-0.12')
        dummy = root/'uf850.urdf'
        dummy.write_text('<robot name="uf850"><link name="link6"/></robot>')
        out = root/'with_vgc10.urdf'
        augment(dummy, out)
        urdf = ET.parse(out).getroot()
        links = {x.attrib['name'] for x in urdf.findall('link')}
        assert {'vgc10_body','vgc10_cup_a','vgc10_cup_b','vgc10_tcp'} <= links
        assert float(urdf.find(".//link[@name='vgc10_body']/inertial/mass").attrib['value']) == 0.814
        assert len([j for j in urdf.findall('joint') if j.get('name', '').startswith('vgc10_channel_')]) == 2

        episode = root/'episode'
        episode.mkdir()
        (episode/'capture_status.json').write_text(json.dumps({'expert_exit_code':0, 'camera_frames':80,'requested_speed_m_s':0.04}))
        fields=['sim_time_s','gripper','lower_x','lower_y','lower_z','upper_x','upper_y','upper_z']
        with (episode/'telemetry.csv').open('w', newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
            for k in range(100):
                t=k*0.05
                x=-0.5+0.04*t
                writer.writerow({'sim_time_s':t,'gripper':'detached' if t>=1.0 else 'attached',
                                 'lower_x':x,'lower_y':0,'lower_z':0.83,
                                 'upper_x':x+0.002,'upper_y':0.002,'upper_z':0.90})
        success=evaluate(episode)
        assert success['verdict']=='POSE_STABLE', success
        (episode/'capture_status.json').write_text(json.dumps({'expert_exit_code':2, 'camera_frames':80,'requested_speed_m_s':0.04}))
        failure=evaluate(episode)
        assert failure['verdict']=='FAIL', failure
    print('PASS: VGC10 geometry, camera config preservation, pose evaluation, abort handling')


if __name__ == '__main__':
    main()
