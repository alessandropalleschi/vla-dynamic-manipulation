#!/usr/bin/env python3
"""Idempotently add an overhead RGB camera to the UF850 Gazebo SDF."""
from pathlib import Path
from shutil import copy2
import xml.etree.ElementTree as ET

world = Path('simulation/worlds/conveyor_station_uf850.sdf')
src = world.read_text()
root = ET.fromstring(src)
w = root.find('./world')
if w is None or w.get('name') != 'conveyor_station':
    raise SystemExit('BLOCKED: Unexpected SDF world name')
if w.find("./model[@name='uf850_overhead_camera']") is not None:
    print('PASS: overhead camera already installed')
    raise SystemExit(0)
if src.count('</world>') != 1:
    raise SystemExit('BLOCKED: Unexpected world closing tag')
plugin = ''
if not any(p.get('name','') == 'gz::sim::systems::Sensors' for p in w.findall('plugin')):
    plugin = '''
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
'''
camera = '''
    <!-- VLA external observation: fixed overhead RGB, looks down along -Z. -->
    <model name="uf850_overhead_camera">
      <static>true</static>
      <pose>-0.35 -0.26 2.45 0 1.570796326795 0</pose>
      <link name="camera_link">
        <sensor name="rgb" type="camera">
          <always_on>true</always_on>
          <update_rate>10</update_rate>
          <topic>/station/overhead/image</topic>
          <gz_frame_id>uf850_overhead_camera_link</gz_frame_id>
          <camera>
            <camera_info_topic>/station/overhead/camera_info</camera_info_topic>
            <horizontal_fov>1.35</horizontal_fov>
            <image>
              <width>640</width><height>480</height><format>R8G8B8</format>
            </image>
            <clip><near>0.1</near><far>6.0</far></clip>
          </camera>
        </sensor>
      </link>
    </model>
'''
updated = src.replace('</world>', plugin + camera + '  </world>')
check = ET.fromstring(updated)
assert check.find("./world/model[@name='uf850_overhead_camera']") is not None
bak = world.with_name(world.name + '.before_vla_camera')
if not bak.exists():
    copy2(world, bak)
world.write_text(updated)
print('PASS: Overhead RGB camera installed (640x480, 10Hz)')
print('BACKUP:', bak)
