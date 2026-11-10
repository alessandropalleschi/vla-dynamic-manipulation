#!/usr/bin/env python3
"""Standalone checks for the VGC10 2x2 patch (no Docker/Gazebo required)."""
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET
from uf850_tool_urdf import augment, cup_locations
from visualize_vgc10_on_gantry import transform


def main():
    assert len(cup_locations()) == 4
    assert {c[0] for c in cup_locations()} == {'a1', 'a2', 'b1', 'b2'}
    assert [c[3] for c in cup_locations()].count('A') == 2
    assert [c[3] for c in cup_locations()].count('B') == 2
    assert len({(c[1], c[2]) for c in cup_locations()}) == 4
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp/'uf850.urdf'
        src.write_text('<robot name="uf850"><link name="link6"/></robot>')
        out = tmp/'combined.urdf'
        augment(src, out)
        root = ET.parse(out).getroot()
        links = {x.get('name') for x in root.findall('link')}
        assert {'vgc10_body', 'vgc10_cup_a1', 'vgc10_cup_a2',
                'vgc10_cup_b1', 'vgc10_cup_b2', 'vgc10_tcp'} <= links
        assert len([j for j in root.findall('joint') if j.get('name','').startswith('vgc10_cup_')]) == 4
        masses = [float(x.get('value')) for x in root.findall('link/inertial/mass')]
        assert abs(sum(masses) - 0.814) < 1e-9
        assert len({j.get('name') for j in root.findall('joint')}) == len(root.findall('joint'))

        scene = tmp/'world.sdf'
        scene.write_text('''<sdf version="1.9"><world name="test"><model name="gantry">
          <link name="z_carriage"><visual name="column"/><visual name="suction_tip"/>
          <visual name="vgc10_cup_a_visual"/></link>
          <joint name="z_joint"><axis><limit><lower>-0.03</lower></limit></axis></joint>
          </model><model name="external_camera"><pose>0 0 2 0 0 0</pose></model></world></sdf>''')
        transform(scene)
        transform(scene)  # rerun must not duplicate visuals
        sc = ET.parse(scene).getroot()
        link = sc.find(".//model[@name='gantry']/link[@name='z_carriage']")
        visual_names = [v.get('name') for v in link.findall('visual')]
        assert set(visual_names) == {'vgc10_body_visual', 'vgc10_cup_a1_visual',
                                     'vgc10_cup_a2_visual', 'vgc10_cup_b1_visual',
                                     'vgc10_cup_b2_visual'}, visual_names
        assert len(visual_names) == 5
        assert sc.find(".//joint[@name='z_joint']/axis/limit/lower").text == '-0.03'
        assert sc.find(".//model[@name='external_camera']/pose").text == '0 0 2 0 0 0'
    print('PASS: 4-cup URDF layout, two channels, nominal mass, idempotent visuals, unchanged Z joint and camera')

if __name__ == '__main__':
    main()
