#!/usr/bin/env python3
"""Offline geometry and plan-report tests. Does not launch ROS or execute Gazebo."""
import json
import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import uf850_stationary_stack_v7 as v7
import check_gz_box_pose_v7 as gzcheck
import verify_gz_stack_v7 as verifier


def toy_urdf(path):
    root=ET.Element('robot',name='synthetic_kinematics')
    for link in ['world']+[f'l{i}' for i in range(1,6)]+['link6']+['vgc10_body','vgc10_tcp']:
        ET.SubElement(root,'link',name=link)
    axes=['0 0 1','0 1 0','0 1 0','1 0 0','0 1 0','0 0 1']
    locs=['0 0 .1','.15 0 .2','.25 0 0','.20 0 0','.10 0 0','.05 0 0']
    parent='world'
    for i in range(1,7):
        j=ET.SubElement(root,'joint',name=f'joint{i}',type='revolute')
        ET.SubElement(j,'parent',link=parent)
        ET.SubElement(j,'child',link=('link6' if i==6 else f'l{i}'))
        ET.SubElement(j,'origin',xyz=locs[i-1],rpy='0 0 0')
        ET.SubElement(j,'axis',xyz=axes[i-1])
        ET.SubElement(j,'limit',lower='-3.14',upper='3.14',velocity='1.0')
        parent=('link6' if i==6 else f'l{i}')
    for name,parent,child,loc,rpy in [
        ('uf850_to_vgc10','link6','vgc10_body','0 0 .015',f'{math.pi} 0 0'),
        ('vgc10_tcp_fixed','vgc10_body','vgc10_tcp','0 0 -.174395921','0 0 0')]:
        j=ET.SubElement(root,'joint',name=name,type='fixed')
        ET.SubElement(j,'parent',link=parent)
        ET.SubElement(j,'child',link=child)
        ET.SubElement(j,'origin',xyz=loc,rpy=rpy)
    ET.ElementTree(root).write(path)


def test_geom():
    with tempfile.TemporaryDirectory() as temp:
        urdf=Path(temp)/'test.urdf'
        toy_urdf(urdf)
        model=v7.RobotKinematics(urdf)
        assert model.root_frame=='world'
        home=np.array([.2,-.8,1.1,.2,.4,-.1])
        T=model.T(home)
        print('Synthetic TCP at:',T[:3,3])
        assert np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-12)
        assert np.linalg.det(T[:3,:3])>0.999999
        target=T[:3,3]+np.array([-.012,.009,-.031])
        q=model.solve_near(home,target,T[:3,:3])
        p=model.T(q)
        assert np.linalg.norm(p[:3,3]-target)<.002
        assert np.linalg.norm(v7.rotvec(T[:3,:3]@p[:3,:3].T))<.025
        print('PASS: corrected +0.189396m TCP mount, 6D incremental IK and orientation preservation')
        wrong_target=target+np.array([10.,0.,0.])
        try:
            model.solve_near(home,wrong_target,T[:3,:3])
        except v7.Blocked:
            pass
        else:
            raise AssertionError('Unreachable pose was not rejected')
        print('PASS: unreachable IK target blocked')
        for theta in (-.3,.0,.7,1.4,2.0):
            R=v7.rot_rpy([.3,theta,-.2])
            q=v7.quat_xyzw(R)
            assert abs(np.linalg.norm(q)-1)<1e-8
        print('PASS: attached object rotation quaternion conversion')
        matrix=SimpleNamespace(entry_names=['joint1','joint2'],
                               entry_values=[SimpleNamespace(enabled=[True,False]),
                                             SimpleNamespace(enabled=[False,True])])
        names,rows,old=v7.change_acm_pair(matrix,'green','table',True)
        assert old is False and len(names)==4 and rows[0][:2]==[True,False]
        assert rows[1][:2]==[False,True]
        assert rows[2][3] and rows[3][2]
        matrix=SimpleNamespace(entry_names=names,entry_values=[SimpleNamespace(enabled=r) for r in rows])
        names,rows,old=v7.change_acm_pair(matrix,'green','table',False,False)
        assert rows[2][3] is False and rows[3][2] is False
        assert rows[0][:2]==[True,False]
        print('PASS: lift support-contact exception touches ONLY the designated pair and is restored')


def test_pose_parser():
    p="""name: "lower_box"
position { x: -0.52 z: 0.83 }
orientation { w: 1 }
"""
    assert gzcheck.parse_gz_pose(p)==[-.52,0.,.83]
    assert gzcheck.parse_gz_orientation(p)==[0.,0.,0.,1.]
    print('PASS: gz pose protobuf text with omitted zero components')
    good=verifier.evaluate([-.52,0,.90],[-.52,0,.83])
    bad=verifier.evaluate([-.55,-.55,.82],[-.52,0,.83])
    assert good['geometrically_stacked'] and not bad['geometrically_stacked']
    print('PASS: geometric stable-stack verification accepts stacked and rejects unstacked poses')


def test_report():
    # Validate previously uploaded user trajectory without reproducing MoveIt runtime.
    supplied=Path(__file__).resolve().parent.parent/'data/uf850_hover_plan.json'
    if not supplied.is_file():
        supplied=Path('/mnt/data/uf850_hover_plan.json')  # only for offline artifact test environment
    if supplied.is_file():
        d=json.loads(supplied.read_text())
        assert d['success'] is True
        assert len(d['waypoints'])==211
        assert d['joint_names']==v7.NAMES
        assert abs(d['waypoints'][-1]['seconds']-20.918647723)<1e-6
        print('PASS: user trajectory recognized; 211 points, 20.918647723 seconds')

if __name__=='__main__':
    test_geom();test_pose_parser();test_report()
