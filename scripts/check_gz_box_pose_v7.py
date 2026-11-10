#!/usr/bin/env python3
"""READ-ONLY: validate current Gazebo box poses against initial station SDF.

Run inside the existing `sim` container, where the gz transport CLI works.
Writes a short-lived attestation used by uf850_stationary_stack_v7.py. No motion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def sdf_position(root, name):
    e=root.find(f"./world/model[@name='{name}']/pose")
    if e is None or not e.text:raise ValueError(f'Missing SDF model pose: {name}')
    vals=[float(x) for x in e.text.split()]
    if len(vals)!=6 or max(abs(x) for x in vals[3:])>1e-5:
        raise ValueError(f'Non-axis-aligned SDF model: {name}')
    return vals[:3]


def parse_gz_pose(text):
    """Protobuf text format elides fields equal to zero."""
    chunks=re.findall(r'position\s*\{([^}]*)\}',text,re.S)
    if not chunks:raise ValueError('No position block in gz topic output')
    s=chunks[-1]
    d={n:float(v) for n,v in re.findall(r'([xyz])\s*:\s*([-+\d.eE]+)',s)}
    return [d.get('x',0.),d.get('y',0.),d.get('z',0.)]


def parse_gz_orientation(text):
    chunks=re.findall(r'orientation\s*\{([^}]*)\}',text,re.S)
    if not chunks:return [0.,0.,0.,1.]
    d={n:float(v) for n,v in re.findall(r'([wxyz])\s*:\s*([-+\d.eE]+)',chunks[-1])}
    return [d.get('x',0.),d.get('y',0.),d.get('z',0.),d.get('w',0.)]


def read_gz(name):
    topic=f'/simulation/{name}/pose'
    try:
        p=subprocess.run(['timeout','3','gz','topic','-e','-t',topic],
                         capture_output=True,text=True,timeout=5)
    except (OSError,subprocess.TimeoutExpired) as e:
        raise ValueError(f'Gazebo CLI could not read {topic}: {e}') from e
    if 'position' not in p.stdout:
        raise ValueError(f'No pose received on {topic}; is Gazebo running? stderr: {p.stderr[-250:]}')
    return parse_gz_pose(p.stdout),parse_gz_orientation(p.stdout)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--world',type=Path,default=Path('/workspace/simulation/worlds/conveyor_station_uf850.sdf'))
    ap.add_argument('--out',type=Path,default=Path('/workspace/data/uf850_gz_pose_snapshot.json'))
    ap.add_argument('--max-xy-error',type=float,default=.005)
    ap.add_argument('--max-z-error',type=float,default=.006)
    args=ap.parse_args()
    root=ET.parse(args.world).getroot()
    report={'timestamp_wall_s':time.time(),
            'world_sha256':hashlib.sha256(args.world.read_bytes()).hexdigest(),
            'boxes':{},'all_ok':True}
    for name in ('upper_box','lower_box'):
        expected=sdf_position(root,name)
        actual,quat=read_gz(name)
        qnorm=sum(v*v for v in quat)**.5
        if abs(qnorm-1)>0.02:
            raise ValueError(f'{name}: invalid Gazebo orientation quaternion {quat}')
        # World SDF expects zero model rotation. Quaternion +/- signs equivalent.
        rotated=2*__import__('math').acos(min(1.,abs(quat[3])/qnorm))
        errors=[abs(a-b) for a,b in zip(actual,expected)]
        valid=max(errors[:2])<args.max_xy_error and errors[2]<args.max_z_error and rotated<.05
        report['all_ok']=report['all_ok'] and valid
        report['boxes'][name]={'expected':expected,'actual':actual,'errors_abs_m':errors,'rot_error_rad':rotated,'ok':valid}
        print(name, 'expected=',expected,'actual=',[round(v,4) for v in actual],
              'delta=',[round(v,4) for v in errors],f'rot={rotated:.4f}rad', 'PASS' if valid else 'BLOCK')
    if not report['all_ok']:
        print('BLOCKED: live Gazebo box poses do not match the static planner snapshot')
        return 2
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2))
    print('PASS: fresh Gazebo box poses match SDF. Snapshot saved:',args.out)
    print('NOTE: conveyor must remain stopped; snapshot does not monitor subsequent motion.')
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,OSError,ET.ParseError) as e:
        print('BLOCKED:',e,file=sys.stderr);sys.exit(2)
