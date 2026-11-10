#!/usr/bin/env python3
"""READ-ONLY Gazebo physics outcome test: green box resting on orange box.

Run inside sim after v7 stationary_stack reports all stages complete. Uses
Gazebo's pose publishers, not MoveIt's predicted poses or camera appearance.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from check_gz_box_pose_v7 import read_gz


def evaluate(green, orange, max_xy=.022, max_z=.014):
    # Gazebo SDF boxes: orange 0.08m tall; green 0.06m tall.
    desired_green_center_z=orange[2]+(.08+.06)/2
    e_xy=max(abs(green[0]-orange[0]),abs(green[1]-orange[1]))
    e_z=abs(green[2]-desired_green_center_z)
    return {'xy_error_m':e_xy,'z_error_m':e_z,
            'expected_green_z_m':desired_green_center_z,
            'geometrically_stacked': e_xy<max_xy and e_z<max_z}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=Path('/workspace/data/uf850_stack_v7/stack_verification.json'))
    ap.add_argument('--settle-seconds',type=float,default=2.)
    args=ap.parse_args()
    poses=[]
    for i in range(2):
        lower,lowq=read_gz('lower_box')
        upper,upq=read_gz('upper_box')
        poses.append({'orange_xyz':lower,'green_xyz':upper,'orange_xyzw':lowq,'green_xyzw':upq})
        print(f'Sample {i+1}: orange={lower}, green={upper}')
        if i==0:time.sleep(args.settle_seconds)
    first,last=poses
    actual=evaluate(last['green_xyz'],last['orange_xyz'])
    green_motion=max(abs(a-b) for a,b in zip(first['green_xyz'],last['green_xyz']))
    orange_motion=max(abs(a-b) for a,b in zip(first['orange_xyz'],last['orange_xyz']))
    stable=green_motion<.004 and orange_motion<.004
    # The contact outcome is only geometrically plausible when box centers
    # align vertically and both positions remain stable across observations.
    success=actual['geometrically_stacked'] and stable
    report={'success':success, 'stationary_two_samples':stable,
            'green_motion_m':green_motion,'orange_motion_m':orange_motion,
            'geometry':actual,'measurements':poses,'timestamp':time.time()}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2))
    print('Vertical error: %.4fm; XY error: %.4fm' % (actual['z_error_m'],actual['xy_error_m']))
    print('Box motion: green %.4fm, orange %.4fm' % (green_motion,orange_motion))
    print('PASS: stationary green-on-orange geometry' if success else 'FAIL: stable stacking NOT confirmed')
    print('Saved:',args.out)
    return 0 if success else 2

if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,OSError) as e:
        print('BLOCKED:',e,file=sys.stderr);sys.exit(2)
