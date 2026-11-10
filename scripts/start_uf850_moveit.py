#!/usr/bin/env python3
"""Start MoveIt's *plan-only* move_group using the already-generated simulation URDF.

Do not connect to real hardware. No automatic motion or execute API enabled.
"""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import yaml
from uf850_moveit_common import URDF_PATH, parse_robot, make_srdf, make_moveit_parameters


def main():
    urdf = URDF_PATH.read_text(encoding='utf-8')
    tree, root_frame, joints, limits = parse_robot(URDF_PATH)
    config = make_moveit_parameters(urdf, make_srdf(tree, joints), limits)
    config_file = Path('/tmp/uf850_move_group_plan_only.yaml')
    config_file.write_text(yaml.safe_dump({'move_group': {'ros__parameters': config}}, sort_keys=False))
    print(f'Planning root: {root_frame}', flush=True)
    print('MoveIt: PLAN ONLY. Controller execution is disabled.', flush=True)
    print('Warning: planning collision boxes are injected by plan_uf850_pickup.py, not by startup.', flush=True)
    cmd = ['ros2','run','moveit_ros_move_group','move_group','--ros-args','--params-file',str(config_file)]
    return subprocess.call(cmd)

if __name__ == '__main__':
    sys.exit(main())
