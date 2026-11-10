# UF850 v3.1 xacro fix

Fixes `package 'xarm_controller' not found` during expansion of upstream UF850 xacro.
The official UF850 robot-device macro is still used, but its automatic Gazebo plugin is disabled.
The project's `prepare_uf850_model.py` then installs `gz_ros2_control` and the project-local controller configuration.
There is no need to build `xarm_controller` or any real-hardware package.

## Install
From `~/conveyor-belief-vla`:
```bash
unzip -o ~/Downloads/conveyor_uf850_v3_1_xacro_fix.zip
bash -n scripts/setup_uf850.sh
bash scripts/setup_uf850.sh
```

If Docker images already built and `xarm_description` installed (as in the reported log), avoid re-running the builds:

```bash
docker compose -f compose.yaml -f compose.uf850.yaml run --rm --no-deps tools bash -c '
  set -eo pipefail
  source /opt/ros/jazzy/setup.bash
  source /workspace/external/xarm_ws/install/setup.bash
  ros2 pkg prefix xarm_description
  ros2 run xacro xacro /workspace/simulation/uf850_description_wrapper.xacro -o /workspace/simulation/generated/uf850_raw.urdf
  /usr/bin/python3 /workspace/scripts/prepare_uf850_model.py /workspace/simulation/generated/uf850_raw.urdf /workspace/simulation/generated/uf850_vgc10.urdf
  /usr/bin/python3 /workspace/scripts/make_uf850_world.py
'
```

Then start the UF850 mode as documented in `docs/UF850_V3_README.md`.

## Validation
Static XML parsing and shell syntax are checked on this patch; full xacro expansion
and Gazebo + ros2_control cannot be executed here. Please share any subsequent error.
