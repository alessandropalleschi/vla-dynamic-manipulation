#!/usr/bin/env bash
set -eo pipefail
# ROS environment setup requires nounset to be disabled.
source /opt/ros/jazzy/setup.bash
set -u
mkdir -p /workspace/data
RUN_ID="trial_$(date -u +%Y%m%dT%H%M%SZ)"
echo "Recording bag to /workspace/data/${RUN_ID} (Ctrl+C to stop)."
exec ros2 bag record --storage mcap -o "/workspace/data/${RUN_ID}" \
  /clock /camera/image_raw /joint_states \
  /gantry/x_cmd /gantry/y_cmd /gantry/z_cmd \
  /conveyor/speed /gripper/attach /gripper/detach /gripper/attached_state /expert/phase
