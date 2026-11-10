#!/usr/bin/env bash
set -eo pipefail
# ROS environment setup requires nounset to be disabled.
source /opt/ros/jazzy/setup.bash
set -u
cleanup() {
  for pid in "${IMG_PID:-}" "${BRIDGE_PID:-}" "${GZ_PID:-}"; do
    if [[ -n "${pid}" ]]; then kill "${pid}" 2>/dev/null || true; fi
  done
  wait || true
}
trap cleanup EXIT INT TERM
# -s: server only; -r: unpause on start; --headless-rendering: EGL/OGRE2 camera without X11.
gz sim --force-version 8 -s -r --headless-rendering -v 3 /workspace/simulation/worlds/conveyor_station.sdf &
GZ_PID=$!
sleep 3
if ! kill -0 "$GZ_PID" 2>/dev/null; then
  echo 'Gazebo exited during startup. Inspect the preceding error log.' >&2
  exit 1
fi
ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=/workspace/config/bridge.yaml &
BRIDGE_PID=$!
ros2 run ros_gz_image image_bridge /station/camera/image --ros-args \
  -r /station/camera/image:=/camera/image_raw -p qos:=sensor_data &
IMG_PID=$!
echo 'Started Gazebo, ROS/GZ bridge and external camera bridge.'
wait -n "$GZ_PID" "$BRIDGE_PID" "$IMG_PID"
echo 'A required service exited; stopping remaining processes.' >&2
