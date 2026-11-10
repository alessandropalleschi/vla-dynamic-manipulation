#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
if [[ -f /workspace/external/xarm_ws/install/setup.bash ]]; then
  source /workspace/external/xarm_ws/install/setup.bash
fi
set -u
export GZ_SIM_RESOURCE_PATH="/workspace/external/xarm_ws/src/xarm_ros2:/workspace/external/xarm_ws/install/xarm_description/share:${GZ_SIM_RESOURCE_PATH:-}"
WORLD=/workspace/simulation/worlds/conveyor_station_uf850.sdf
MODEL=/workspace/simulation/generated/uf850_vgc10.urdf
if [[ ! -f "$WORLD" || ! -f "$MODEL" ]]; then
  echo 'Missing generated files. First run: bash scripts/setup_uf850.sh' >&2
  exit 2
fi
cleanup() {
  for pid in "${RSP_PID:-}" "${IMG_PID:-}" "${BRIDGE_PID:-}" "${GZ_PID:-}"; do
    if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; fi
  done
  wait || true
}
trap cleanup EXIT INT TERM

gz sim --force-version 8 -s -r --headless-rendering -v 3 "$WORLD" &
GZ_PID=$!
sleep 4
if ! kill -0 "$GZ_PID"; then echo 'Gazebo server failed to start' >&2; exit 1; fi
ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=/workspace/config/bridge.uf850.yaml &
BRIDGE_PID=$!
ros2 run ros_gz_image image_bridge /station/camera/image --ros-args \
  -r /station/camera/image:=/camera/image_raw -p qos:=sensor_data &
IMG_PID=$!

RSP_PARAMS=/tmp/uf850_rsp_params.yaml

/usr/bin/python3   /workspace/scripts/make_uf850_rsp_params.py   "$MODEL" "$RSP_PARAMS"

ros2 run robot_state_publisher robot_state_publisher   --ros-args --params-file "$RSP_PARAMS" &

RSP_PID=$!

RSP_READY=0

for i in $(seq 1 20); do
  if ! kill -0 "$RSP_PID" 2>/dev/null; then
    echo "robot_state_publisher exited unexpectedly" >&2
    exit 1
  fi

  if ros2 topic info /robot_description 2>/dev/null | grep -Eq 'Publisher count: [1-9]'; then
    RSP_READY=1
    echo "UF850 /robot_description publisher ready."
    break
  fi

  sleep 1
done

if [[ "$RSP_READY" -ne 1 ]]; then
  echo "Timed out waiting for /robot_description publisher" >&2
  exit 1
fi

echo 'Waiting for Gazebo world create service...'
READY=0
for i in $(seq 1 45); do
  if gz service -l | grep -qx '/world/conveyor_station/create'; then READY=1; break; fi
  if ! kill -0 "$GZ_PID" 2>/dev/null; then echo 'Gazebo server exited' >&2; exit 1; fi
  sleep 1
done
if [[ "$READY" != 1 ]]; then echo 'Timed out waiting for Gazebo create service' >&2; exit 1; fi

echo 'Spawning official UF850 geometry + 4-cup VGC10 proxy...'
ros2 run ros_gz_sim create --world conveyor_station --file "$MODEL" --name uf850 \
  -x -0.05 -y -0.47 -z 0.78 || { echo 'UF850 spawn failed' >&2; exit 1; }

echo 'Starting ROS 2 controllers...'
ros2 run controller_manager spawner joint_state_broadcaster \
  -c /controller_manager --controller-manager-timeout 60 || exit 1
ros2 run controller_manager spawner uf850_arm_controller \
  -c /controller_manager --controller-manager-timeout 60 || exit 1

echo 'UF850 ready for joint-level smoke tests (no suction physics / no autonomous stacking).'
wait -n "$GZ_PID" "$BRIDGE_PID" "$IMG_PID" "$RSP_PID"
echo 'A required simulation process exited' >&2
