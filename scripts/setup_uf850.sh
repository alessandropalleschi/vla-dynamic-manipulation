#!/usr/bin/env bash
# Run once from project root. Downloads official robot description, then generates Gazebo URDF.
set -eo pipefail
cd "$(dirname "$0")/.."

WS="$PWD/external/xarm_ws"
REPO="$WS/src/xarm_ros2"
if [[ ! -d "$REPO/.git" ]]; then
  echo 'Cloning official xArm ROS 2 Jazzy source (requires Internet on Ubuntu host)...'
  mkdir -p "$WS/src"
  git clone --depth 1 --branch jazzy https://github.com/xArm-Developer/xarm_ros2.git "$REPO"
else
  echo 'Using existing official source checkout; no automatic updates.'
fi
( cd "$REPO" && git rev-parse HEAD ) > "$WS/xarm_ros2.commit"

echo 'Building UF850 Docker image (separate from baseline)...'
docker compose build
docker compose -f compose.yaml -f compose.uf850.yaml build sim

echo 'Building only xarm_description; xarm_api/real hardware drivers are NOT launched.'
docker compose -f compose.yaml -f compose.uf850.yaml run --rm --no-deps tools bash -c '
  set -eo pipefail
  source /opt/ros/jazzy/setup.bash
  cd /workspace/external/xarm_ws
  colcon build --base-paths src/xarm_ros2 --packages-select xarm_description --symlink-install --event-handlers console_direct+
  source install/setup.bash
  ros2 pkg prefix xarm_description
  XACRO=/workspace/simulation/uf850_description_wrapper.xacro
  mkdir -p /workspace/simulation/generated
  ros2 run xacro xacro "$XACRO" -o /workspace/simulation/generated/uf850_raw.urdf
  /usr/bin/python3 /workspace/scripts/prepare_uf850_model.py \
    /workspace/simulation/generated/uf850_raw.urdf \
    /workspace/simulation/generated/uf850_vgc10.urdf
  /usr/bin/python3 /workspace/scripts/make_uf850_world.py
'

echo 'Generated world and model. Source revision:'
cat "$WS/xarm_ros2.commit"
echo 'Next: docker compose -f compose.yaml -f compose.uf850.yaml up -d'
