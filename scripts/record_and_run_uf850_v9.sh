#!/usr/bin/env bash
# Reset the simulation, then run the unchanged v9 expert with overhead recording.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
C=(docker compose -f compose.yaml -f compose.uf850.yaml -f compose.uf850.vacuum.yaml -f compose.uf850.moveit.yaml -f compose.user.yaml)

/usr/bin/python3 scripts/install_uf850_overhead_camera.py
if [[ ! -f scripts/uf850_moving_stack_v9.py ]]; then
  echo 'BLOCKED: working scripts/uf850_moving_stack_v9.py missing'; exit 2
fi
EP="uf850_v9_$(date +%Y%m%d_%H%M%S)"
trap '"${C[@]}" exec -T sim bash -lc '\''gz topic -t /conveyor/speed -m gz.msgs.Double -p "data: 0"'\'' >/dev/null 2>&1 || true' EXIT

echo '== Reset robot, boxes and conveyor =='
"${C[@]}" up -d --force-recreate sim moveit

echo '== Check overhead camera was created =='
"${C[@]}" exec -T sim bash -lc '
  source /opt/ros/jazzy/setup.bash
  for attempt in $(seq 1 45); do
    if gz topic -l | grep -qx /station/overhead/image; then
      echo "PASS: Gazebo camera publisher available"
      exit 0
    fi
    sleep 1
  done
  echo "BLOCKED: Gazebo camera topic missing; inspect sim logs / Sensors plugin"
  exit 2
'

echo '== Verify starting box poses and create a fresh planning snapshot =='
"${C[@]}" exec -T sim bash -lc '
  source /opt/ros/jazzy/setup.bash
  gz topic -t /conveyor/speed -m gz.msgs.Double -p "data: 0.0"
  /usr/bin/python3 /workspace/scripts/check_gz_box_pose_v7.py
'

echo "== Record images, labels and live v9 manipulation: $EP =="
"${C[@]}" exec -T -e "EP=$EP" moveit bash -lc '
  source /opt/ros/jazzy/setup.bash
  export UF850_VEL_SCALE=0.60 UF850_ACC_SCALE=0.60
  /usr/bin/python3 /workspace/scripts/record_uf850_vla_episode.py \
    --out "/workspace/data/vla_demos/$EP" --fps 5
'

echo '== Verify physics and preserve the result in this episode =='
"${C[@]}" exec -T -e "EP=$EP" sim bash -lc '
  source /opt/ros/jazzy/setup.bash
  /usr/bin/python3 /workspace/scripts/verify_gz_stack_v7.py \
    --out "/workspace/data/vla_demos/$EP/stack_verification.json"
'

echo "PASS: Recorded moving-stack episode: data/vla_demos/$EP"
