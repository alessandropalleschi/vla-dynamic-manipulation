#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
bash scripts/setup_local_user.sh
C=(docker compose -f compose.yaml -f compose.uf850.yaml -f compose.uf850.vacuum.yaml -f compose.uf850.moveit.yaml)
[[ ! -f compose.user.yaml ]] || C+=(-f compose.user.yaml)
stop_belt() { "${C[@]}" exec -T sim bash -lc 'gz topic -t /conveyor/speed -m gz.msgs.Double -p "data: 0.0"' >/dev/null 2>&1 || true; }
trap stop_belt EXIT
"${C[@]}" up -d --force-recreate sim moveit
"${C[@]}" exec -T sim bash -lc 'source /opt/ros/jazzy/setup.bash; gz topic -t /conveyor/speed -m gz.msgs.Double -p "data: 0"; /usr/bin/python3 /workspace/scripts/check_gz_box_pose_v7.py'
"${C[@]}" exec -T \
  -e "VLA_BELT_SPEED_M_S=${VLA_BELT_SPEED_M_S:-0.020}" \
  -e "VLA_WARMUP_SIM_S=${VLA_WARMUP_SIM_S:-0}" \
  -e "VLA_IK_SEED=${VLA_IK_SEED:-91}" \
  moveit bash -lc 'source /opt/ros/jazzy/setup.bash; /usr/bin/python3 /workspace/scripts/expert_moving_stack.py --ack-simulation-only'
"${C[@]}" exec -T sim bash -lc 'source /opt/ros/jazzy/setup.bash; /usr/bin/python3 /workspace/scripts/verify_gz_stack_v7.py'
echo 'PASS: moving stack'
