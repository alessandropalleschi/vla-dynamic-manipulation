#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
C=(docker compose -f compose.yaml -f compose.uf850.yaml -f compose.uf850.vacuum.yaml -f compose.uf850.moveit.yaml -f compose.user.yaml)

stop_belt() {
  "${C[@]}" exec -T sim bash -lc \
    'gz topic -t /conveyor/speed -m gz.msgs.Double -p "data: 0.0"' \
    >/dev/null 2>&1 || true
}
trap stop_belt EXIT

mkdir -p data/baselines/before_first_moving_stack
for f in data/uf850_stack_v7/stack_verification.json data/uf850_stack_v7/run_trace.jsonl; do
  if [[ -f "$f" ]]; then cp -n "$f" data/baselines/before_first_moving_stack/; fi
done

echo '== Reset Gazebo and MoveIt to the box starting positions =='
"${C[@]}" up -d --force-recreate sim moveit
sleep 8

echo '== Ensure conveyor stopped and verify starting box poses =='
"${C[@]}" exec -T sim bash -lc '
  source /opt/ros/jazzy/setup.bash
  gz topic -t /conveyor/speed -m gz.msgs.Double -p "data: 0.0"
  /usr/bin/python3 /workspace/scripts/check_gz_box_pose_v7.py
'

echo '== Plan each motion and execute moving-target stacking in Gazebo =='
"${C[@]}" exec -T moveit bash -lc '
  source /opt/ros/jazzy/setup.bash
  export UF850_VEL_SCALE=0.60
  export UF850_ACC_SCALE=0.60
  /usr/bin/python3 /workspace/scripts/uf850_moving_stack_v9.py --ack-simulation-only
'

echo '== Physics-verify the stack with conveyor stopped =='
"${C[@]}" exec -T sim bash -lc '
  source /opt/ros/jazzy/setup.bash
  /usr/bin/python3 /workspace/scripts/verify_gz_stack_v7.py
'
echo 'PASS: Moving-target trial and final Gazebo stack verification.'
