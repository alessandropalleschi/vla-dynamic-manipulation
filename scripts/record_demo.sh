#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
bash scripts/setup_local_user.sh
C=(docker compose -f compose.yaml -f compose.uf850.yaml -f compose.uf850.vacuum.yaml -f compose.uf850.moveit.yaml)
[[ ! -f compose.user.yaml ]] || C+=(-f compose.user.yaml)
stop_belt() { "${C[@]}" exec -T sim bash -lc 'gz topic -t /conveyor/speed -m gz.msgs.Double -p "data: 0.0"' >/dev/null 2>&1 || true; }
trap stop_belt EXIT
/usr/bin/python3 scripts/setup_camera.py
EP="${VLA_EPISODE_NAME:-demo_$(date +%Y%m%d_%H%M%S)}"
"${C[@]}" up -d --force-recreate sim moveit

echo "Waiting for MoveIt planning service..."

ready=0

for i in $(seq 1 45); do
    if "${C[@]}" exec -T moveit bash -lc '
        source /opt/ros/jazzy/setup.bash
        ros2 service list | grep -qx /plan_kinematic_path
    ' >/dev/null 2>&1; then
        ready=1
        break
    fi

    sleep 1
done

if [[ "$ready" -ne 1 ]]; then
    echo "ERROR: MoveIt planning service unavailable"
    "${C[@]}" ps -a
    "${C[@]}" logs --tail=50 moveit
    exit 1
fi

echo "PASS: MoveIt planning service ready"
"${C[@]}" exec -T sim bash -lc 'source /opt/ros/jazzy/setup.bash; for i in $(seq 1 45); do gz topic -l | grep -qx /station/overhead/image && exit 0; sleep 1; done; echo "No camera topic"; exit 2'
"${C[@]}" exec -T sim bash -lc 'source /opt/ros/jazzy/setup.bash; gz topic -t /conveyor/speed -m gz.msgs.Double -p "data: 0"; /usr/bin/python3 /workspace/scripts/check_gz_box_pose_v7.py'
# VLA_ROS_READY_CHECK
echo "== Waiting for ROS clock and joint states =="

"${C[@]}" exec -T moveit bash -lc '
source /opt/ros/jazzy/setup.bash

for attempt in $(seq 1 12); do
    if timeout 8 ros2 topic echo /clock --once         --qos-reliability best_effort >/dev/null 2>&1 &&
       timeout 8 ros2 topic echo /joint_states --once         --qos-reliability best_effort >/dev/null 2>&1; then
        echo "PASS: ROS clock and joint states ready"
        exit 0
    fi

    sleep 1
done

echo "BLOCKED: ROS streams not ready" >&2
exit 2
'

"${C[@]}" exec -T \
  -e "VLA_BELT_SPEED_M_S=${VLA_BELT_SPEED_M_S:-0.020}" \
  -e "VLA_WARMUP_SIM_S=${VLA_WARMUP_SIM_S:-0}" \
  -e "VLA_IK_SEED=${VLA_IK_SEED:-91}" \
  moveit bash -lc "source /opt/ros/jazzy/setup.bash; /usr/bin/python3 /workspace/scripts/record_episode.py --out /workspace/data/vla_demos/$EP --fps 5"
"${C[@]}" exec -T sim bash -lc "source /opt/ros/jazzy/setup.bash; /usr/bin/python3 /workspace/scripts/verify_gz_stack_v7.py --out /workspace/data/vla_demos/$EP/stack_verification.json"
echo "PASS: recorded episode $EP"
