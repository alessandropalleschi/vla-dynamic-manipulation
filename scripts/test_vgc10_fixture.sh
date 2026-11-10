#!/usr/bin/env bash
# Positive attachment/detachment integration test in a tiny dedicated Gazebo world.
# Does not touch the UF850 simulation or the ROS graph.
set -euo pipefail
cd "$(dirname "$0")/.."
COMP=(docker compose -f compose.yaml -f compose.uf850.yaml -f compose.uf850.vacuum.yaml)
"${COMP[@]}" run --rm --no-deps --entrypoint bash sim -lc '
  set -eo pipefail
  source /opt/ros/jazzy/setup.bash
  export GZ_PARTITION="vgc10_fixture_$$"
  export GZ_SIM_SYSTEM_PLUGIN_PATH=/workspace/simulation/plugins/vgc10/build
  LOG=/tmp/vgc10_fixture.log
  gz sim --force-version 8 -s -r -v 3 /workspace/simulation/tests/vgc10_fixture.sdf >"$LOG" 2>&1 &
  GZ_PID=$!
  trap "kill $GZ_PID 2>/dev/null || true" EXIT
  for i in $(seq 1 30); do
    if gz topic -l | grep -qx /vgc10/status; then break; fi
    sleep 1
  done
  if ! gz topic -l | grep -qx /vgc10/status; then
    tail -80 "$LOG"; echo "FAIL: VGC10 plugin not publishing" >&2; exit 1
  fi
  echo "Fixture running; vacuum off should mean detached..."
  before=$(timeout 4 gz topic -e -t /vgc10/status || true)
  echo "$before" | grep -q "DETACHED_NO_VALID_SEAL" || { echo "$before"; exit 1; }
  gz topic -t /vgc10/channel_a/enable -m gz.msgs.Boolean -p "data: true"
  gz topic -t /vgc10/channel_b/enable -m gz.msgs.Boolean -p "data: true"
  sleep 1
  after=$(timeout 2 gz topic -e -t /vgc10/attached || true)
  echo "$after" | grep -q "data: true" || {
    echo "FAIL: expected attachment with aligned tool and vacuum ON"
    gz topic -e -t /vgc10/diagnostics | head -15 || true
    tail -40 "$LOG"; exit 1;
  }
  gz topic -t /vgc10/channel_a/enable -m gz.msgs.Boolean -p "data: false"
  sleep 1
  released=$(timeout 4 gz topic -e -t /vgc10/status || true)
  echo "$released" | grep -q "DETACHED_NO_VALID_SEAL" || {
    echo "FAIL: release did not detach"; tail -40 "$LOG"; exit 1;
  }
  echo "PASS: cold start detached, aligned four-cup attachment, channel-off release"
'
