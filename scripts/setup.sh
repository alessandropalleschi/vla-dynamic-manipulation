#!/usr/bin/env bash
# Rebuild the UF850 robot description and Gazebo VGC10 suction plugin.
# Run on the Linux host from anywhere. No Docker/Compose configuration changes.
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

COMPOSE_FILES=(
  compose.yaml
  compose.uf850.yaml
  compose.uf850.vacuum.yaml
  compose.uf850.moveit.yaml
)

for file in "${COMPOSE_FILES[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "ERROR: missing Compose file: $PROJECT_ROOT/$file" >&2
    exit 2
  fi
done

for command in docker id; do
  if ! command -v "$command" >/dev/null; then
    echo "ERROR: '$command' is not installed on this host." >&2
    exit 2
  fi
done

if [[ "$(id -u)" -eq 0 ]]; then
  echo "ERROR: run as your normal host user, not with sudo." >&2
  exit 2
fi

# Intentionally do NOT apply compose.user.yaml. The verified working baseline
# runs sim and moveit using the original Compose configuration. Only the
# temporary build container gets the host UID/GID so generated files are owned
# by the invoking user.
COMPOSE=(docker compose)
for file in "${COMPOSE_FILES[@]}"; do
  COMPOSE+=(-f "$file")
done

"${COMPOSE[@]}" config --quiet

run_build() {
  "${COMPOSE[@]}" run --rm -T --no-deps \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e XDG_CACHE_HOME=/tmp/vla-build-cache \
    --entrypoint /bin/bash sim -lc "$1"
}

if [[ "${1:-}" == "--verify-only" ]]; then
  echo '== Check robot-description meshes and VGC10 plugin (no build) =='
  run_build '
    set -Eeuo pipefail
    set +u  # ROS 2 environment scripts may reference unset AMENT_TRACE_SETUP_FILES
  source /opt/ros/jazzy/setup.bash
  set -u
    root=/workspace
    mesh="$root/external/xarm_ws/install/xarm_description/share/xarm_description/meshes/uf850"
    for kind in visual collision; do
      for link in link_base link1 link2 link3 link4 link5 link6; do
        test -s "$mesh/$kind/$link.stl" || { echo "MISSING mesh: $mesh/$kind/$link.stl" >&2; exit 3; }
      done
    done
    lib="$root/simulation/plugins/vgc10/build/libVgc10Vacuum.so"
    test -s "$lib" || { echo "MISSING plugin: $lib" >&2; exit 3; }
    if ldd "$lib" | grep "not found"; then
      echo "ERROR: Gazebo plugin has missing shared libraries" >&2
      exit 3
    fi
    echo "PASS: 14 UF850 meshes and VGC10 shared library are accessible"
  '
  exit 0
fi

if [[ $# -ne 0 ]]; then
  echo "Usage: bash scripts/setup.sh [--verify-only]" >&2
  exit 2
fi

echo '== Building xarm_description in the project-mounted ROS workspace =='
run_build '
  set -Eeuo pipefail
  set +u  # ROS 2 environment scripts may reference unset AMENT_TRACE_SETUP_FILES
  source /opt/ros/jazzy/setup.bash
  set -u
  cd /workspace/external/xarm_ws
  test -d src || { echo "MISSING: xarm workspace source (external/xarm_ws/src)" >&2; exit 3; }
  command -v colcon >/dev/null || { echo "MISSING: colcon inside sim image" >&2; exit 3; }
  colcon list --names-only | grep -qx xarm_description || {
    echo "MISSING: xarm_description package under external/xarm_ws/src" >&2
    exit 3
  }
  colcon build --symlink-install --packages-up-to xarm_description \
    --event-handlers console_direct+
  mesh=/workspace/external/xarm_ws/install/xarm_description/share/xarm_description/meshes/uf850
  for kind in visual collision; do
    for link in link_base link1 link2 link3 link4 link5 link6; do
      test -s "$mesh/$kind/$link.stl" || { echo "MISSING: $mesh/$kind/$link.stl" >&2; exit 3; }
    done
  done
  echo "PASS: xarm_description built and all 14 UF850 STL meshes resolve"
'

echo '== Building VGC10 plugin against ROS Jazzy Gazebo Harmonic vendors =='
run_build '
  set -Eeuo pipefail
  set +u  # ROS 2 environment scripts may reference unset AMENT_TRACE_SETUP_FILES
  source /opt/ros/jazzy/setup.bash
  set -u
  cd /workspace/simulation/plugins/vgc10
  test -f CMakeLists.txt && test -f Vgc10Vacuum.cc || {
    echo "MISSING: VGC10 CMakeLists.txt or Vgc10Vacuum.cc" >&2; exit 3;
  }
  cmake -S . -B build.setup -DCMAKE_BUILD_TYPE=Release
  cmake --build build.setup --parallel 2
  candidate=build.setup/libVgc10Vacuum.so
  test -s "$candidate" || { echo "MISSING built VGC10 plugin" >&2; exit 3; }
  if ldd "$candidate" | grep "not found"; then
    echo "ERROR: Gazebo plugin has missing shared libraries" >&2
    exit 3
  fi
  mkdir -p build
  # Build and verify before replacing the verified location. Atomic rename
  # avoids leaving a partly copied .so in the simulation plugin directory.
  tmp="build/.libVgc10Vacuum.so.new.$$"
  trap '\''rm -f "$tmp"'\'' EXIT
  install -m 755 "$candidate" "$tmp"
  mv -f "$tmp" build/libVgc10Vacuum.so
  echo "PASS: VGC10 plugin rebuilt and installed"
'

echo '== Verify rebuilt artifacts =='
run_build '
  set -Eeuo pipefail
  set +u  # ROS 2 environment scripts may reference unset AMENT_TRACE_SETUP_FILES
  source /opt/ros/jazzy/setup.bash
  set -u
  mesh=/workspace/external/xarm_ws/install/xarm_description/share/xarm_description/meshes/uf850
  for kind in visual collision; do
    for link in link_base link1 link2 link3 link4 link5 link6; do
      test -s "$mesh/$kind/$link.stl" || exit 3
    done
  done
  lib=/workspace/simulation/plugins/vgc10/build/libVgc10Vacuum.so
  test -s "$lib"
  if ldd "$lib" | grep "not found"; then exit 3; fi
  echo "PASS: reproducible ROS + Gazebo plugin build verified"
'

echo
printf '%s\n' \
  'SETUP COMPLETE' \
  'This script did not recreate your running sim/moveit containers.' \
  'When ready to restart the simulator, use your existing verified v9 launcher.'
