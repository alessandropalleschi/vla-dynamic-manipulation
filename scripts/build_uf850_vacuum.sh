#!/usr/bin/env bash
# VGC10 build: Python dependencies and installer run inside the UF850 container.
# No PyYAML installation required in the host's Conda environment.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f compose.uf850.yaml || ! -f docker/Dockerfile.uf850 ]]; then
  echo 'Missing UF850 Compose overlay / Dockerfile. Install the UF850 patch first.' >&2
  exit 1
fi

DOCKERFILE=docker/Dockerfile.uf850
if [[ ! -e "${DOCKERFILE}.pre_vacuum_backup" ]]; then
  cp "$DOCKERFILE" "${DOCKERFILE}.pre_vacuum_backup"
fi

# Idempotently add missing packages to the UF850 image definition.
if ! grep -q 'ros-jazzy-gz-sim-vendor' "$DOCKERFILE"; then
  cat >> "$DOCKERFILE" <<'DOCKER'

# Gazebo Harmonic development dependencies for the VGC10 C++ system plugin.
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake g++ pkg-config ros-jazzy-gz-sim-vendor ros-jazzy-gz-plugin-vendor \
    ros-jazzy-gz-transport-vendor ros-jazzy-gz-msgs-vendor \
    && rm -rf /var/lib/apt/lists/*
DOCKER
fi
if ! grep -q 'python3-yaml' "$DOCKERFILE"; then
  cat >> "$DOCKERFILE" <<'DOCKER'

# PyYAML for the VGC10 bridge/world installer (container system Python).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-yaml && rm -rf /var/lib/apt/lists/*
DOCKER
fi

COMPOSE_BASE=(docker compose -f compose.yaml -f compose.uf850.yaml)
echo 'Building UF850 Docker image, including PyYAML and Gazebo plugin headers...'
"${COMPOSE_BASE[@]}" build sim

echo 'Patching generated world and bridge INSIDE the container...'
"${COMPOSE_BASE[@]}" run --rm --no-deps --entrypoint bash tools -lc '
  set -eo pipefail
  /usr/bin/python3 -c "import yaml; print(\"PASS: PyYAML available in UF850 container:\", yaml.__version__)"
  /usr/bin/python3 /workspace/scripts/install_uf850_vacuum.py --root /workspace
'

COMPOSE_VAC=(docker compose -f compose.yaml -f compose.uf850.yaml -f compose.uf850.vacuum.yaml)
echo 'Compiling VGC10 plugin inside the Gazebo container...'
"${COMPOSE_VAC[@]}" run --rm --no-deps --entrypoint bash sim -lc '
  set -eo pipefail
  source /opt/ros/jazzy/setup.bash
  cmake -S /workspace/simulation/plugins/vgc10 -B /workspace/simulation/plugins/vgc10/build \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build /workspace/simulation/plugins/vgc10/build -j2
  test -s /workspace/simulation/plugins/vgc10/build/libVgc10Vacuum.so
  echo "PASS: VGC10 Gazebo plugin built"
'
echo 'Build complete. Start with:'
echo 'docker compose -f compose.yaml -f compose.uf850.yaml -f compose.uf850.vacuum.yaml up -d --force-recreate'
