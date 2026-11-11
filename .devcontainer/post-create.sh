#!/usr/bin/env bash
# Runs once after the dev container is created. TORCH_VARIANT=cuda|cpu is set by the devcontainer config.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install --upgrade pip
git lfs install --local --force && git lfs pull || echo "(git lfs pull failed: meshes may be pointer files; see README)"

# NVIDIA's EGL vendor ICD: the container toolkit normally installs it when the `graphics` capability is
# requested; create it if the driver library is present but the ICD file is not.
if [ "${TORCH_VARIANT:-cpu}" = "cuda" ]; then
  EGL_LIB=$(ls /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 2>/dev/null || true)
  if [ -n "$EGL_LIB" ] && ! ls /usr/share/glvnd/egl_vendor.d/*nvidia*.json >/dev/null 2>&1; then
    sudo mkdir -p /usr/share/glvnd/egl_vendor.d
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
      | sudo tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json >/dev/null
  fi
  # The default PyPI torch wheel targets the newest CUDA (13.x) and fails with "driver too old" on CUDA-12 hosts.
  # Pick the wheel from the CUDA version the host driver reports (CUDA 12.x minor-version compatibility lets the
  # cu126 wheel run on any 12.x driver >= 525).
  CUDA_VER=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\)\..*/\1/p' | head -1)
  case "${CUDA_VER:-}" in
    12) python -m pip install torch --index-url https://download.pytorch.org/whl/cu126 ;;
    *)  python -m pip install torch ;;   # CUDA 13+ driver, or no driver visible at build time
  esac
else
  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
python -m pip install -e ".[dev]"
PYTHON=python bash .devcontainer/install-hooks.sh   # fails loudly if the commit guard is not active
python -m pytest
python -m vla_mujoco.gpu_check || echo "(no usable CUDA device - training will fall back to CPU)"
echo "Ready. Try:  make demo"
