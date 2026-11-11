#!/usr/bin/env bash
# Runs once after the dev container is created. TORCH_VARIANT=cuda|cpu is set by the devcontainer config.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install --upgrade pip
git lfs install --local && git lfs pull || echo "(git lfs pull failed: meshes may be pointer files; see README)"

# NVIDIA's EGL vendor ICD: the container toolkit normally installs it when the `graphics` capability is
# requested; create it if the driver library is present but the ICD file is not.
if [ "${TORCH_VARIANT:-cpu}" = "cuda" ]; then
  EGL_LIB=$(ls /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 2>/dev/null || true)
  if [ -n "$EGL_LIB" ] && ! ls /usr/share/glvnd/egl_vendor.d/*nvidia*.json >/dev/null 2>&1; then
    sudo mkdir -p /usr/share/glvnd/egl_vendor.d
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
      | sudo tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json >/dev/null
  fi
  # The default Linux torch wheel on PyPI is a CUDA build (bundles the CUDA runtime; only the host driver is needed).
  python -m pip install torch
else
  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
python -m pip install -e ".[dev]"
PYTHON=python bash .devcontainer/install-hooks.sh   # fails loudly if the commit guard is not active
python -m pytest
python -m vla_mujoco.gpu_check || echo "(no usable CUDA device - training will fall back to CPU)"
echo "Ready. Try:  make demo"
