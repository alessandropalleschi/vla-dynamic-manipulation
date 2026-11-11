FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# Headless OpenGL (EGL, software fallback via Mesa llvmpipe) for MuJoCo camera rendering,
# plus ffmpeg for encoding videos.
# The base image ships a Yarn apt source whose signing key has expired, which makes `apt-get update` fail with
# NO_PUBKEY. We do not use Yarn, so drop that source first.
RUN rm -f /etc/apt/sources.list.d/yarn.list /etc/apt/sources.list.d/yarn.sources \
    && apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
       libegl1 libgl1 libglx-mesa0 libgl1-mesa-dri libosmesa6 libglfw3 ffmpeg git-lfs \
    && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=egl
