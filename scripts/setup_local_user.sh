#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

UID_HOST=$(id -u)
GID_HOST=$(id -g)

if [ "$UID_HOST" -eq 0 ]; then
    echo "ERROR: Run this script as your normal Ubuntu user, not root."
    exit 1
fi

if [ -f compose.user.yaml ]; then
    cp compose.user.yaml compose.user.yaml.before_uid_fix
fi

cat > compose.user.yaml <<YAML
services:
  sim:
    user: "$UID_HOST:$GID_HOST"
    environment:
      HOME: /tmp
      XDG_CACHE_HOME: /tmp/uf850-cache
      ROS_LOG_DIR: /tmp/uf850-ros

  moveit:
    user: "$UID_HOST:$GID_HOST"
    environment:
      HOME: /tmp
      XDG_CACHE_HOME: /tmp/uf850-cache
      ROS_LOG_DIR: /tmp/uf850-ros

  tools:
    user: "$UID_HOST:$GID_HOST"
    environment:
      HOME: /tmp
      XDG_CACHE_HOME: /tmp/uf850-cache
      ROS_LOG_DIR: /tmp/uf850-ros
YAML

echo "PASS: Docker services configured for UID:GID $UID_HOST:$GID_HOST"
