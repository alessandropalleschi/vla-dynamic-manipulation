#!/usr/bin/env bash
# Run: docker compose exec tools bash /workspace/scripts/diagnose.sh
set -e
printf '\n== Python selection ==\n'
command -v python3 || true
python3 --version || true
/usr/bin/python3 --version
printf '\n== ROS environment ==\n'
printf 'ROS_DISTRO=%s\n' "${ROS_DISTRO:-<unset>}"
printf 'PYTHONPATH=%s\n' "${PYTHONPATH:-<unset>}"
[ -r /opt/ros/jazzy/setup.bash ] && echo 'ROS setup file exists'
printf '\n== System Python ROS imports ==\n'
/usr/bin/python3 - <<'PYTHON_DIAGNOSTIC'
import sys
print('interpreter:', sys.executable)
print('sys.path:', sys.path)
try:
    import rclpy
    from sensor_msgs.msg import Image
    print('PASS: direct exec rclpy at', rclpy.__file__)
except Exception as exc:
    print('FAIL:', type(exc).__name__ + ':', exc)
    raise
PYTHON_DIAGNOSTIC
printf '\n== Sourced ROS check (checks both execution environments) ==\n'
bash -c 'source /opt/ros/jazzy/setup.bash && /usr/bin/python3 -c "import rclpy; print(\"PASS: sourced rclpy at\", rclpy.__file__)"'
printf '\n== ROS topics (may be empty if simulation is not running) ==\n' 
ros2 topic list || true
