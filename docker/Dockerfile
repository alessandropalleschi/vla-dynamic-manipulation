FROM ros:jazzy-ros-base-noble
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=jazzy \
    PYTHONUNBUFFERED=1
# Docker Compose exec creates a new process with image/container creation-time env.
# It does NOT inherit `source /opt/ros/jazzy/setup.bash` from our entrypoint.
# Bake the essential runtime paths into the IMAGE so direct `docker compose exec
# tools /usr/bin/python3 ...` and `docker compose exec tools ros2 ...` both work.
ENV PATH="/opt/ros/jazzy/bin:${PATH}" \
    PYTHONPATH="/opt/ros/jazzy/lib/python3.12/site-packages" \
    LD_LIBRARY_PATH="/opt/ros/jazzy/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu" \
    AMENT_PREFIX_PATH="/opt/ros/jazzy" \
    CMAKE_PREFIX_PATH="/opt/ros/jazzy"
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-ros-gz \
    ros-jazzy-rclpy \
    ros-jazzy-sensor-msgs \
    ros-jazzy-std-msgs \
    ros-jazzy-rosgraph-msgs \
    ros-jazzy-rosbag2 \
    ros-jazzy-rosbag2-storage-mcap \
    python3-yaml python3-pytest python3-pil \
    libegl1 libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*
# Also test the exact UNSOURCED exec environment used by users.
RUN /usr/bin/python3 -c "import rclpy; from sensor_msgs.msg import Image; from rosgraph_msgs.msg import Clock; from PIL import Image as PILImage; print('PASS: direct ROS Python imports')" && \
    ros2 --help >/dev/null
RUN source /opt/ros/jazzy/setup.bash && \
    /usr/bin/python3 -c "import rclpy; print('PASS: sourced ROS Python imports')"
WORKDIR /workspace
COPY scripts/entrypoint.sh /usr/local/bin/conveyor_entrypoint
RUN chmod +x /usr/local/bin/conveyor_entrypoint
ENTRYPOINT ["/usr/local/bin/conveyor_entrypoint"]
CMD ["bash"]
