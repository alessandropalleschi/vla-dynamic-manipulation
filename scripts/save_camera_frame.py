#!/usr/bin/python3
"""Save the next Gazebo external-camera ROS image to a PNG for framing checks."""
import pathlib
import sys
import time

import rclpy
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


OUTPUT = pathlib.Path('/workspace/data/camera_snapshot.png')


class Capture(Node):
    def __init__(self):
        super().__init__('camera_snapshot')
        self.done = False
        self.create_subscription(Image, '/camera/image_raw', self.on_image, qos_profile_sensor_data)

    def on_image(self, msg):
        if self.done:
            return
        encoding = msg.encoding.lower()
        if encoding == 'rgb8':
            image = PILImage.frombytes('RGB', (msg.width, msg.height), bytes(msg.data), 'raw', 'RGB', msg.step, 1)
        elif encoding == 'bgr8':
            image = PILImage.frombytes('RGB', (msg.width, msg.height), bytes(msg.data), 'raw', 'BGR', msg.step, 1)
        elif encoding == 'mono8':
            image = PILImage.frombytes('L', (msg.width, msg.height), bytes(msg.data), 'raw', 'L', msg.step, 1)
        else:
            self.get_logger().error(f'Unsupported image encoding: {msg.encoding}')
            self.done = True
            return
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        image.save(OUTPUT)
        self.get_logger().info(f'Saved {msg.width}x{msg.height} {msg.encoding} image to {OUTPUT}')
        self.done = True


def main():
    rclpy.init()
    node = Capture()
    deadline = time.monotonic() + 20
    while rclpy.ok() and not node.done and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
    ok = node.done and OUTPUT.exists()
    node.destroy_node()
    rclpy.shutdown()
    if not ok:
        print('No usable camera image received within 20s. Check ROS/Gazebo camera bridge.', file=sys.stderr)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
