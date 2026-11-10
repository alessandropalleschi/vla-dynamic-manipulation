#!/usr/bin/env python3
"""Non-motion test of simulation-only VGC10 vacuum interface.

Leaves robot and boxes where they are; verifies command/state pathway.
Always disables vacuum on exit. Cannot assert grasp until tool is near box.
"""
import sys
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64, String


class Probe(Node):
    def __init__(self):
        super().__init__('vgc10_vacuum_smoke')
        self.last = {}
        self.pa = self.create_publisher(Bool, '/vgc10/channel_a/enable', 10)
        self.pb = self.create_publisher(Bool, '/vgc10/channel_b/enable', 10)
        for topic, typ in [('/vgc10/attached', Bool),
                           ('/vgc10/status', String),
                           ('/vgc10/diagnostics', String),
                           ('/vgc10/cup_contacts', String),
                           ('/vgc10/grasp_quality', Float64)]:
            self.create_subscription(typ, topic,
                lambda m, k=topic: self.last.__setitem__(k, m.data), 10)

    def publish(self, a, b, duration=0.7):
        end = time.monotonic() + duration
        while time.monotonic() < end:
            self.pa.publish(Bool(data=bool(a)))
            self.pb.publish(Bool(data=bool(b)))
            rclpy.spin_once(self, timeout_sec=0.08)


def main():
    rclpy.init()
    p = Probe()
    try:
        p.publish(False, False, 2.0)
        if '/vgc10/status' not in p.last:
            raise RuntimeError('No Gazebo /vgc10/status: plugin/bridge not running')
        if p.last.get('/vgc10/attached') is not False:
            raise RuntimeError('Unexpected attached state with vacuum OFF')
        print('Start:', p.last)
        p.publish(True, True, 1.0)
        print('Channels ON:', p.last)
        # Far from box -> 0 contacts, 0 attachment. Close to box -> may attach.
        # Do not assume tool's initial pose is distant from target.
        p.publish(False, False, 1.0)
        if p.last.get('/vgc10/attached') is not False:
            raise RuntimeError('Failed to release after vacuum OFF')
        print('Channels OFF:', p.last)
        print('PASS: VGC10 commands, Gazebo state, and release interface work.')
    finally:
        try: p.publish(False, False, .4)
        except Exception: pass
        p.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        print('FAIL:', exc, file=sys.stderr)
        raise SystemExit(1)
