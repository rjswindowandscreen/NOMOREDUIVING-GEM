#!/usr/bin/env python3
"""
Control main loop — Stanley + slow-speed + GPS goal stop.

Usage:
    # Terminal 1
    ros2 launch gem_launch gem_init.launch.py

    # Terminal 2
    python3 src/control/run_control.py
"""

import math
import os
import sys
import signal
import atexit

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stanley_controller import StanleyController
import util


class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        print('ControlNode: Initializing...')

        self.current_odom = None
        self.shutting_down = False

        # Localization output (fused IMU + GPS + INS)
        self.create_subscription(Odometry, '/odometry/global', self._odom_callback, 10)

        # Controller
        self.controller = StanleyController(node=self)

        # Timing for viz
        self.start_time     = self.get_clock().now()
        self.prev_plot_time = self.start_time

        # Per-metric storage for visualization
        self.speed_times = []
        self.speeds      = []
        self.xte_times   = []
        self.xte_vals    = []
        self.he_times    = []
        self.he_vals     = []

        # Run at 100 Hz
        self.timer = self.create_timer(0.01, self._run_loop)
        print('ControlNode: Ready. Running at 100 Hz.')

    def _odom_callback(self, msg):
        self.current_odom = msg

    def _run_loop(self):
        if self.shutting_down:
            self.controller.stop()
            return

        if self.current_odom is None:
            return

        odom = self.current_odom
        cur_time = self.get_clock().now()
        elapsed  = (cur_time - self.start_time).nanoseconds / 1e9

        self.controller.execute(odom)

        speed = self.controller.speed

        # Status print every 5 s
        if not hasattr(self, '_last_print') or (elapsed - self._last_print) >= 5.0:
            self._last_print = elapsed
            print(
                f'[{elapsed:.1f}s] speed: {speed:.2f} m/s  '
                f'steer: {self.controller.steering:.3f} rad  '
                f'XTE: {self.controller.xte:.2f}  '
                f'goal_dist: {self.controller.last_goal_dist:.1f} m'
            )

        # Visualization plots
        util.visualization(
            self, cur_time,
            speed=speed,
            xte=self.controller.xte,
            he=math.degrees(self.controller.he),
        )

        # Arrival: near goal and stopped → exit
        if self.controller.near_goal and abs(speed) < 0.05:
            print(f'\nArrived at goal in {elapsed:.1f}s '
                  f'(distance {self.controller.last_goal_dist:.2f} m).')
            self.controller.stop()
            os._exit(0)

    def stop_vehicle(self):
        if not self.shutting_down:
            self.shutting_down = True
            self.controller.stop()
            print('Vehicle stopped.')

    def destroy_node(self):
        self.stop_vehicle()
        super().destroy_node()


global_node = None


def signal_handler(signum, frame):
    global global_node
    print(f'\nReceived signal {signum}, shutting down...')
    if global_node is not None:
        global_node.stop_vehicle()
    rclpy.shutdown()


def main():
    rclpy.init()

    global global_node
    node = None
    try:
        node = ControlNode()
        global_node = node

        signal.signal(signal.SIGINT,  signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        atexit.register(lambda: node.stop_vehicle() if node else None)

        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nKeyboard interrupt.')
    except Exception as e:
        print(f'Error: {e}')
        import traceback; traceback.print_exc()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
