#!/usr/bin/env python3
"""
Control main loop for Highbay Track.
Adapted from MP2 main.py.

Usage:
    # Terminal 1
    ros2 launch gem_launch gem_init.launch.py

    # Terminal 2 (after recording waypoints)
    python3 src/control/scripts/main.py
"""

import math
import os
import sys
import signal
import atexit

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDrive

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controller_with_pid import VehicleController
import util

# ── Load waypoints ─────────────────────────────────────────────────────────────
# Once you have run waypoint_recorder.py and generated highbay_waypoints.py,
# this import will work. Until then it will raise an ImportError telling you
# to record waypoints first.
try:
    from highbay_waypoints import WayPoints
except ImportError:
    print(
        '\nERROR: highbay_waypoints.py not found.\n'
        'Please record waypoints first:\n'
        '  python3 src/control/scripts/waypoint_recorder.py\n'
    )
    sys.exit(1)


class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        print('ControlNode: Initializing...')

        self.current_odom = None
        self.shutting_down = False

        # Subscribe to odometry
        self.create_subscription(Odometry, 'odom', self._odom_callback, 10)

        # Controller
        self.controller = VehicleController(node=self)

        # Load waypoints
        waypoints = WayPoints()
        self.waypoints = waypoints.getWayPoints()
        print(f'ControlNode: Loaded {len(self.waypoints)} waypoints.')

        # Timing
        self.start_time     = self.get_clock().now()
        self.prev_plot_time = self.start_time

        # Per-metric storage for visualization
        self.speed_times = []
        self.speeds      = []
        self.xte_times   = []
        self.xte_vals    = []
        self.he_times    = []
        self.he_vals     = []

        # Run at 100 Hz — same as MP2
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

        # Run controller
        self.controller.execute(odom, self.waypoints)

        speed = self.controller.speed

        # Print status every 5 seconds
        if not hasattr(self, '_last_print') or (elapsed - self._last_print) >= 5.0:
            self._last_print = elapsed
            wp_idx = self.controller.current_wp_idx
            print(
                f'[{elapsed:.1f}s] speed: {speed:.2f} m/s  '
                f'wp: {wp_idx}/{len(self.waypoints)}  '
                f'steering: {self.controller.steering:.3f} rad'
            )

        # Visualization plots
        util.visualization(
            self, cur_time,
            speed=speed,
            xte=self.controller.xte,
            he=math.degrees(self.controller.he)
        )

        # Check if all waypoints reached
        if self.controller.current_wp_idx >= len(self.waypoints) - 1:
            print(f'\nCompleted track in {elapsed:.1f}s!')
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
