#!/usr/bin/env python3
"""
Waypoint Recorder for Highbay Track.

Drive the car manually around the track with drive.py while this script
is running. It records your position from /odom and saves the waypoints
to a file when you press Ctrl+C.

Usage:
    # Terminal 1: start sim
    ros2 launch gem_launch gem_init.launch.py

    # Terminal 2: start teleop
    python3 src/gem_simulator/gem_gazebo/scripts/drive.py

    # Terminal 3: record waypoints
    python3 src/control/scripts/waypoint_recorder.py

    When done driving, press Ctrl+C. The waypoints will be saved to
    src/control/scripts/highbay_waypoints.py
"""

import os
import sys
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


# Minimum distance between recorded waypoints (meters).
# Too small = too many waypoints, noisy path.
# Too large = corners get cut.
MIN_WAYPOINT_DIST = 0.5


class WaypointRecorder(Node):
    def __init__(self):
        super().__init__('waypoint_recorder')

        self._waypoints = []
        self._last_x = None
        self._last_y = None

        self.create_subscription(Odometry, '/odom', self._odom_callback, 10)
        self.get_logger().info(
            'Waypoint recorder started. Drive the car around the track. '
            'Press Ctrl+C when done.'
        )

    def _odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Only record if we've moved far enough from the last waypoint
        if self._last_x is not None:
            dist = math.hypot(x - self._last_x, y - self._last_y)
            if dist < MIN_WAYPOINT_DIST:
                return

        self._waypoints.append((x, y))
        self._last_x = x
        self._last_y = y

        if len(self._waypoints) % 10 == 0:
            print(f'Recorded {len(self._waypoints)} waypoints...')

    def save(self):
        if not self._waypoints:
            print('No waypoints recorded.')
            return

        # Save as a Python file in the same format as MP2 waypoint files
        # so the controller can load it directly
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'highbay_waypoints.py'
        )

        with open(out_path, 'w') as f:
            f.write('#!/usr/bin/env python3\n')
            f.write('"""Auto-generated highbay waypoints. Do not edit manually."""\n\n')
            f.write('class WayPoints:\n')
            f.write('    def __init__(self):\n')
            f.write('        self.waypoints = [\n')
            for x, y in self._waypoints:
                f.write(f'            ({x:.4f}, {y:.4f}),\n')
            f.write('        ]\n\n')
            f.write('    def getWayPoints(self):\n')
            f.write('        return self.waypoints\n')

        print(f'\nSaved {len(self._waypoints)} waypoints to {out_path}')
        print('You can now run the controller with these waypoints.')


def main():
    rclpy.init()
    node = WaypointRecorder()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nStopping recorder...')
    finally:
        node.save()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
