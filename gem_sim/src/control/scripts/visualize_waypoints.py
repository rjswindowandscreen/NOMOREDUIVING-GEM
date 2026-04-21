#!/usr/bin/env python3
"""
Publishes highbay waypoints as RViz markers so you can verify
they actually align with the track in Gazebo.

Usage:
    python3 src/control/scripts/visualize_waypoints.py

Then in RViz: Add -> By topic -> /waypoint_markers -> MarkerArray
"""

import os
import sys
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from noisy_waypoints import WayPoints
except ImportError:
    print('ERROR: highbay_waypoints.py not found')
    sys.exit(1)


class WaypointVisualizer(Node):
    def __init__(self):
        super().__init__('waypoint_visualizer')

        self.pub = self.create_publisher(MarkerArray, '/waypoint_markers', 10)

        waypoints = WayPoints().getWayPoints()

        # Build marker array
        marker_array = MarkerArray()

        # Line strip connecting all waypoints
        line = Marker()
        line.header.frame_id = 'odom'
        line.ns   = 'waypoints'
        line.id   = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.2   # line width
        line.color.r = 0.0
        line.color.g = 1.0
        line.color.b = 0.0
        line.color.a = 1.0
        line.pose.orientation.w = 1.0

        for x, y in waypoints:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.3   # slightly above ground so it's visible
            line.points.append(p)

        marker_array.markers.append(line)

        # Numbered spheres every 10 waypoints
        for i, (x, y) in enumerate(waypoints):
            if i % 10 != 0:
                continue
            sphere = Marker()
            sphere.header.frame_id = 'odom'
            sphere.ns   = 'waypoint_numbers'
            sphere.id   = i + 1000
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(x)
            sphere.pose.position.y = float(y)
            sphere.pose.position.z = 0.5
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 1.0
            sphere.scale.y = 1.0
            sphere.scale.z = 1.0
            sphere.color.r = 1.0
            sphere.color.g = 0.0
            sphere.color.b = 0.0
            sphere.color.a = 1.0
            marker_array.markers.append(sphere)

        # Publish at 1Hz so RViz always has fresh markers
        self.marker_array = marker_array
        self.create_timer(1.0, self.publish)
        self.get_logger().info(
            f'Publishing {len(waypoints)} waypoints on /waypoint_markers. '
            'Open RViz and add MarkerArray topic to verify alignment.'
        )

    def publish(self):
        now = self.get_clock().now().to_msg()
        for m in self.marker_array.markers:
            m.header.stamp = now
        self.pub.publish(self.marker_array)


def main():
    rclpy.init()
    node = WaypointVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()