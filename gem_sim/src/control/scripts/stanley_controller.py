#!/usr/bin/env python3
"""
stanley_controller.py — Standalone pure Stanley lane-following node.

All topic names come from topics.py — never hardcoded here.

ROS2 parameters (set by run_all.py):
    mode         \'sim\' or \'real\'   default \'sim\'
    goal_lat     float             default 40.1164
    goal_lon     float             default -88.2434
    goal_radius  float (m)         default 2.0
"""

import math, os, sys, time
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32MultiArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stanley_math import stanley, update_speed
from util import quaternion_to_euler

# topics.py: control/scripts/ -> control/ -> src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from topics import get_topics

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "localization", "scripts"))
from gps_utils import latlon_to_xy

MAX_SPEED = 0.8; MAX_ACCEL = 0.2; STOP_DECEL = 0.6
STANLEY_K = 0.3; MAX_STEER = 0.61; LANE_STALE_AFTER = 0.5
XTE_SIGN = +1; HE_SIGN = +1
STOP_DIST = 5.0; SLOW_DIST = 9.0; OBSTACLE_LATERAL_GATE = 3.0


class StanleyNode(Node):
    def __init__(self):
        super().__init__("stanley_controller")
        self.declare_parameter("mode",        "sim")
        self.declare_parameter("goal_lat",    40.1164)
        self.declare_parameter("goal_lon",   -88.2434)
        self.declare_parameter("goal_radius",  2.0)
        self.mode        = self.get_parameter("mode").value
        self.goal_lat    = self.get_parameter("goal_lat").value
        self.goal_lon    = self.get_parameter("goal_lon").value
        self.goal_radius = self.get_parameter("goal_radius").value
        t = get_topics(self.mode)
        self.get_logger().info(f"StanleyController  mode={self.mode}  odom=" + t["odom"])
        if self.mode == "real":
            self.get_logger().info(f"  GPS goal ({self.goal_lat:.6f}, {self.goal_lon:.6f})  +/-{self.goal_radius:.1f} m")
        self.xte = self.he = 0.0
        self.last_lane_time = None
        self.obstacles = []
        self.near_goal = False
        self.last_goal_dist = float("inf")
        self.speed = self.steering = 0.0
        self._last_tick = None
        self.control_pub = self.create_publisher(AckermannDrive, t["ackermann_cmd"], 1)
        self.create_subscription(Odometry,          t["odom"],       self._odom_cb,     10)
        self.create_subscription(Float32MultiArray, t["lane_error"], self._lane_cb,     10)
        self.create_subscription(Float32MultiArray, t["obstacles"],  self._obstacle_cb, 10)
        self.create_subscription(NavSatFix,         t["navsatfix"],  self._gps_cb,      10)

    def _odom_cb(self, msg):   self._execute(msg)
    def _lane_cb(self, msg):
        if len(msg.data) >= 2:
            self.xte = float(msg.data[0]); self.he = float(msg.data[1])
            self.last_lane_time = time.time()
    def _obstacle_cb(self, msg):
        self.obstacles = []
        for i in range(0, len(msg.data), 3):
            if i + 2 < len(msg.data):
                self.obstacles.append({"x": float(msg.data[i]), "y": float(msg.data[i+1]), "area": float(msg.data[i+2])})
    def _gps_cb(self, msg):
        if msg.status.status < 0: return
        dx, dy = latlon_to_xy(msg.latitude, msg.longitude, self.goal_lat, self.goal_lon)
        self.last_goal_dist = math.hypot(dx, dy)
        self.near_goal = self.last_goal_dist < self.goal_radius

    def _extract(self, odom):
        p = odom.pose.pose.position; q = odom.pose.pose.orientation
        _, _, yaw = quaternion_to_euler([q.x, q.y, q.z, q.w])
        return p.x, p.y, odom.twist.twist.linear.x, yaw

    def _check_obstacle_ahead(self, cx, cy, yaw):
        if not self.obstacles: return False, False
        cy2 = math.cos(-yaw); sy = math.sin(-yaw)
        blocked = near = False
        for obs in self.obstacles:
            dx = obs["x"] - cx; dy = obs["y"] - cy
            fx = cy2*dx - sy*dy; fy = sy*dx + cy2*dy
            if fx <= 0 or abs(fy) > OBSTACLE_LATERAL_GATE: continue
            if   fx < STOP_DIST: blocked = True
            elif fx < SLOW_DIST: near    = True
        return blocked, near

    def _execute(self, odom):
        now = time.time()
        dt = (now - self._last_tick) if self._last_tick else 0.01
        self._last_tick = now
        cx, cy, cv, cyaw = self._extract(odom)
        lane_stale = self.last_lane_time is None or (now - self.last_lane_time) > LANE_STALE_AFTER
        obs_blocked, obs_near = self._check_obstacle_ahead(cx, cy, cyaw)
        goal_reached = self.near_goal and self.mode == "real"
        self.speed = update_speed(cv, dt, max_speed=MAX_SPEED, max_accel=MAX_ACCEL,
            stop_decel=STOP_DECEL, obs_blocked=obs_blocked, obs_near=obs_near,
            near_goal=goal_reached, lane_stale=lane_stale)
        self.steering = (0.0 if lane_stale else
            stanley(XTE_SIGN*self.xte, HE_SIGN*self.he, cv, k=STANLEY_K, max_steer=MAX_STEER))
        cmd = AckermannDrive()
        cmd.speed = float(self.speed); cmd.steering_angle = float(self.steering)
        self.control_pub.publish(cmd)
        if goal_reached and self.speed < 0.05:
            self.get_logger().info(f"GPS goal reached ({self.last_goal_dist:.2f} m) — stopping.")
            self.stop(); import os as _os; _os._exit(0)

    def stop(self):
        cmd = AckermannDrive(); cmd.speed = 0.0; cmd.steering_angle = 0.0
        self.control_pub.publish(cmd)


def main():
    rclpy.init()
    node = StanleyNode()
    try:   rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.stop(); node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == "__main__":
    main()