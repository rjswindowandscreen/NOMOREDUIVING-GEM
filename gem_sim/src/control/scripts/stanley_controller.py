#!/usr/bin/env python3
"""
stanley_controller.py — Pure Stanley lateral controller with slow-speed
longitudinal control, GPS goal stop, and obstacle stop.

Subs:
  /odometry/global   nav_msgs/Odometry            (pose + speed, from localization)
  /lane_error        std_msgs/Float32MultiArray   ([XTE, HE] from perception)
  /obstacles         std_msgs/Float32MultiArray   ([x, y, area, ...] world frame)
  /navsatfix         sensor_msgs/NavSatFix        (for distance-to-goal check)

Pubs:
  /ackermann_cmd     ackermann_msgs/AckermannDrive

Sign convention (standard Stanley):
  xte > 0 → lane is LEFT of vehicle → steer LEFT (+)
  he  > 0 → vehicle heading is RIGHT of lane tangent → steer LEFT (+)
If the car steers AWAY from the lane on first sim run, flip XTE_SIGN below.
"""

import math
import os
import sys
import time

from ackermann_msgs.msg import AckermannDrive
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32MultiArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stanley_math import stanley, update_speed
from util import quaternion_to_euler

# latlon_to_xy lives in the localization package's scripts dir
_LOC_SCRIPTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'localization', 'scripts'
)
sys.path.insert(0, _LOC_SCRIPTS)
from gps_utils import latlon_to_xy


# ── CONFIGURATION ─────────────────────────────────────────────────────────────

# GPS goal — set these to your actual destination
GOAL_LAT    = 40.1164
GOAL_LON    = -88.2434
GOAL_RADIUS = 2.0          # m

# Speed
MAX_SPEED  = 0.8           # m/s — walking pace
MAX_ACCEL  = 0.2           # m/s²
STOP_DECEL = 0.6           # m/s²

# Stanley
STANLEY_K        = 0.3
MAX_STEER        = 0.61    # rad at road wheel
LANE_STALE_AFTER = 0.5     # s
XTE_SIGN         = +1      # flip to -1 if car steers away from lane on first test
HE_SIGN          = +1

# Obstacle (vehicle-frame thresholds)
STOP_DIST = 5.0
SLOW_DIST = 9.0
OBSTACLE_LATERAL_GATE = 3.0   # m — ignore obstacles further sideways than this


class StanleyController:
    """Pure Stanley + slow speed. Replaces controller.py / controller_with_pid.py."""

    def __init__(self, node):
        self.node = node

        self.controlPub = self.node.create_publisher(
            AckermannDrive, '/ackermann_cmd', 1)

        # Perception state
        self.xte = 0.0
        self.he  = 0.0
        self.last_lane_time = None

        self.obstacles = []   # world-frame: list of {'x','y','area'}

        # GPS goal state
        self.near_goal = False
        self.last_goal_dist = float('inf')

        # Subscribers
        self.node.create_subscription(
            Float32MultiArray, '/lane_error', self._lane_cb, 10)
        self.node.create_subscription(
            Float32MultiArray, '/obstacles', self._obstacle_cb, 10)
        self.node.create_subscription(
            NavSatFix, '/navsatfix', self._gps_cb, 10)

        # Output state
        self.speed    = 0.0
        self.steering = 0.0
        self._last_tick = None

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _lane_cb(self, msg):
        if len(msg.data) >= 2:
            self.xte = float(msg.data[0])
            self.he  = float(msg.data[1])
            self.last_lane_time = time.time()

    def _obstacle_cb(self, msg):
        self.obstacles = []
        for i in range(0, len(msg.data), 3):
            if i + 2 < len(msg.data):
                self.obstacles.append({
                    'x':    float(msg.data[i]),
                    'y':    float(msg.data[i + 1]),
                    'area': float(msg.data[i + 2]),
                })

    def _gps_cb(self, msg):
        if msg.status.status < 0:
            return
        # latlon_to_xy(lat, lon, lat0, lon0) → (x, y) of (lat,lon) relative to
        # (lat0, lon0). Using GOAL as the origin gives us distance to goal directly.
        dx, dy = latlon_to_xy(msg.latitude, msg.longitude, GOAL_LAT, GOAL_LON)
        self.last_goal_dist = math.hypot(dx, dy)
        self.near_goal = self.last_goal_dist < GOAL_RADIUS

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _extract(self, odom):
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        _, _, yaw = quaternion_to_euler([q.x, q.y, q.z, q.w])
        v = odom.twist.twist.linear.x
        return p.x, p.y, v, yaw

    def _check_obstacle_ahead(self, car_x, car_y, yaw):
        """Project world-frame obstacles into vehicle frame; flag blocked / near."""
        if not self.obstacles:
            return False, False
        cy = math.cos(-yaw)
        sy = math.sin(-yaw)
        blocked = near = False
        for obs in self.obstacles:
            dx = obs['x'] - car_x
            dy = obs['y'] - car_y
            # Rotate world delta into vehicle frame (forward = +x)
            fx = cy * dx - sy * dy
            fy = sy * dx + cy * dy
            if fx <= 0:                              # behind us
                continue
            if abs(fy) > OBSTACLE_LATERAL_GATE:      # too far to the side
                continue
            if   fx < STOP_DIST: blocked = True
            elif fx < SLOW_DIST: near    = True
        return blocked, near

    # ── Main control step ────────────────────────────────────────────────────

    def execute(self, odom):
        if odom is None:
            return

        now = time.time()
        dt = (now - self._last_tick) if self._last_tick is not None else 0.01
        self._last_tick = now

        cx, cy, cv, cyaw = self._extract(odom)

        # Stop conditions
        lane_stale = (
            self.last_lane_time is None
            or (now - self.last_lane_time) > LANE_STALE_AFTER
        )
        obs_blocked, obs_near = self._check_obstacle_ahead(cx, cy, cyaw)

        # Longitudinal
        self.speed = update_speed(
            cv, dt,
            max_speed=MAX_SPEED,
            max_accel=MAX_ACCEL,
            stop_decel=STOP_DECEL,
            obs_blocked=obs_blocked,
            obs_near=obs_near,
            near_goal=self.near_goal,
            lane_stale=lane_stale,
        )

        # Lateral
        if lane_stale:
            self.steering = 0.0
        else:
            self.steering = stanley(
                XTE_SIGN * self.xte,
                HE_SIGN  * self.he,
                cv,
                k=STANLEY_K,
                max_steer=MAX_STEER,
            )

        # Publish
        cmd = AckermannDrive()
        cmd.speed          = float(self.speed)
        cmd.steering_angle = float(self.steering)
        self.controlPub.publish(cmd)

    def stop(self):
        cmd = AckermannDrive()
        cmd.speed          = 0.0
        cmd.steering_angle = 0.0
        self.controlPub.publish(cmd)

    def destroy(self):
        pass
