#!/usr/bin/env python3
"""
Hybrid Vehicle Controller: Waypoint following + Lane keeping
- Pure Pursuit for waypoint navigation and speed
- Stanley controller for lane keeping
- Blended steering = waypoint steering + stanley correction
"""

import math
import os
import sys
import time
import numpy as np

from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Float32MultiArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import quaternion_to_euler


class VehicleController:

    def __init__(self, node=None):
        if node is not None:
            self.node = node
            self.own_node = False
        else:
            import rclpy
            self.node = rclpy.create_node('vehicle_controller')
            self.own_node = True

        self.controlPub = self.node.create_publisher(
            AckermannDrive, '/ackermann_cmd', 1
        )

        # Subscribe to lane error (XTE, HE)
        self.xte = 0.0
        self.he  = 0.0
        self.last_lane_time = None

        self.node.create_subscription(
            Float32MultiArray,
            "/lane_error",
            self.lane_callback,
            10
        )

        # ===== WAYPOINT FOLLOWING =====
        self.L = 1.75
        self.current_wp_idx = 0

        self.max_speed  = 4.0
        self.min_speed  = 2.0
        self.speed_ramp = 0.1

        self.wp_accept_radius = 2.0
        self.lookahead_dist   = 3.0

        self.densify_lookahead_wps = 20
        self.curve_threshold       = math.radians(2)
        self.densify_n             = 8

        # ===== STANLEY CONTROLLER =====
        # k — cross track gain: higher = more aggressive XTE correction
        # tune: 0.3=gentle, 0.5=moderate, 0.8=aggressive
        self.stanley_k = 0.3

        # How much to blend Stanley vs Pure Pursuit
        # 0.0 = pure waypoint, 1.0 = pure stanley
        # Stanley takes over when lane data is fresh
        self.stanley_weight   = 0.2
        self.waypoint_weight  = 0.8

        # How old lane data can be before falling back to pure waypoints
        self.lane_staleness_threshold = 0.5  # seconds

        self.speed    = 0.0
        self.steering = 0.0
        self.prev_vel = 0.0

    # ── Lane error callback ───────────────────────────────────────────────────
    def lane_callback(self, msg):
        self.xte            = msg.data[0]
        self.he             = msg.data[1]
        self.last_lane_time = time.time()

    # ── Extract vehicle info ──────────────────────────────────────────────────
    def extract_vehicle_info(self, currentPose):
        pos_x = currentPose.pose.pose.position.x
        pos_y = currentPose.pose.pose.position.y
        q     = currentPose.pose.pose.orientation
        _, _, yaw = quaternion_to_euler([q.x, q.y, q.z, q.w])
        vel   = currentPose.twist.twist.linear.x
        return pos_x, pos_y, vel, yaw

    # ── Advance waypoint index ────────────────────────────────────────────────
    def advance_waypoint(self, curr_x, curr_y, waypoints):
        while self.current_wp_idx < len(waypoints) - 1:
            wp   = waypoints[self.current_wp_idx]
            dist = math.hypot(wp[0] - curr_x, wp[1] - curr_y)
            if dist < self.wp_accept_radius:
                self.current_wp_idx += 1
            else:
                break

    # ── Dynamic path densification ────────────────────────────────────────────
    def densify_path(self, curr_x, curr_y, future_waypoints):
        scan = future_waypoints[:self.densify_lookahead_wps]
        if len(scan) < 3:
            return list(future_waypoints)

        densified = []
        for i in range(len(scan)):
            densified.append(scan[i])
            if i + 2 >= len(scan):
                continue

            ax, ay = scan[i]
            bx, by = scan[i + 1]
            cx, cy = scan[i + 2]

            ab_angle    = math.atan2(by - ay, bx - ax)
            bc_angle    = math.atan2(cy - by, cx - bx)
            angle_change = abs(bc_angle - ab_angle)
            while angle_change > math.pi:
                angle_change = abs(angle_change - 2 * math.pi)

            if angle_change > self.curve_threshold:
                for k in range(1, self.densify_n + 1):
                    t = k / (self.densify_n + 1)
                    densified.append((
                        ax + t * (bx - ax),
                        ay + t * (by - ay)
                    ))

        densified.extend(future_waypoints[self.densify_lookahead_wps:])
        return densified

    # ── Lookahead point ───────────────────────────────────────────────────────
    def get_lookahead_point(self, curr_x, curr_y, future_waypoints, lookahead_dist):
        accumulated  = 0.0
        prev_x, prev_y = curr_x, curr_y
        for wp in future_waypoints:
            dx  = wp[0] - prev_x
            dy  = wp[1] - prev_y
            seg = math.hypot(dx, dy)
            if accumulated + seg >= lookahead_dist:
                remaining = lookahead_dist - accumulated
                ratio = remaining / seg if seg > 0.001 else 0.0
                return prev_x + ratio * dx, prev_y + ratio * dy
            accumulated += seg
            prev_x, prev_y = wp[0], wp[1]
        return future_waypoints[-1][0], future_waypoints[-1][1]

    # ── Longitudinal controller ───────────────────────────────────────────────
    def longitudinal_controller(self, curr_x, curr_y, curr_vel, curr_yaw,
                                future_waypoints):
        if len(future_waypoints) < 2:
            return self.min_speed

        lx, ly = self.get_lookahead_point(
            curr_x, curr_y, future_waypoints, self.lookahead_dist * 1.5)

        target_angle = math.atan2(ly - curr_y, lx - curr_x)
        diff_angle   = target_angle - curr_yaw
        while diff_angle >  math.pi: diff_angle -= 2 * math.pi
        while diff_angle < -math.pi: diff_angle += 2 * math.pi

        if abs(diff_angle) < math.pi / 8:
            return min(curr_vel + self.speed_ramp, self.max_speed)
        else:
            return max(self.min_speed, curr_vel - self.speed_ramp)

    # ── Pure Pursuit lateral controller ──────────────────────────────────────
    def pure_pursuit_lateral_controller(self, curr_x, curr_y, curr_yaw,
                                        future_waypoints):
        if len(future_waypoints) < 2:
            return 0.0

        lx, ly = self.get_lookahead_point(
            curr_x, curr_y, future_waypoints, self.lookahead_dist)

        diff_x       = lx - curr_x
        diff_y       = ly - curr_y
        target_angle = math.atan2(diff_y, diff_x)
        diff_angle   = target_angle - curr_yaw
        while diff_angle >  math.pi: diff_angle -= 2 * math.pi
        while diff_angle < -math.pi: diff_angle += 2 * math.pi

        ld = math.hypot(diff_x, diff_y)
        if ld < 0.001:
            return 0.0

        steering  = math.atan((2 * self.L * math.sin(diff_angle)) / ld)
        max_steer = 0.61
        return max(-max_steer, min(max_steer, steering))

    # ── Stanley lane keeping controller ──────────────────────────────────────
    def stanley_controller(self, speed):
        """
        Stanley controller for lane keeping.
        steering = HE + arctan(k * XTE / speed)

        HE  — heading error (radians): angle between car heading and lane tangent
        XTE — cross track error (meters): signed lateral distance from centerline
        k   — cross track gain
        """
        speed_safe    = max(speed, 0.5)  # avoid division by zero at low speed
        xte_term      = math.atan(self.stanley_k * self.xte / speed_safe)
        steering      = self.he + xte_term
        max_steer     = 0.61
        return max(-max_steer, min(max_steer, steering))

    # ── Main execute loop ─────────────────────────────────────────────────────
    def execute(self, currentPose, waypoints):
        if currentPose is None or not waypoints:
            return

        curr_x, curr_y, curr_vel, curr_yaw = self.extract_vehicle_info(currentPose)

        self.advance_waypoint(curr_x, curr_y, waypoints)

        if self.current_wp_idx >= len(waypoints) - 1:
            self.stop()
            return

        self.future_unreached_waypoints = waypoints[self.current_wp_idx:]

        dense_path = self.densify_path(
            curr_x, curr_y, self.future_unreached_waypoints)

        # ===== Waypoint steering =====
        waypoint_steering = self.pure_pursuit_lateral_controller(
            curr_x, curr_y, curr_yaw, dense_path)

        # ===== Speed =====
        self.speed = self.longitudinal_controller(
            curr_x, curr_y, curr_vel, curr_yaw, dense_path)

        # ===== Blend Stanley + Pure Pursuit =====
        lane_fresh = (
            self.last_lane_time is not None and
            (time.time() - self.last_lane_time) < self.lane_staleness_threshold
        )

        if lane_fresh:
            stanley_steering = self.stanley_controller(curr_vel)
            self.steering = (self.waypoint_weight  * waypoint_steering +
                             self.stanley_weight   * stanley_steering)
        else:
            # Lane data stale or unavailable — fall back to pure waypoints
            self.steering = waypoint_steering

        max_steer     = 0.61
        self.steering = max(-max_steer, min(max_steer, self.steering))

        cmd                = AckermannDrive()
        cmd.speed          = float(self.speed)
        cmd.steering_angle = float(self.steering)
        self.controlPub.publish(cmd)
        self.prev_vel = curr_vel

    def stop(self):
        cmd                = AckermannDrive()
        cmd.speed          = 0.0
        cmd.steering_angle = 0.0
        self.controlPub.publish(cmd)

    def destroy(self):
        if self.own_node:
            self.node.destroy_node()