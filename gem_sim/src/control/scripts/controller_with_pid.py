#!/usr/bin/env python3
"""
Hybrid Vehicle Controller: Waypoint following + Lane keeping
- Pure Pursuit for waypoint navigation
- XTE/HE PID for lane correction
- Blended steering = waypoint steering + lane correction
"""

import math
import os
import sys
import time

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
        self.he = 0.0
        self.prev_xte = 0.0
        self.prev_he = 0.0
        self.xte_integral = 0.0
        self.he_integral = 0.0
        self.last_lane_time = None
       
        self.node.create_subscription(
            Float32MultiArray,
            "/lane_error",
            self.lane_callback,
            10
        )

        # ===== WAYPOINT FOLLOWING =====
        self.L = 1.75       # wheelbase (meters)
        self.current_wp_idx = 0

        self.max_speed  = 4
        self.min_speed  = 2
        self.speed_ramp = 0.1

        # Waypoint advancement radius
        self.wp_accept_radius = 2.0   # meters

        # Lookahead distance in meters
        self.lookahead_dist = 3.0

        # Curve densification
        self.densify_lookahead_wps = 20
        self.curve_threshold = math.radians(2)
        self.densify_n = 8

        # ===== LANE KEEPING (PID) =====
        # PID gains for XTE (position error)
        self.k_p_xte = 0.3      # Proportional
        self.k_i_xte = 0.05     # Integral
        self.k_d_xte = 0.15     # Derivative

        # PID gains for HE (heading error)
        self.k_p_he = 0.8       # Proportional
        self.k_i_he = 0.02      # Integral
        self.k_d_he = 0.2       # Derivative

        # Integral limits (anti-windup)
        self.i_limit = 0.3

        # Lane correction weight: how much to blend lane correction into steering
        # 0.0 = pure waypoint following, 1.0 = pure lane keeping
        self.lane_correction_weight = 0.4

        self.speed    = 0.0
        self.steering = 0.0
        self.prev_vel = 0.0

    # ── Lane error callback ───────────────────────────────────────────────────
    def lane_callback(self, msg):
        self.xte = msg.data[0]
        self.he = msg.data[1]

    # ── Extract vehicle info ──────────────────────────────────────────────────
    def extract_vehicle_info(self, currentPose):
        pos_x = currentPose.pose.pose.position.x
        pos_y = currentPose.pose.pose.position.y
        q = currentPose.pose.pose.orientation
        _, _, yaw = quaternion_to_euler([q.x, q.y, q.z, q.w])
        vel = currentPose.twist.twist.linear.x
        return pos_x, pos_y, vel, yaw

    # ── Advance waypoint index ────────────────────────────────────────────────
    def advance_waypoint(self, curr_x, curr_y, waypoints):
        """
        Move to the next waypoint only when the car is within
        wp_accept_radius of the current one. Never goes backward.
        """
        while self.current_wp_idx < len(waypoints) - 1:
            wp = waypoints[self.current_wp_idx]
            dist = math.hypot(wp[0] - curr_x, wp[1] - curr_y)
            if dist < self.wp_accept_radius:
                self.current_wp_idx += 1
            else:
                break

    # ── Dynamic path densification ────────────────────────────────────────────
    def densify_path(self, curr_x, curr_y, future_waypoints):
        """
        Insert extra interpolated points between waypoints where
        the path curves sharply.
        """
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

            ab_angle = math.atan2(by - ay, bx - ax)
            bc_angle = math.atan2(cy - by, cx - bx)

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

    # ── Lookahead point ──────────────────────────────────────────────────────
    def get_lookahead_point(self, curr_x, curr_y, future_waypoints, lookahead_dist):
        accumulated = 0.0
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
            curr_x, curr_y, future_waypoints, self.lookahead_dist * 1.5
        )

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
            curr_x, curr_y, future_waypoints, self.lookahead_dist
        )

        diff_x = lx - curr_x
        diff_y = ly - curr_y

        target_angle = math.atan2(diff_y, diff_x)
        diff_angle   = target_angle - curr_yaw
        while diff_angle >  math.pi: diff_angle -= 2 * math.pi
        while diff_angle < -math.pi: diff_angle += 2 * math.pi

        ld = math.hypot(diff_x, diff_y)
        if ld < 0.001:
            return 0.0

        steering = math.atan((2 * self.L * math.sin(diff_angle)) / ld)

        # Clamp to physical steering limits
        max_steer = 0.61
        return max(-max_steer, min(max_steer, steering))

    # ── Lane-keeping PID controller ──────────────────────────────────────────
    def lane_keeping_controller(self):
        """
        Compute steering correction to stay in lane using XTE and HE.
        Returns correction steering angle (positive/negative adjustment).
        """
        current_time = time.time()
        if self.last_lane_time is None:
            self.last_lane_time = current_time
            self.prev_xte = self.xte
            self.prev_he = self.he
       
        dt = current_time - self.last_lane_time
        if dt == 0:
            dt = 0.01
       
        # ===== XTE (Position Error) PID =====
        xte_error = self.xte
       
        # Proportional
        xte_p = -self.k_p_xte * xte_error
       
        # Integral (with anti-windup)
        self.xte_integral += xte_error * dt
        self.xte_integral = max(-self.i_limit, min(self.i_limit, self.xte_integral))
        xte_i = -self.k_i_xte * self.xte_integral
       
        # Derivative
        xte_derivative = (xte_error - self.prev_xte) / dt
        xte_d = -self.k_d_xte * xte_derivative
       
        xte_correction = xte_p + xte_i + xte_d
       
        # ===== HE (Heading Error) PID =====
        he_error = self.he
       
        # Proportional
        he_p = self.k_p_he * he_error
       
        # Integral (with anti-windup)
        self.he_integral += he_error * dt
        self.he_integral = max(-self.i_limit, min(self.i_limit, self.he_integral))
        he_i = self.k_i_he * self.he_integral
       
        # Derivative
        he_derivative = (he_error - self.prev_he) / dt
        he_d = self.k_d_he * he_derivative
       
        he_correction = he_p + he_i + he_d
       
        # Combined lane correction
        lane_correction = xte_correction + he_correction
       
        # Update previous values
        self.prev_xte = xte_error
        self.prev_he = he_error
        self.last_lane_time = current_time
       
        return lane_correction

    # ── Main execute loop ─────────────────────────────────────────────────────
    def execute(self, currentPose, waypoints):
        if currentPose is None or not waypoints:
            return

        curr_x, curr_y, curr_vel, curr_yaw = self.extract_vehicle_info(currentPose)

        # Advance waypoint index
        prev_wp_idx = self.current_wp_idx
        self.advance_waypoint(curr_x, curr_y, waypoints)
       
        # Reset integral terms when waypoint changes (avoid buildup)
        if self.current_wp_idx != prev_wp_idx:
            self.xte_integral = 0.0
            self.he_integral = 0.0

        if self.current_wp_idx >= len(waypoints) - 1:
            self.stop()
            return

        self.future_unreached_waypoints = waypoints[self.current_wp_idx:]

        # Densify path on curves
        dense_path = self.densify_path(
            curr_x, curr_y, self.future_unreached_waypoints
        )

        # ===== Get steering from both controllers =====
        waypoint_steering = self.pure_pursuit_lateral_controller(
            curr_x, curr_y, curr_yaw, dense_path
        )
       
        # Only use lane correction if lane errors are recent and valid
        lane_correction = 0.0
        if self.last_lane_time is not None:
            # Check if lane data is fresh (within 1 second)
            time_since_lane = time.time() - self.last_lane_time
            if time_since_lane < 1.0:
                lane_correction = self.lane_keeping_controller()
            else:
                # Lane data is stale, rely on waypoint following
                pass
       
        # ===== Blend the two steering commands =====
        # Waypoint steering is primary, lane correction is secondary
        self.steering = waypoint_steering + self.lane_correction_weight * lane_correction
       
        # Clamp final steering to valid range
        max_steer = 0.61
        self.steering = max(-max_steer, min(max_steer, self.steering))

        # Speed logic unchanged
        self.speed = self.longitudinal_controller(
            curr_x, curr_y, curr_vel, curr_yaw, dense_path
        )

        # Publish command
        cmd = AckermannDrive()
        cmd.speed          = float(self.speed)
        cmd.steering_angle = float(self.steering)
        self.controlPub.publish(cmd)
        self.prev_vel = curr_vel

    def stop(self):
        cmd = AckermannDrive()
        cmd.speed = 0.0
        cmd.steering_angle = 0.0
        self.controlPub.publish(cmd)

    def destroy(self):
        if self.own_node:
            self.node.destroy_node()