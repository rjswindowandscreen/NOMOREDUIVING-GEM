#!/usr/bin/env python3
"""
FIXED Parking Controller v3 - With Proper Waypoint Detection
- FIXED: Proper waypoint advancement checking
- FIXED: Better pure pursuit steering
- FIXED: Verbose logging to see what's happening
- FIXED: Correct odometry integration
"""

import math
import os
import sys
import time
import numpy as np
from typing import List, Tuple, Optional
from collections import deque

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry 
from std_msgs.msg import Float32MultiArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from util import quaternion_to_euler
except ImportError:
    def quaternion_to_euler(q):
        x, y, z, w = q
        roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = math.asin(2*(w*y - z*x))
        yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return roll, pitch, yaw


class RRTNode:
    """Node in RRT tree"""
    def __init__(self, x: float, y: float, theta: float = 0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.parent = None
        self.cost = 0.0

    def distance_to(self, other: 'RRTNode') -> float:
        """Euclidean distance"""
        return math.hypot(self.x - other.x, self.y - other.y)

    def __repr__(self):
        return f"Node({self.x:.2f},{self.y:.2f},θ={math.degrees(self.theta):.0f}°)"


class RRTPlanner:
    """RRT* planner"""
    
    def __init__(self, max_iterations: int = 1500, max_step: float = 0.4):
        self.max_iterations = max_iterations
        self.max_step = max_step
        self.goal_sample_rate = 0.15
        self.search_radius = 2.0
        
        self.wheelbase = 1.75
        self.vehicle_width = 0.8
        self.vehicle_length = 2.5
        self.min_turning_radius = 1.5
        
        self.obstacles = []
        self.nodes = []
    
    def set_obstacles(self, obstacles: List[dict]):
        self.obstacles = obstacles
    
    def get_obstacle_radius(self, obs: dict) -> float:
        """Calculate obstacle radius from area"""
        area = obs.get('area', 1.0)
        area=2.5
        if area <= 0:
            area = 1.0
        
        radius_from_area = math.sqrt(area / math.pi)
        safety_margin = 0.4
        collision_radius = radius_from_area + safety_margin + self.vehicle_width / 2
        
        return max(0.5, collision_radius)
    
    def is_collision_free(self, n1: RRTNode, n2: RRTNode) -> bool:
        """Check if line segment collides with obstacles"""
        if not self.obstacles:
            return True
        
        x1, y1 = n1.x, n1.y
        x2, y2 = n2.x, n2.y
        
        for obs in self.obstacles:
            ox = obs['x']
            oy = obs['y']
            obs_radius = self.get_obstacle_radius(obs)
            
            dx = x2 - x1
            dy = y2 - y1
            length_sq = dx*dx + dy*dy
            
            if length_sq < 1e-6:
                dist = math.hypot(ox - x1, oy - y1)
                if dist < obs_radius:
                    return False
                continue
            
            t = max(0, min(1, ((ox - x1)*dx + (oy - y1)*dy) / length_sq))
            closest_x = x1 + t * dx
            closest_y = y1 + t * dy
            
            dist = math.hypot(ox - closest_x, oy - closest_y)
            
            if dist < obs_radius:
                return False
        
        return True
    
    def is_feasible_heading(self, n_parent: RRTNode, n_new: RRTNode) -> bool:
        """Check if heading change is feasible"""
        if n_parent.parent is None:
            return True
        
        curr_heading = math.atan2(n_parent.y - n_parent.parent.y,
                                   n_parent.x - n_parent.parent.x)
        new_heading = math.atan2(n_new.y - n_parent.y,
                                  n_new.x - n_parent.x)
        
        heading_diff = new_heading - curr_heading
        
        while heading_diff > math.pi:
            heading_diff -= 2 * math.pi
        while heading_diff < -math.pi:
            heading_diff += 2 * math.pi
        
        step_length = n_parent.distance_to(n_new)
        max_heading_change = step_length / self.min_turning_radius
        
        return abs(heading_diff) <= max_heading_change + 0.1
    
    def plan(self, start: RRTNode, goal: RRTNode) -> Optional[List[RRTNode]]:
        """Execute RRT* planning"""
        self.nodes = [start]
        start.cost = 0.0
        
        for iteration in range(self.max_iterations):
            if np.random.random() < self.goal_sample_rate:
                n_rand = goal
            else:
                n_rand = RRTNode(
                    np.random.uniform(-50, 50),
                    np.random.uniform(-50, 50),
                    np.random.uniform(-math.pi, math.pi)
                )
            
            n_near = min(self.nodes, key=lambda n: n.distance_to(n_rand))
            
            dist = n_near.distance_to(n_rand)
            if dist < 0.01:
                continue
            
            step = min(self.max_step, dist)
            n_new = RRTNode(
                n_near.x + (step / dist) * (n_rand.x - n_near.x),
                n_near.y + (step / dist) * (n_rand.y - n_near.y),
                math.atan2(n_rand.y - n_near.y, n_rand.x - n_near.x)
            )
            
            if not self.is_collision_free(n_near, n_new):
                continue
            
            if not self.is_feasible_heading(n_near, n_new):
                continue
            
            n_new.parent = n_near
            n_new.cost = n_near.cost + n_near.distance_to(n_new)
            self.nodes.append(n_new)
            
            if n_new.distance_to(goal) < self.max_step:
                goal.parent = n_new
                goal.cost = n_new.cost + n_new.distance_to(goal)
                
                path = []
                current = goal
                while current is not None:
                    path.append(current)
                    current = current.parent
                
                return list(reversed(path))
        
        return None


class ParkingController(Node):
    """Fixed Parking Controller with Proper Waypoint Detection"""
    
    def __init__(self):
        super().__init__('parking_controller_v3')
        
        # Current pose (CRITICAL - must be updated from odom)
        self.current_pose = RRTNode(0.0, 0.0, 0.0)
        self.last_odom_time = time.time()
        
        # Planning
        self.planner = RRTPlanner(max_iterations=1500, max_step=0.35)
        self.goal_pose = None
        self.planned_path = None
        self.current_wp_idx = 0
        
        # State machine
        self.parking_state = "idle"
        
        # Control parameters
        self.max_speed = 1.5
        self.min_speed = 0.3
        self.lookahead_dist = 0.6  # SMALLER for better waypoint following
        self.wheelbase = 1.75
        self.max_steering = 0.61
        self.wp_accept_dist = 2.5  # INCREASE THIS to make waypoint detection more lenient
        
        # Obstacle handling
        self.obstacles = []
        self.obstacle_stop_dist = 1.2
        self.obstacle_slow_dist = 2.5
        
        # Stuck detection
        self.pose_history = deque(maxlen=20)
        self.stuck_threshold = 0.1
        self.stuck_time = 3.0
        self.reversed_count = 0
        self.max_reversals = 2
        
        # DEBUG: Track waypoint progress
        self.last_wp_advance_time = time.time()
        self.wp_advance_count = 0
        
        # Publishers/Subscribers
        self.control_pub = self.create_publisher(AckermannDrive, '/ackermann_cmd', 1)
        
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Float32MultiArray, '/obstacles', 
                                self.obstacle_callback, 10)
        
        self.create_timer(0.02, self.control_loop)
        
        self.get_logger().info("✓ Parking Controller V3 initialized")
        self.get_logger().info(f"  Waypoint acceptance distance: {self.wp_accept_dist}m")
        self.get_logger().info(f"  Lookahead distance: {self.lookahead_dist}m")
        self.get_logger().info(f"  Max speed: {self.max_speed} m/s")
    
    def odom_callback(self, msg: Odometry):
        """Update current pose from odometry - CRITICAL!"""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        
        try:
            _, _, yaw = quaternion_to_euler([q.x, q.y, q.z, q.w])
        except Exception as e:
            self.get_logger().warn(f"Quaternion conversion error: {e}")
            yaw = self.current_pose.theta
        
        # UPDATE POSE - This is critical!
        old_pose = self.current_pose
        self.current_pose = RRTNode(p.x, p.y, yaw)
        
        # Track pose for stuck detection
        self.pose_history.append((p.x, p.y, time.time()))
        
        # DEBUG: Print pose updates occasionally
        if time.time() % 2 < 0.05:  # Every ~2 seconds
            self.get_logger().info(
                f"📍 POSE: ({p.x:.2f}, {p.y:.2f}) θ={math.degrees(yaw):.0f}°"
            )
    
    def obstacle_callback(self, msg: Float32MultiArray):
        """Update obstacles from detection"""
        self.obstacles = []
        
        for i in range(0, len(msg.data), 3):
            if i + 2 < len(msg.data):
                self.obstacles.append({
                    'x': msg.data[i],
                    'y': msg.data[i + 1],
                    'area': msg.data[i + 2]
                })
    
    def is_stuck(self) -> bool:
        """Detect if vehicle is stuck"""
        if len(self.pose_history) < 10:
            return False
        
        x_values = [p[0] for p in self.pose_history]
        y_values = [p[1] for p in self.pose_history]
        times = [p[2] for p in self.pose_history]
        
        dist_moved = math.hypot(x_values[-1] - x_values[0], 
                                y_values[-1] - y_values[0])
        time_elapsed = times[-1] - times[0]
        
        if time_elapsed < 0.01:
            return False
        
        velocity = dist_moved / time_elapsed
        return velocity < self.stuck_threshold
    
    def check_obstacle_ahead(self) -> Tuple[bool, Optional[float]]:
        """Check for obstacles blocking path"""
        if not self.obstacles:
            return False, None
        
        closest_dist = float('inf')
        
        for obs in self.obstacles:
            forward_dist = -obs['y']
            lateral_dist = obs['x']
            
            if forward_dist > 0 and abs(lateral_dist) < 2.0:
                if forward_dist < closest_dist:
                    closest_dist = forward_dist
        
        if closest_dist < self.obstacle_stop_dist:
            return True, closest_dist
        
        return False, closest_dist
    
    def pure_pursuit_steering(self) -> float:
        """Calculate steering using pure pursuit"""
        if not self.planned_path or self.current_wp_idx >= len(self.planned_path):
            return 0.0
        
        curr_x = self.current_pose.x
        curr_y = self.current_pose.y
        curr_yaw = self.current_pose.theta
        
        # Find lookahead point
        accumulated_dist = 0.0
        lookahead_x = curr_x
        lookahead_y = curr_y
        
        for i in range(self.current_wp_idx, len(self.planned_path)):
            wp = self.planned_path[i]
            dx = wp.x - lookahead_x
            dy = wp.y - lookahead_y
            dist = math.hypot(dx, dy)
            
            if accumulated_dist + dist >= self.lookahead_dist:
                remaining = self.lookahead_dist - accumulated_dist
                if dist > 0.001:
                    ratio = remaining / dist
                    lookahead_x = lookahead_x + ratio * dx
                    lookahead_y = lookahead_y + ratio * dy
                break
            
            accumulated_dist += dist
            lookahead_x = wp.x
            lookahead_y = wp.y
        
        # Pure pursuit calculation
        dx = lookahead_x - curr_x
        dy = lookahead_y - curr_y
        target_angle = math.atan2(dy, dx)
        
        angle_err = target_angle - curr_yaw
        
        while angle_err > math.pi:
            angle_err -= 2 * math.pi
        while angle_err < -math.pi:
            angle_err += 2 * math.pi
        
        ld = math.hypot(dx, dy)
        
        if ld < 0.01:
            return 0.0
        
        steering = math.atan((2 * self.wheelbase * math.sin(angle_err)) / ld)
        steering = max(-self.max_steering, min(self.max_steering, steering))
        
        return steering
    
    def advance_waypoints(self):
        """FIXED: Advance to next waypoint with proper detection and logging"""
        if not self.planned_path:
            return
        
        # Check current waypoint distance
        if self.current_wp_idx >= len(self.planned_path):
            return
        
        wp = self.planned_path[self.current_wp_idx]
        dist_to_wp = math.hypot(
            wp.x - self.current_pose.x,
            wp.y - self.current_pose.y
        )
        
        # Log waypoint distance every second
        if time.time() - self.last_wp_advance_time > 1.0:
            self.get_logger().info(
                f"📍 Waypoint {self.current_wp_idx}/{len(self.planned_path)}: "
                f"({wp.x:.2f}, {wp.y:.2f}) - Distance: {dist_to_wp:.2f}m "
                f"(target: {self.wp_accept_dist}m)"
            )
            self.last_wp_advance_time = time.time()
        
        # MAIN LOOP: Keep advancing while close to waypoints
        while self.current_wp_idx < len(self.planned_path) - 1:
            wp = self.planned_path[self.current_wp_idx]
            dist_to_wp = math.hypot(
                wp.x - self.current_pose.x,
                wp.y - self.current_pose.y
            )
            
            if dist_to_wp < self.wp_accept_dist:
                # Reached waypoint!
                self.current_wp_idx += 1
                self.wp_advance_count += 1
                self.get_logger().info(
                    f"✓ Waypoint {self.current_wp_idx-1} REACHED! "
                    f"Advancing to WP{self.current_wp_idx} "
                    f"(Total advanced: {self.wp_advance_count})"
                )
            else:
                # Not close enough yet
                break
    
    def start_parking(self, goal_x: float, goal_y: float, 
                     goal_yaw: float) -> bool:
        """Start parking maneuver"""
        self.get_logger().info(
            f"🎯 Starting parking to ({goal_x:.2f}, {goal_y:.2f}) "
            f"@ {math.degrees(goal_yaw):.1f}°"
        )
        
        self.goal_pose = RRTNode(goal_x, goal_y, goal_yaw)
        
        self.planner.set_obstacles(self.obstacles)
        
        self.get_logger().info("🤖 Planning path...")
        self.planned_path = self.planner.plan(self.current_pose, self.goal_pose)
        
        if self.planned_path is None:
            self.get_logger().error("❌ Planning FAILED!")
            self.parking_state = "idle"
            return False
        
        self.get_logger().info(
            f"✓ Path found: {len(self.planned_path)} waypoints"
        )
        
        # Print first few waypoints
        for i, wp in enumerate(self.planned_path[:5]):
            self.get_logger().info(
                f"  WP{i}: ({wp.x:.2f}, {wp.y:.2f}) θ={math.degrees(wp.theta):.0f}°"
            )
        
        self.current_wp_idx = 0
        self.parking_state = "executing"
        self.reversed_count = 0
        self.pose_history.clear()
        self.wp_advance_count = 0
        
        return True
    
    def control_loop(self):
        """Main control loop (50 Hz)"""
        
        if self.parking_state == "idle":
            self.stop()
            return
        
        elif self.parking_state == "finished":
            self.stop()
            return
        
        elif self.parking_state != "executing" or not self.planned_path:
            self.stop()
            return
        
        # CRITICAL: Advance waypoints regularly
        self.advance_waypoints()
        
        # Check if goal reached
        dist_to_goal = self.current_pose.distance_to(self.goal_pose)
        if dist_to_goal < 1:
            self.get_logger().info("🎉 GOAL REACHED!")
            self.parking_state = "finished"
            self.stop()
            return
        
        # Get steering command
        steering = self.pure_pursuit_steering()
        
        # Default speed
        speed = self.max_speed
        
        # Check obstacles
        obs_blocking, obs_dist = self.check_obstacle_ahead()
        
        if obs_blocking:
            self.get_logger().warn(f"⛔ Obstacle blocking at {obs_dist:.2f}m - STOP")
            speed = 0.0
        elif obs_dist is not None and obs_dist < self.obstacle_slow_dist:
            self.get_logger().info(f"⚠️  Obstacle at {obs_dist:.2f}m - slow down")
            speed = self.min_speed
        
        # Check if stuck
        # if self.is_stuck():
        #     if self.reversed_count < self.max_reversals:
        #         self.get_logger().warn("🔄 Vehicle stuck - reversing...")
        #         speed = -0.5
        #         steering = 0.0
        #         self.reversed_count += 1
        #     else:
        #         self.get_logger().error("❌ Too many reversals, aborting")
        #         self.parking_state = "idle"
        #         self.stop()
        #         return
        # else:
        self.reversed_count = 0
        
        # Publish command
        cmd = AckermannDrive()
        cmd.speed = float(speed)
        cmd.steering_angle = float(steering)
        self.control_pub.publish(cmd)
    
    def stop(self):
        """Stop vehicle"""
        cmd = AckermannDrive()
        cmd.speed = 0.0
        cmd.steering_angle = 0.0
        self.control_pub.publish(cmd)


def main():
    rclpy.init()
    controller = ParkingController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info("Interrupted")
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


