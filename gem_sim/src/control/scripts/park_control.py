#!/usr/bin/env python3
"""
park_control.py — RRT-based park-to-drive controller.

Drives from the car's current position to a specified goal (the lane entry
point) using RRT path planning and Pure Pursuit execution.

Publishes /parking_done (std_msgs/Bool True) once the goal is reached,
so run_all.py knows to transition to lane following.

ROS2 parameters (set by run_all.py via --ros-args):
    goal_x       float  target x in local frame       default -20.0
    goal_y       float  target y in local frame        default  0.0
    goal_yaw     float  target heading (rad)           default  0.0
    drive_speed  float  m/s for the maneuver           default  0.8
    odom_topic   str    odometry topic to subscribe to default 'odom'
"""

import math
import os
import sys
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32MultiArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import quaternion_to_euler

# topics.py: control/scripts/ -> control/ -> src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from topics import get_topics


# ─────────────────────────────────────────────────────────────────────────────
# RRT data structures  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

class RRTNode:
    def __init__(self, x: float, y: float, theta: float = 0.0):
        self.x      = x
        self.y      = y
        self.theta  = theta
        self.parent = None

    def distance_to(self, other: 'RRTNode') -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


class RRTPlanner:
    def __init__(self, max_iterations: int = 2000, max_step: float = 0.5):
        self.max_iterations  = max_iterations
        self.max_step        = max_step
        self.goal_sample_rate = 0.15
        self.obstacles       = []
        self.nodes           = []

        self.wheelbase          = 1.75
        self.planner_max_steer  = 0.6
        self.min_turning_radius = self.wheelbase / math.tan(self.planner_max_steer)

    def set_obstacles(self, obstacles: List[dict]):
        self.obstacles = obstacles

    def is_feasible_turn(self, n_parent: RRTNode, n_new: RRTNode) -> bool:
        if n_parent.parent is None:
            return True
        angle_new = math.atan2(n_new.y    - n_parent.y,
                               n_new.x    - n_parent.x)
        angle_old = math.atan2(n_parent.y - n_parent.parent.y,
                               n_parent.x - n_parent.parent.x)
        diff = abs((angle_new - angle_old + math.pi) % (2 * math.pi) - math.pi)
        return diff <= self.max_step / self.min_turning_radius

    def is_collision_free(self, n1: RRTNode, n2: RRTNode) -> bool:
        if not self.obstacles:
            return True
        for obs in self.obstacles:
            ox, oy     = obs['x'], obs['y']
            obs_radius = math.sqrt(obs.get('area', 1.0) / math.pi) + 0.6
            dx, dy     = n2.x - n1.x, n2.y - n1.y
            d_sq       = dx * dx + dy * dy
            t = (max(0, min(1, ((ox - n1.x) * dx + (oy - n1.y) * dy) / d_sq))
                 if d_sq > 0 else 0)
            dist = math.hypot(ox - (n1.x + t * dx), oy - (n1.y + t * dy))
            if dist < obs_radius:
                return False
        return True

    def plan(self, start: RRTNode, goal: RRTNode) -> Optional[List[RRTNode]]:
        self.nodes = [start]
        for _ in range(self.max_iterations):
            n_rand = (goal if np.random.random() < self.goal_sample_rate
                      else RRTNode(np.random.uniform(-40, 40),
                                   np.random.uniform(-40, 40)))
            n_near = min(self.nodes, key=lambda n: n.distance_to(n_rand))
            dist   = n_near.distance_to(n_rand)
            if dist < 1e-6:
                continue
            step  = min(self.max_step, dist)
            n_new = RRTNode(n_near.x + (step / dist) * (n_rand.x - n_near.x),
                            n_near.y + (step / dist) * (n_rand.y - n_near.y))
            if (self.is_collision_free(n_near, n_new) and
                    self.is_feasible_turn(n_near, n_new)):
                n_new.parent = n_near
                self.nodes.append(n_new)
                if n_new.distance_to(goal) < self.max_step:
                    goal.parent = n_new
                    path, curr  = [], goal
                    while curr:
                        path.append(curr)
                        curr = curr.parent
                    return list(reversed(path))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ParkingController node
# ─────────────────────────────────────────────────────────────────────────────

class ParkingController(Node):

    def __init__(self):
        super().__init__('parking_controller')

        # ── ROS2 parameters ────────────────────────────────────────────────
        self.declare_parameter('mode',        'sim')
        self.declare_parameter('goal_x',      -20.0)
        self.declare_parameter('goal_y',        0.0)
        self.declare_parameter('goal_yaw',      0.0)
        self.declare_parameter('drive_speed',   0.8)

        mode         = self.get_parameter('mode').value
        goal_x       = self.get_parameter('goal_x').value
        goal_y       = self.get_parameter('goal_y').value
        goal_yaw     = self.get_parameter('goal_yaw').value
        self.speed   = self.get_parameter('drive_speed').value

        # odom topic from topics.py based on mode
        t            = get_topics(mode)
        odom_topic   = t['odom']

        self.get_logger().info(
            f'ParkingController: goal=({goal_x:.2f}, {goal_y:.2f}, '
            f'{math.degrees(goal_yaw):.1f}°)  speed={self.speed:.1f} m/s  '
            f'odom={odom_topic}'
        )

        # ── Goal ───────────────────────────────────────────────────────────
        self.goal_pose = RRTNode(goal_x, goal_y, goal_yaw)

        # ── Controller state ───────────────────────────────────────────────
        self.current_pose   = RRTNode(0.0, 0.0, 0.0)
        self.planned_path   = None
        self.current_path_idx = 0
        self.parking_state  = 'idle'       # idle → planning → executing → finished
        self.planner        = RRTPlanner()
        self.obstacles      = []

        # Pure Pursuit params
        self.max_steer     = 0.61
        self.lookahead_dist = 1.8
        self.L              = 1.75

        # ── Publishers ─────────────────────────────────────────────────────
        self.control_pub = self.create_publisher(AckermannDrive, '/ackermann_cmd',  1)
        self.done_pub    = self.create_publisher(Bool,           '/parking_done',   1)

        # ── Subscribers ────────────────────────────────────────────────────
        self.create_subscription(Odometry,         odom_topic,   self.odom_cb,     10)
        self.create_subscription(Float32MultiArray, '/obstacles', self.obstacle_cb, 10)

        # ── Control timer: 20 Hz ───────────────────────────────────────────
        self.create_timer(0.05, self.control_loop)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def odom_cb(self, msg):
        p               = msg.pose.pose.position
        q               = msg.pose.pose.orientation
        _, _, yaw       = quaternion_to_euler([q.x, q.y, q.z, q.w])
        self.current_pose.x     = p.x
        self.current_pose.y     = p.y
        self.current_pose.theta = yaw

        # Auto-start planning on first odom — no external trigger needed
        if self.parking_state == 'idle':
            self.parking_state = 'planning'
            self.get_logger().info(
                f'First odom received at ({p.x:.2f}, {p.y:.2f}) — '
                f'planning RRT path...'
            )
            self._run_planner()

    def obstacle_cb(self, msg):
        self.obstacles = []
        for i in range(0, len(msg.data), 3):
            self.obstacles.append({
                'x':    msg.data[i],
                'y':    msg.data[i + 1],
                'area': msg.data[i + 2],
            })

    # ── Planning ──────────────────────────────────────────────────────────────

    def _run_planner(self):
        start = RRTNode(self.current_pose.x,
                        self.current_pose.y,
                        self.current_pose.theta)
        self.planner.set_obstacles(self.obstacles)
        path = self.planner.plan(start, self.goal_pose)

        if path:
            self.planned_path     = path
            self.current_path_idx = 0
            self.parking_state    = 'executing'
            self.get_logger().info(
                f'Path found — {len(path)} waypoints. Executing...'
            )
        else:
            self.parking_state = 'idle'
            self.get_logger().error(
                'RRT failed to find a path. Retrying on next odom...'
            )

    # ── Control loop ──────────────────────────────────────────────────────────

    def control_loop(self):
        if self.parking_state != 'executing' or not self.planned_path:
            return

        cx = self.current_pose.x
        cy = self.current_pose.y
        cyaw = self.current_pose.theta

        # ── Goal check ────────────────────────────────────────────────────
        dist_to_goal = math.hypot(
            self.planned_path[-1].x - cx,
            self.planned_path[-1].y - cy,
        )
        if dist_to_goal < 0.6:
            self.get_logger().info('Goal reached — parking complete.')
            self.stop()
            self.parking_state = 'finished'
            # Publish completion signal for run_all.py
            self.done_pub.publish(Bool(data=True))
            return

        # ── Pure Pursuit lookahead ─────────────────────────────────────────
        target = self.planned_path[self.current_path_idx]
        for i in range(self.current_path_idx, len(self.planned_path)):
            d = math.hypot(
                self.planned_path[i].x - cx,
                self.planned_path[i].y - cy,
            )
            if d > self.lookahead_dist:
                target                = self.planned_path[i]
                self.current_path_idx = i
                break

        # ── Steering ──────────────────────────────────────────────────────
        alpha    = (math.atan2(target.y - cy, target.x - cx)
                    - cyaw + math.pi) % (2 * math.pi) - math.pi
        steering = math.atan2(
            2.0 * self.L * math.sin(alpha), self.lookahead_dist
        )
        steering = max(-self.max_steer, min(self.max_steer, steering))

        cmd                = AckermannDrive()
        cmd.speed          = float(self.speed)
        cmd.steering_angle = float(steering)
        self.control_pub.publish(cmd)

    # ── Stop ──────────────────────────────────────────────────────────────────

    def stop(self):
        cmd                = AckermannDrive()
        cmd.speed          = 0.0
        cmd.steering_angle = 0.0
        self.control_pub.publish(cmd)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = ParkingController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()