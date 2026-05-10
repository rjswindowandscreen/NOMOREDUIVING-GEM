#!/usr/bin/env python3
import math
import os
import sys
import numpy as np
from typing import List, Tuple, Optional
import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry 
from std_msgs.msg import Float32MultiArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import quaternion_to_euler

class RRTNode:
    def __init__(self, x: float, y: float, theta: float = 0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.parent = None

    def distance_to(self, other: 'RRTNode') -> float:
        """Euclidean distance calculation used by the planner"""
        return math.hypot(self.x - other.x, self.y - other.y)

class RRTPlanner:
    def __init__(self, max_iterations: int = 2000, max_step: float = 0.5):
        self.max_iterations = max_iterations
        self.max_step = max_step
        self.goal_sample_rate = 0.15
        self.obstacles = []
        self.nodes = []
        
        # --- PLANNER-ONLY STEERING LIMIT ---
        self.wheelbase = 1.75
        self.planner_max_steer = 0.6  # The planner designs for gentle turns
        self.min_turning_radius = self.wheelbase / math.tan(self.planner_max_steer)
        
    def set_obstacles(self, obstacles: List[dict]):
        self.obstacles = obstacles
    
    def is_feasible_turn(self, n_parent: RRTNode, n_new: RRTNode) -> bool:
        """Restricts the planner from making sharp 'snaking' turns"""
        if n_parent.parent is None: return True
        
        angle_new = math.atan2(n_new.y - n_parent.y, n_new.x - n_parent.x)
        angle_old = math.atan2(n_parent.y - n_parent.parent.y, n_parent.x - n_parent.parent.x)
        
        diff = abs((angle_new - angle_old + math.pi) % (2 * math.pi) - math.pi)
        
        # geometric limit: delta_theta <= step / radius
        max_allowed_diff = self.max_step / self.min_turning_radius
        return diff <= max_allowed_diff

    def is_collision_free(self, n1: RRTNode, n2: RRTNode) -> bool:
        if not self.obstacles: return True
        for obs in self.obstacles:
            ox, oy = obs['x'], obs['y']
            obs_radius = math.sqrt(obs.get('area', 1.0) / math.pi) + 0.6
            print("RAD "+str(obs_radius))
            obs_radius = 20.0
            dx, dy = n2.x - n1.x, n2.y - n1.y
            d_sq = dx*dx + dy*dy
            t = max(0, min(1, ((ox - n1.x) * dx + (oy - n1.y) * dy) / d_sq)) if d_sq > 0 else 0
            dist = math.hypot(ox - (n1.x + t*dx), oy - (n1.y + t*dy))
            if dist < obs_radius: return False
        return True
    
    def plan(self, start: RRTNode, goal: RRTNode) -> Optional[List[RRTNode]]:
        self.nodes = [start]
        for _ in range(self.max_iterations):
            n_rand = goal if np.random.random() < self.goal_sample_rate else \
                     RRTNode(np.random.uniform(-40, 40), np.random.uniform(-40, 40))
            
            n_near = min(self.nodes, key=lambda n: n.distance_to(n_rand))
            dist = n_near.distance_to(n_rand)
            if dist < 1e-6: continue
            
            step = min(self.max_step, dist)
            n_new = RRTNode(n_near.x + (step/dist)*(n_rand.x - n_near.x),
                            n_near.y + (step/dist)*(n_rand.y - n_near.y))
            
            if self.is_collision_free(n_near, n_new) and self.is_feasible_turn(n_near, n_new):
                n_new.parent = n_near
                self.nodes.append(n_new)
                if n_new.distance_to(goal) < self.max_step:
                    goal.parent = n_new
                    path, curr = [], goal
                    while curr: 
                        path.append(curr)
                        curr = curr.parent
                    return list(reversed(path))
        return None

class ParkingController(Node):
    def __init__(self):
        super().__init__('parking_controller')
        self.start_pose = RRTNode(0.0, 0.0, 0.0) 
        self.planned_path = None
        self.current_path_idx = 0
        self.parking_state = "idle"
        self.planner = RRTPlanner()
        self.obstacles = []

        # CONTROLLER PARAMS (Full Physical Ability)
        self.max_physical_steer = 0.61 
        self.lookahead_dist = 1.8 
        self.L = 1.75             
        
        self.controlPub = self.create_publisher(AckermannDrive, '/ackermann_cmd', 1)
        self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.create_subscription(Float32MultiArray, "/obstacles", self.obstacle_callback, 10)
        self.create_timer(0.05, self.control_loop)

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = quaternion_to_euler([q.x, q.y, q.z, q.w])
        self.start_pose.x, self.start_pose.y, self.start_pose.theta = p.x, p.y, yaw

    def obstacle_callback(self, msg):
        new_obs = []
        for i in range(0, len(msg.data), 3):
            new_obs.append({'x': msg.data[i], 'y': msg.data[i+1], 'area': msg.data[i+2]})
        self.obstacles = new_obs

    def start_parking(self, start_pose, goal_x, goal_y, goal_yaw):
        self.goal_pose = RRTNode(goal_x, goal_y, goal_yaw)
        self.planner.set_obstacles(self.obstacles)
        self.planned_path = self.planner.plan(self.start_pose, self.goal_pose)
        
        if self.planned_path:
            self.current_path_idx = 0
            self.parking_state = "executing"
            self.get_logger().info(f"Smooth path found! Waypoints: {len(self.planned_path)}")
            return True
        return False

    def control_loop(self):
        if self.parking_state != "executing" or not self.planned_path:
            return

        curr_x, curr_y, curr_yaw = self.start_pose.x, self.start_pose.y, self.start_pose.theta
        print(f"Pos: ({curr_x:.1f}, {curr_y:.1f}, YAW:{curr_yaw:.1f} )") 

        # Pure Pursuit Lookahead Hunting
        target = self.planned_path[self.current_path_idx]
        for i in range(self.current_path_idx, len(self.planned_path)):
            d = math.hypot(self.planned_path[i].x - curr_x, self.planned_path[i].y - curr_y)
            if d > self.lookahead_dist:
                target = self.planned_path[i]
                self.current_path_idx = i
                break

        if math.hypot(self.planned_path[-1].x - curr_x, self.planned_path[-1].y - curr_y) < 0.6:
            self.get_logger().info("GOAL REACHED")
            self.stop()
            self.parking_state = "finished"
            return

        # Steering
        alpha = (math.atan2(target.y - curr_y, target.x - curr_x) - curr_yaw + math.pi) % (2 * math.pi) - math.pi
        steering = math.atan2(2.0 * self.L * math.sin(alpha), self.lookahead_dist)
        
        cmd = AckermannDrive()
        cmd.speed = 4.0
        cmd.steering_angle = float(max(-self.max_physical_steer, min(self.max_physical_steer, steering)))
        self.controlPub.publish(cmd)

    def stop(self):
        cmd = AckermannDrive()
        cmd.speed, cmd.steering_angle = 0.0, 0.0
        self.controlPub.publish(cmd)

def main():
    rclpy.init()
    node = ParkingController()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()


