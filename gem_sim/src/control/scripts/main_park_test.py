#!/usr/bin/env python3
"""
Main loop to test the Parking Control (RRT-based) system.
This script initializes the ParkingController and sends a specific
parking goal to the RRT planner.
"""


import rclpy
import sys
import os
import math
import signal
from nav_msgs.msg import Odometry


# Ensure script can find local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from park_control import ParkingController


# --- CONFIGURATION ---
# Set your desired parking spot coordinates here
GOAL_X = 0
GOAL_Y = 0
GOAL_YAW = 0 # 90 degrees (facing "up")
# ---------------------


class ParkingRunner:
   def __init__(self):
       self.node = ParkingController()
       self.goal_sent = False
      
       # Subscribe to odom to get the current position for the start pose
       self.odom_sub = self.node.create_subscription(
           Odometry,
           'odom',
           self.odom_callback,
           10
       )
      
       self.node.get_logger().info("Waiting for Odometry to set start pose...")


   def odom_callback(self, msg):
       # We only want to trigger the planning once
       if not self.goal_sent:
           self.node.get_logger().info("Odometry received. Planning path to parking spot...")
          
           # Start the parking maneuver using the current odom as start_pose
           success = self.node.start_parking(
               goal_x=GOAL_X,
               goal_y=GOAL_Y,
               goal_yaw=GOAL_YAW
           )
          
           if success:
               self.node.get_logger().info("Path found! Executing maneuver...")
               self.goal_sent = True
           else:
               self.node.get_logger().error("RRT failed to find a path. Retrying in 2 seconds...")
               # In a real scenario, you might want to move slightly or wait
               import time
               time.sleep(2.0)


def main(args=None):
   rclpy.init(args=args)
  
   runner = ParkingRunner()


   def signal_handler(sig, frame):
       runner.node.stop()
       rclpy.shutdown()
       sys.exit(0)


   signal.signal(signal.SIGINT, signal_handler)


   try:
       rclpy.spin(runner.node)
   except KeyboardInterrupt:
       pass
   finally:
       runner.node.destroy()
       if rclpy.ok():
           rclpy.shutdown()


if __name__ == '__main__':
   main()






