#!/usr/bin/env python3
"""
localization_node.py — Fuses IMU + GPS + INS velocity into /odometry/global.

Subscriptions  (names from ros_topics.py)
  /imu          sensor_msgs/Imu                         (~50 Hz) predict step
  /navsatfix    sensor_msgs/NavSatFix                   (~40 Hz) position update
  /twist_ins    geometry_msgs/TwistWithCovarianceStamped (~40 Hz) velocity update + ZUPT

Publications
  /odometry/global  nav_msgs/Odometry

Two key improvements over the basic IMU+GPS version:

1. Gyro bias correction via ZUPT
   The IMU gyro has a small constant offset (bias) that makes yaw drift even
   when stationary. We detect stationary periods using /twist_ins speed, and
   during those periods we update a running bias estimate using an exponential
   moving average. The bias is subtracted from every gyro reading.
   No fixed time window assumed — works whenever the car naturally stops.

2. Velocity correction from Septentrio INS (/twist_ins)
   Instead of relying on noisy IMU acceleration integration for velocity,
   we use the Septentrio's fused velocity output directly as a KF update.
   This eliminates the 0.2-0.5 m/s drift seen when stationary.
   When the car is confirmed stationary, we also do a zero-velocity update
   (ZUPT) with tight covariance to actively pull velocity to zero.
"""

import os
import sys
import math
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg    import Odometry
from geometry_msgs.msg import TwistWithCovarianceStamped

from kalman_filter import FusionKF
from gps_utils     import latlon_to_xy, euler_to_quat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# topics.py: localization/scripts/ -> localization/ -> src/
import sys as _loc_sys
_loc_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from topics import IMU as IMU_TOPIC, NAVSATFIX as GPS_TOPIC, TWIST_INS as TWIST_INS_TOPIC, REAL_ODOM as ODOM_TOPIC


# ── ZUPT tuning ───────────────────────────────────────────────────────────────
# car is considered stationary when INS speed stays below this for long enough
STATIONARY_SPEED_THRESH = 0.05   # m/s — below this we might be stopped
STATIONARY_CONFIRM_MSGS = 5      # need this many consecutive low-speed messages
                                 # at ~40 Hz that's ~0.125 s — fast enough to
                                 # catch a stop, slow enough to ignore noise

# exponential moving average rate for gyro bias update
# 0.02 means it takes ~50 stationary readings to fully converge (~1.25 s)
BIAS_EMA_ALPHA = 0.02


class LocalizationNode(Node):

    def __init__(self):
        super().__init__('localization_node')

        self.kf = FusionKF()

        # GPS datum
        self.lat0 = None
        self.lon0 = None

        # IMU timing
        self.last_imu_time = None

        # gyro bias state — updated automatically during stationary periods
        self.gyro_bias        = 0.0
        self.stationary_count = 0
        self.is_stationary    = False

        # subscribers
        self.create_subscription(Imu,                        IMU_TOPIC,       self.imu_cb,       10)
        self.create_subscription(NavSatFix,                  GPS_TOPIC,       self.gps_cb,       10)
        self.create_subscription(TwistWithCovarianceStamped, TWIST_INS_TOPIC, self.twist_ins_cb, 10)

        # publisher
        self.odom_pub = self.create_publisher(Odometry, ODOM_TOPIC, 10)

        self.get_logger().info('LocalizationNode started')
        self.get_logger().info(f'  IMU       : {IMU_TOPIC}')
        self.get_logger().info(f'  GPS       : {GPS_TOPIC}')
        self.get_logger().info(f'  INS vel   : {TWIST_INS_TOPIC}')
        self.get_logger().info(f'  Output    : {ODOM_TOPIC}')
        self.get_logger().info('Waiting for GPS datum...')

    # -------------------------------------------------------------------------
    def imu_cb(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9

        if self.last_imu_time is None:
            self.last_imu_time = now
            return

        dt = now - self.last_imu_time
        self.last_imu_time = now

        if dt <= 0 or dt > 0.5:
            return

        ax     = msg.linear_acceleration.x
        ay     = msg.linear_acceleration.y
        gyro_z = msg.angular_velocity.z

        # update gyro bias using EMA when car is confirmed stationary
        # this happens naturally whenever the car stops — no time window needed
        if self.is_stationary:
            self.gyro_bias = (1.0 - BIAS_EMA_ALPHA) * self.gyro_bias \
                           + BIAS_EMA_ALPHA * gyro_z

        # subtract bias before passing to KF
        corrected_gyro_z = gyro_z - self.gyro_bias

        self.kf.predict(ax, ay, corrected_gyro_z, dt)
        self.publish_odom()

    # -------------------------------------------------------------------------
    def gps_cb(self, msg):
        if msg.status.status < 0:
            return

        if self.lat0 is None:
            self.lat0 = msg.latitude
            self.lon0 = msg.longitude
            self.get_logger().info(
                f'GPS datum set: {self.lat0:.6f}, {self.lon0:.6f}'
            )
            return

        gps_x, gps_y = latlon_to_xy(
            msg.latitude, msg.longitude, self.lat0, self.lon0
        )
        self.kf.update_gps(gps_x, gps_y)

    # -------------------------------------------------------------------------
    def twist_ins_cb(self, msg):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        speed = math.hypot(vx, vy)

        # ── ZUPT detection ────────────────────────────────────────────────
        # requires consecutive low-speed readings — avoids false triggers
        # from brief GPS glitches or momentary noise
        if speed < STATIONARY_SPEED_THRESH:
            self.stationary_count += 1
        else:
            self.stationary_count = 0
            self.is_stationary    = False

        if self.stationary_count >= STATIONARY_CONFIRM_MSGS:
            if not self.is_stationary:
                self.get_logger().debug(
                    f'Stationary detected — gyro bias: {self.gyro_bias:.5f} rad/s'
                )
            self.is_stationary = True

        # ── velocity KF update ────────────────────────────────────────────
        if self.is_stationary:
            # ZUPT: car is confirmed stopped — force velocity to zero
            # use very tight covariance since we are certain
            self.kf.update_velocity(0.0, 0.0, cov_vx=0.01, cov_vy=0.01)
        else:
            # use INS velocity directly
            # pull covariance from message if the driver provides it,
            # otherwise fall back to a reasonable default
            cov = msg.twist.covariance   # 6x6 flat row-major
            cov_vx = cov[0] if cov[0] > 1e-9 else 0.1
            cov_vy = cov[7] if cov[7] > 1e-9 else 0.1
            self.kf.update_velocity(vx, vy, cov_vx=cov_vx, cov_vy=cov_vy)

    # -------------------------------------------------------------------------
    def publish_odom(self):
        msg = Odometry()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.child_frame_id  = 'base_link'

        x, y   = self.kf.get_position()
        vx, vy = self.kf.get_velocity()

        msg.pose.pose.position.x  = x
        msg.pose.pose.position.y  = y
        msg.pose.pose.position.z  = 0.0
        msg.pose.pose.orientation = euler_to_quat(self.kf.get_yaw())

        msg.twist.twist.linear.x  = vx
        msg.twist.twist.linear.y  = vy
        msg.twist.twist.angular.z = self.kf.get_yaw_rate()

        self.odom_pub.publish(msg)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = LocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()