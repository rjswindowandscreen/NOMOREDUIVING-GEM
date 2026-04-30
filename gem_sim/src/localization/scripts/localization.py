#!/usr/bin/env python3
"""
localization_node.py — ROS2 node that fuses IMU + Visual Odometry + GPS
into a single /odometry/global estimate for the GEM vehicle.

Subscriptions
  /imu/data          sensor_msgs/Imu          (high frequency ~100 Hz)
  /camera/image_raw  sensor_msgs/Image        (camera frames for VO, optional)
  /gps/fix           sensor_msgs/NavSatFix    (absolute GPS fix ~10 Hz)

Publications
  /odometry/global   nav_msgs/Odometry        (fused pose + velocity)

Parameters
  use_camera  (bool, default True)
      Set false to disable VO entirely — useful when testing GPS+IMU alone
      or when the camera topic is noisy / occupied by the simulator.
      Example: ros2 run ... --ros-args -p use_camera:=false
"""

import math
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu, Image, NavSatFix
from nav_msgs.msg   import Odometry

from cv_bridge import CvBridge

from kalman_filters  import YawKF, LocalKF, GlobalKF
from visual_odometry import VisualOdometry
from gps_utils       import latlon_to_xy, euler_to_quat


class LocalizationNode(Node):

    def __init__(self):
        super().__init__('localization_node')

        # ----- Parameters ---------------------------------------------------
        self.declare_parameter('use_camera', True)
        use_camera = self.get_parameter('use_camera').get_parameter_value().bool_value

        # ----- Kalman Filters -----------------------------------------------
        self.yaw_kf    = YawKF()
        self.local_kf  = LocalKF()
        self.global_kf = GlobalKF()

        # ----- Visual Odometry (only if camera enabled) ---------------------
        self.vo     = VisualOdometry(n_features=500) if use_camera else None
        self.bridge = CvBridge()

        # ----- GPS datum (set on first valid fix) ---------------------------
        self.lat0: float | None = None
        self.lon0: float | None = None

        # ----- Timing -------------------------------------------------------
        self.last_imu_time: float | None = None
        self.last_cam_time: float | None = None

        # ----- Subscribers --------------------------------------------------
        self.create_subscription(Imu,       '/imu/data', self.imu_cb, 10)
        self.create_subscription(NavSatFix, '/gps/fix',  self.gps_cb, 10)

        if use_camera:
            self.create_subscription(Image, '/camera/image_raw', self.cam_cb, 10)
            self.get_logger().info('Camera/VO enabled')
        else:
            self.get_logger().info('Camera/VO DISABLED (use_camera:=false)')

        # ----- Publisher ----------------------------------------------------
        self.odom_pub = self.create_publisher(Odometry, '/odometry/global', 10)

        self.get_logger().info('LocalizationNode started — waiting for GPS datum...')

    # -----------------------------------------------------------------------
    # IMU callback — drives the predict loop at ~100 Hz
    # -----------------------------------------------------------------------
    def imu_cb(self, msg: Imu):
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

        # 1. Update yaw from gyro
        self.yaw_kf.predict(gyro_z, dt)

        # 2. Predict body-frame velocity from IMU acceleration
        self.local_kf.predict(ax, ay, dt)

        # 3. Integrate world-frame position
        self.local_kf.integrate_position(self.yaw_kf.yaw, dt)

        # 4. Propagate global KF (GPS corrects this at 5–10 Hz)
        vx_w, vy_w = self._world_velocity()
        self.global_kf.predict(vx_w, vy_w, dt)

        # Sync global KF position with local integration
        # (GPS feedback from gps_cb overrides this when a fix arrives)
        self.global_kf.x[0, 0] = self.local_kf.world_x
        self.global_kf.x[1, 0] = self.local_kf.world_y

        self._publish_odom(now)

    # -----------------------------------------------------------------------
    # Camera callback — VO update (only called when use_camera=True)
    # -----------------------------------------------------------------------
    def cam_cb(self, msg: Image):
        if self.vo is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        if self.last_cam_time is None:
            self.last_cam_time = now
            return

        dt = now - self.last_cam_time
        self.last_cam_time = now

        if dt <= 0 or dt > 1.0:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        result = self.vo.update(frame, self.local_kf.speed, dt)

        if result is None:
            return

        dx, dy, dyaw = result

        # Correct yaw with VO heading estimate
        vo_yaw = self.yaw_kf.yaw + dyaw
        self.yaw_kf.update_vo(vo_yaw)

        # Correct body-frame velocity with VO displacement
        self.local_kf.update_vo(dx, dy, dt)

    # -----------------------------------------------------------------------
    # GPS callback — absolute position correction
    # -----------------------------------------------------------------------
    def gps_cb(self, msg: NavSatFix):
        if msg.status.status < 0:
            return

        # Set datum from first valid fix
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

        # Correct global KF with absolute GPS position
        self.global_kf.update_gps(gps_x, gps_y)

        # Feed GPS correction back into local KF world position
        gx, gy = self.global_kf.position
        self.local_kf.world_x = gx
        self.local_kf.world_y = gy

    # -----------------------------------------------------------------------
    def _world_velocity(self):
        """Rotate body-frame velocity into world frame using current yaw."""
        vx_b = self.local_kf.speed
        vy_b = float(self.local_kf.x[1, 0])
        yaw  = self.yaw_kf.yaw
        vx_w =  math.cos(yaw) * vx_b - math.sin(yaw) * vy_b
        vy_w =  math.sin(yaw) * vx_b + math.cos(yaw) * vy_b
        return vx_w, vy_w

    def _publish_odom(self, timestamp: float):
        msg = Odometry()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.child_frame_id  = 'base_link'

        gx, gy = self.global_kf.position
        msg.pose.pose.position.x = gx
        msg.pose.pose.position.y = gy
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation = euler_to_quat(self.yaw_kf.yaw)

        vx_w, vy_w = self._world_velocity()
        msg.twist.twist.linear.x  = vx_w
        msg.twist.twist.linear.y  = vy_w
        msg.twist.twist.angular.z = self.yaw_kf.yaw_rate

        self.odom_pub.publish(msg)


# ---------------------------------------------------------------------------
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