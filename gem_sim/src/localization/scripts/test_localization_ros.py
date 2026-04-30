#!/usr/bin/env python3
"""
test_localization_ros.py — ROS2 integration test for the localization node.

IMPORTANT: Run the localization node with camera DISABLED to isolate GPS+IMU:
    python3 localization.py --ros-args -p use_camera:=false

Then in a second terminal:
    python3 test_localization_ros.py

If the GEM simulator is running alongside, the use_camera:=false flag prevents
VO from picking up simulator camera frames and inflating the position estimate.
"""

import sys
import math
import time
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus
from nav_msgs.msg    import Odometry


# ============================================================
# Configuration
# ============================================================
GPS_LAT0      = 40.1164
GPS_LON0      = -88.2434
DRIVE_SPEED   = 2.0          # m/s simulated forward speed
DRIVE_SECONDS = 8.0          # how long to publish fake sensors
DRAIN_SEC     = 0.5          # wait for in-flight msgs after sensors stop
IDLE_SEC      = 2.0          # verify node goes quiet after sensors stop
IMU_HZ        = 50
GPS_HZ        = 5

EXPECTED_X    = DRIVE_SPEED * DRIVE_SECONDS    # 16.0 m
POSITION_TOL  = 5.0                            # metres


class LocalizationIntegrationTest(Node):

    def __init__(self):
        super().__init__('localization_integration_test')

        self.imu_pub = self.create_publisher(Imu,       '/imu/data', 10)
        self.gps_pub = self.create_publisher(NavSatFix, '/gps/fix',  10)

        self.odom_msgs   = []
        self._drive_end_idx = None

        # Diagnostic: track GPS x vs odom x per GPS update
        self._gps_log  = []   # [(t, gps_x)]
        self._odom_log = []   # sampled (t, odom_x) at each GPS fire

        self.create_subscription(Odometry, '/odometry/global', self._odom_cb, 10)

        self._t0      = time.time()
        self._sim_x   = 0.0
        self._sim_y   = 0.0
        self._sim_yaw = 0.0
        self._sim_dt  = 1.0 / IMU_HZ

        self.imu_timer = self.create_timer(1.0 / IMU_HZ, self._publish_imu)
        self.gps_timer = self.create_timer(1.0 / GPS_HZ, self._publish_gps)

        self.get_logger().info(
            f'Test: {DRIVE_SPEED} m/s east for {DRIVE_SECONDS} s  '
            f'→ expected x = {EXPECTED_X:.1f} m\n'
            f'  Make sure localization.py is running with:\n'
            f'    python3 localization.py --ros-args -p use_camera:=false'
        )

    # -----------------------------------------------------------------------
    def _odom_cb(self, msg: Odometry):
        self.odom_msgs.append(msg)

    def _current_odom_x(self):
        return self.odom_msgs[-1].pose.pose.position.x if self.odom_msgs else 0.0

    # -----------------------------------------------------------------------
    def _publish_imu(self):
        if time.time() - self._t0 >= DRIVE_SECONDS:
            self.imu_timer.cancel()
            self.gps_timer.cancel()
            self.get_logger().info(
                f'Drive done — sim_x={self._sim_x:.2f} m, '
                f'odom_x={self._current_odom_x():.2f} m'
            )
            return

        msg = Imu()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.linear_acceleration.x = 0.0
        msg.linear_acceleration.y = 0.0
        msg.linear_acceleration.z = 9.81
        msg.angular_velocity.z    = 0.0
        msg.orientation.w         = 1.0
        self.imu_pub.publish(msg)

        self._sim_x += DRIVE_SPEED * math.cos(self._sim_yaw) * self._sim_dt
        self._sim_y += DRIVE_SPEED * math.sin(self._sim_yaw) * self._sim_dt

    def _publish_gps(self):
        R_EARTH = 6_371_000.0
        lat_mid = math.radians(GPS_LAT0)
        d_lat   = self._sim_y / R_EARTH
        d_lon   = self._sim_x / (R_EARTH * math.cos(lat_mid))

        msg = NavSatFix()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'gps'
        msg.status.status   = NavSatStatus.STATUS_FIX
        msg.latitude        = GPS_LAT0 + math.degrees(d_lat)
        msg.longitude       = GPS_LON0 + math.degrees(d_lon)
        msg.altitude        = 220.0
        self.gps_pub.publish(msg)

        # Log GPS vs current odom for the diagnostic table
        t = time.time() - self._t0
        self._gps_log.append((t, self._sim_x))
        self._odom_log.append((t, self._current_odom_x()))

    # -----------------------------------------------------------------------
    def run_and_check(self) -> bool:
        # Phase 1 — drive
        drive_end = time.time() + DRIVE_SECONDS
        while time.time() < drive_end:
            rclpy.spin_once(self, timeout_sec=0.02)

        # Phase 2 — drain in-flight DDS messages
        drain_end = time.time() + DRAIN_SEC
        while time.time() < drain_end:
            rclpy.spin_once(self, timeout_sec=0.02)

        self._drive_end_idx = len(self.odom_msgs)

        # Phase 3 — idle: verify node goes quiet
        idle_end = time.time() + IDLE_SEC
        while time.time() < idle_end:
            rclpy.spin_once(self, timeout_sec=0.02)

        n_idle = len(self.odom_msgs) - self._drive_end_idx

        self._print_gps_trace()
        return self._evaluate(n_idle)

    # -----------------------------------------------------------------------
    def _print_gps_trace(self):
        """Print GPS truth vs odom at each GPS update so you can see tracking."""
        print('\n  GPS truth vs Odom position (sampled at each GPS fire):')
        print(f'  {"t(s)":>5}  {"GPS x(m)":>9}  {"Odom x(m)":>10}  {"Error(m)":>9}')
        print('  ' + '-' * 42)
        for (t, gps_x), (_, odom_x) in zip(self._gps_log, self._odom_log):
            err = odom_x - gps_x
            print(f'  {t:>5.1f}  {gps_x:>9.2f}  {odom_x:>10.2f}  {err:>+9.2f}')

    # -----------------------------------------------------------------------
    def _evaluate(self, n_idle: int) -> bool:
        results = []
        n_drive = self._drive_end_idx

        # Check 1 — messages received
        ok1 = n_drive > 0
        results.append((
            'Odometry messages received during drive', ok1,
            f'{n_drive} messages' if ok1
            else 'NONE — is localization.py running with use_camera:=false?'
        ))
        if not ok1:
            self._print_results(results)
            return False

        last = self.odom_msgs[self._drive_end_idx - 1]
        gx   = last.pose.pose.position.x
        gy   = last.pose.pose.position.y

        # Check 2 — frame IDs
        ok2 = (last.header.frame_id == 'map' and last.child_frame_id == 'base_link')
        results.append((
            'Correct frame IDs (map → base_link)', ok2,
            f"header={last.header.frame_id}, child={last.child_frame_id}"
        ))

        # Check 3 — position moved in +x
        ok3 = gx > 1.0
        results.append((
            'Position moved in +x (east)', ok3,
            f'x={gx:.2f} m, y={gy:.2f} m  (expected ~{EXPECTED_X:.1f} m)'
        ))

        # Check 4 — within tolerance
        error = math.hypot(gx - EXPECTED_X, gy)
        ok4   = error < POSITION_TOL
        results.append((
            f'Position within {POSITION_TOL:.0f} m of truth ({EXPECTED_X:.1f}, 0)', ok4,
            f'error = {error:.2f} m'
            + ('' if ok4 else
               '\n         Hint: run localization.py with --ros-args -p use_camera:=false')
        ))

        # Check 5 — yaw near 0
        q   = last.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        ok5 = abs(yaw) < 0.3
        results.append((
            'Yaw near 0° (driving east)', ok5,
            f'yaw = {math.degrees(yaw):.1f}°'
        ))

        # Check 6 — velocity non-negative
        vx  = last.twist.twist.linear.x
        ok6 = vx >= 0.0
        note = '  ← 0 expected without VO' if vx == 0.0 else ''
        results.append((
            'Forward velocity vx ≥ 0', ok6, f'vx = {vx:.3f} m/s{note}'
        ))

        # Check 7 — publish rate
        rate = n_drive / DRIVE_SECONDS
        ok7  = rate > 20
        results.append((
            'Publish rate ≥ 20 Hz', ok7, f'{rate:.1f} Hz'
        ))

        # Check 8 — node goes quiet when sensors stop
        ok8 = n_idle <= 5
        results.append((
            f'Node quiet after sensors stop (≤5 stragglers)', ok8,
            f'{n_idle} messages in {IDLE_SEC:.0f}s idle phase'
        ))

        self._print_results(results)
        return all(r[1] for r in results)

    # -----------------------------------------------------------------------
    def _print_results(self, results):
        print('\n' + '=' * 60)
        print('  LOCALIZATION INTEGRATION TEST RESULTS')
        print('=' * 60)
        all_pass = True
        for name, ok, detail in results:
            status = '  PASS' if ok else '  FAIL'
            if not ok:
                all_pass = False
            print(f'{status}  {name}')
            print(f'         {detail}')
        print('=' * 60)
        print(f'  Overall: {"ALL TESTS PASSED ✓" if all_pass else "SOME TESTS FAILED ✗"}')
        print('=' * 60)


# ============================================================
def main():
    rclpy.init()
    node = LocalizationIntegrationTest()
    passed = node.run_and_check()
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()