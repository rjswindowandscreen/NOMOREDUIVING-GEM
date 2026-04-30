#!/usr/bin/env python3
"""
test_localization.py — Unit tests for the localization module.

No ROS2 required. Run from the src/localization/ directory:
    python3 test_localization.py

Tests cover:
  1. GPS coordinate conversion (latlon_to_xy)
  2. YawKF  — predict + update
  3. LocalKF — predict + VO update + position integration
  4. GlobalKF — GPS update + outlier rejection
  5. Full pipeline: simulated straight-line drive
"""

import sys
import os
import math
import unittest
import numpy as np

# Add scripts/ to path (same as localization.py does at runtime)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from kalman_filters  import YawKF, LocalKF, GlobalKF
from gps_utils       import latlon_to_xy, xy_to_latlon, wrap_angle, euclidean_distance


# ============================================================
# 1. GPS Utilities
# ============================================================
class TestGPSUtils(unittest.TestCase):

    def test_latlon_to_xy_origin(self):
        """Datum point should map to (0, 0)."""
        lat0, lon0 = 40.1164, -88.2434   # UIUC campus
        x, y = latlon_to_xy(lat0, lon0, lat0, lon0)
        self.assertAlmostEqual(x, 0.0, places=6)
        self.assertAlmostEqual(y, 0.0, places=6)

    def test_latlon_to_xy_north(self):
        """Moving 0.001° north ≈ 111 m north."""
        lat0, lon0 = 40.1164, -88.2434
        x, y = latlon_to_xy(lat0 + 0.001, lon0, lat0, lon0)
        self.assertAlmostEqual(x, 0.0, places=1)
        self.assertAlmostEqual(y, 111.195, delta=0.5)   # ≈111 m/deg at this lat

    def test_latlon_to_xy_east(self):
        """Moving 0.001° east should give positive x."""
        lat0, lon0 = 40.1164, -88.2434
        x, y = latlon_to_xy(lat0, lon0 + 0.001, lat0, lon0)
        self.assertGreater(x, 0.0)
        self.assertAlmostEqual(y, 0.0, places=1)

    def test_roundtrip(self):
        """xy_to_latlon should be the inverse of latlon_to_xy."""
        lat0, lon0 = 40.1164, -88.2434
        for lat_off, lon_off in [(0.002, 0.003), (-0.001, 0.005), (0.0, -0.002)]:
            x, y = latlon_to_xy(lat0 + lat_off, lon0 + lon_off, lat0, lon0)
            lat_back, lon_back = xy_to_latlon(x, y, lat0, lon0)
            self.assertAlmostEqual(lat_back, lat0 + lat_off, places=6)
            self.assertAlmostEqual(lon_back, lon0 + lon_off, places=6)

    def test_wrap_angle(self):
        """wrap_angle should keep angles in [-π, π]."""
        self.assertAlmostEqual(wrap_angle(0.0),        0.0,       places=6)
        self.assertAlmostEqual(wrap_angle(math.pi),    math.pi,   places=6)
        self.assertAlmostEqual(wrap_angle(3 * math.pi), math.pi,  places=5)
        self.assertAlmostEqual(wrap_angle(-3 * math.pi), -math.pi, places=5)

    def test_euclidean_distance(self):
        self.assertAlmostEqual(euclidean_distance(0, 0, 3, 4), 5.0, places=6)


# ============================================================
# 2. YawKF
# ============================================================
class TestYawKF(unittest.TestCase):

    def test_initial_state(self):
        kf = YawKF()
        self.assertAlmostEqual(kf.yaw,      0.0, places=6)
        self.assertAlmostEqual(kf.yaw_rate, 0.0, places=6)

    def test_predict_constant_gyro(self):
        """Constant gyro of 0.1 rad/s for 1 s should yield yaw ≈ 0.1 rad."""
        kf = YawKF()
        for _ in range(100):
            kf.predict(gyro_z=0.1, dt=0.01)
        self.assertAlmostEqual(kf.yaw, 0.1, delta=0.01)
        self.assertAlmostEqual(kf.yaw_rate, 0.1, delta=0.01)

    def test_predict_zero_gyro(self):
        """Zero gyro should not change yaw."""
        kf = YawKF()
        for _ in range(50):
            kf.predict(gyro_z=0.0, dt=0.01)
        self.assertAlmostEqual(kf.yaw, 0.0, delta=1e-6)

    def test_update_vo_pulls_toward_measurement(self):
        """VO update with a yaw different from predicted should move estimate."""
        kf = YawKF()
        # Predict with zero gyro (stays at 0)
        for _ in range(10):
            kf.predict(0.0, 0.01)
        yaw_before = kf.yaw
        # VO says yaw is 0.3 rad
        kf.update_vo(0.3)
        self.assertGreater(kf.yaw, yaw_before)   # pulled toward 0.3

    def test_covariance_decreases_after_update(self):
        """P[0,0] should shrink after a measurement update."""
        kf = YawKF()
        kf.predict(0.0, 0.1)
        p_before = kf.P[0, 0]
        kf.update_vo(0.0)
        self.assertLess(kf.P[0, 0], p_before)

    def test_wrap_on_predict(self):
        """Yaw should stay in [-π, π] when it crosses ±π."""
        kf = YawKF()
        # Spin at 1 rad/s for 4 seconds — crosses π
        for _ in range(400):
            kf.predict(gyro_z=1.0, dt=0.01)
        self.assertGreaterEqual(kf.yaw, -math.pi)
        self.assertLessEqual(kf.yaw,     math.pi)


# ============================================================
# 3. LocalKF
# ============================================================
class TestLocalKF(unittest.TestCase):

    def test_initial_state(self):
        kf = LocalKF()
        self.assertAlmostEqual(float(kf.x[0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(kf.x[1, 0]), 0.0, places=6)
        self.assertAlmostEqual(kf.world_x, 0.0, places=6)
        self.assertAlmostEqual(kf.world_y, 0.0, places=6)

    def test_predict_accelerates(self):
        """Forward acceleration should increase vx."""
        kf = LocalKF()
        for _ in range(50):
            kf.predict(ax=1.0, ay=0.0, dt=0.01)
        self.assertGreater(float(kf.x[0, 0]), 0.0)

    def test_integrate_position_straight(self):
        """
        Driving straight (yaw=0) at 2 m/s for 5 s should give world_x ≈ 10 m.
        """
        kf = LocalKF()
        kf.x[0, 0] = 2.0   # set vx directly (no IMU noise in test)
        for _ in range(500):
            kf.integrate_position(yaw=0.0, dt=0.01)
        self.assertAlmostEqual(kf.world_x, 10.0, delta=0.1)
        self.assertAlmostEqual(kf.world_y,  0.0, delta=0.1)

    def test_integrate_position_90deg(self):
        """
        Driving at yaw=π/2 (facing north) at 2 m/s for 5 s
        should give world_y ≈ 10 m, world_x ≈ 0.
        """
        kf = LocalKF()
        kf.x[0, 0] = 2.0
        for _ in range(500):
            kf.integrate_position(yaw=math.pi / 2, dt=0.01)
        self.assertAlmostEqual(kf.world_x,  0.0, delta=0.1)
        self.assertAlmostEqual(kf.world_y, 10.0, delta=0.1)

    def test_vo_update_corrects_velocity(self):
        """VO displacement implying vx=3 should pull estimate away from 0."""
        kf = LocalKF()
        kf.update_vo(dx=0.3, dy=0.0, dt=0.1)   # implies vx=3 m/s
        self.assertGreater(float(kf.x[0, 0]), 0.0)

    def test_speed_property(self):
        kf = LocalKF()
        kf.x[0, 0] = 3.0
        kf.x[1, 0] = 4.0
        self.assertAlmostEqual(kf.speed, 5.0, places=5)


# ============================================================
# 4. GlobalKF
# ============================================================
class TestGlobalKF(unittest.TestCase):

    def test_initial_state(self):
        kf = GlobalKF()
        gx, gy = kf.position
        self.assertAlmostEqual(gx, 0.0, places=6)
        self.assertAlmostEqual(gy, 0.0, places=6)

    def test_gps_update_corrects_position(self):
        """GPS fix at (5, 5) should pull estimate from (0,0) toward (5,5)."""
        kf = GlobalKF()
        kf.predict(vx_w=0.0, vy_w=0.0, dt=0.1)
        kf.update_gps(5.0, 5.0)
        gx, gy = kf.position
        self.assertGreater(gx, 0.0)
        self.assertGreater(gy, 0.0)

    def test_gps_outlier_rejected(self):
        """GPS jump larger than reject threshold should be ignored."""
        kf = GlobalKF()
        kf.predict(0.0, 0.0, 0.1)
        # Position near origin
        kf.update_gps(0.1, 0.1)
        gx_before, gy_before = kf.position

        # Huge jump — should be rejected
        kf.update_gps(500.0, 500.0)
        gx_after, gy_after = kf.position

        self.assertAlmostEqual(gx_before, gx_after, delta=0.1)
        self.assertAlmostEqual(gy_before, gy_after, delta=0.1)

    def test_covariance_grows_on_predict(self):
        """Covariance should grow during prediction (no sensor update)."""
        kf = GlobalKF()
        p0 = kf.P[0, 0]
        for _ in range(10):
            kf.predict(1.0, 0.0, 0.1)
        self.assertGreater(kf.P[0, 0], p0)

    def test_repeated_gps_converges(self):
        """
        Repeated GPS updates at the same point should converge the estimate.
        We use a small target (3, 4) m — within the 8 m outlier threshold —
        so the GPS updates aren't rejected.
        """
        kf = GlobalKF()
        for _ in range(30):
            kf.predict(0.0, 0.0, 0.1)
            kf.update_gps(3.0, 4.0)
        gx, gy = kf.position
        self.assertAlmostEqual(gx, 3.0, delta=1.0)
        self.assertAlmostEqual(gy, 4.0, delta=1.0)


# ============================================================
# 5. Full pipeline — simulated straight-line drive
# ============================================================
class TestFullPipeline(unittest.TestCase):
    """
    Simulate the complete IMU → LocalKF → GlobalKF pipeline for
    a vehicle driving due east at 2 m/s for 10 seconds.

    Expected result: world_x ≈ 20 m, world_y ≈ 0 m.
    GPS fixes every 10 IMU ticks keep it anchored.
    """

    def test_straight_drive_east(self):
        yaw_kf    = YawKF()
        local_kf  = LocalKF()
        global_kf = GlobalKF()

        dt        = 0.01    # 100 Hz IMU
        duration  = 10.0    # seconds
        n_steps   = int(duration / dt)
        gps_every = 10      # GPS update every 10 IMU steps ≈ 10 Hz

        # Simulate: vehicle at yaw=0, forward accel = 0 (constant velocity trick:
        # set vx directly to 2 m/s — avoids needing to ramp up)
        local_kf.x[0, 0] = 2.0   # vx_body = 2 m/s

        for i in range(n_steps):
            # IMU predict (zero accel — constant velocity)
            yaw_kf.predict(gyro_z=0.0, dt=dt)
            local_kf.predict(ax=0.0, ay=0.0, dt=dt)
            local_kf.integrate_position(yaw_kf.yaw, dt)

            vx_w = math.cos(yaw_kf.yaw) * float(local_kf.x[0, 0])
            vy_w = math.sin(yaw_kf.yaw) * float(local_kf.x[0, 0])
            global_kf.predict(vx_w, vy_w, dt)
            global_kf.x[0, 0] = local_kf.world_x
            global_kf.x[1, 0] = local_kf.world_y

            # GPS update every gps_every steps
            if i % gps_every == 0:
                true_x = 2.0 * (i * dt)   # ground truth position
                # Add small noise (0.5 m std)
                noisy_x = true_x + np.random.normal(0, 0.5)
                noisy_y = np.random.normal(0, 0.5)
                global_kf.update_gps(noisy_x, noisy_y)
                local_kf.world_x = global_kf.x[0, 0]
                local_kf.world_y = global_kf.x[1, 0]

        gx, gy = global_kf.position
        print(f'\n[pipeline] Final position: x={gx:.2f} m, y={gy:.2f} m '
              f'(expected x≈20, y≈0)')
        self.assertAlmostEqual(gx, 20.0, delta=2.0)   # within 2 m of truth
        self.assertAlmostEqual(gy,  0.0, delta=2.0)

    def test_turning_drive(self):
        """
        Vehicle turns at 0.1 rad/s while moving at 2 m/s for 5 s.
        Just checks the pipeline doesn't crash and position is plausible.
        """
        yaw_kf    = YawKF()
        local_kf  = LocalKF()
        global_kf = GlobalKF()

        local_kf.x[0, 0] = 2.0
        dt = 0.01

        for i in range(500):
            yaw_kf.predict(gyro_z=0.1, dt=dt)
            local_kf.predict(ax=0.0, ay=0.0, dt=dt)
            local_kf.integrate_position(yaw_kf.yaw, dt)
            vx_w = math.cos(yaw_kf.yaw) * 2.0
            vy_w = math.sin(yaw_kf.yaw) * 2.0
            global_kf.predict(vx_w, vy_w, dt)
            global_kf.x[0, 0] = local_kf.world_x
            global_kf.x[1, 0] = local_kf.world_y

        gx, gy = global_kf.position
        dist = math.hypot(gx, gy)
        print(f'\n[pipeline] Turning drive final pos: x={gx:.2f}, y={gy:.2f}, '
              f'dist from origin={dist:.2f} m')
        # After 5 s at 2 m/s, distance from origin ≤ 10 m (arc, not straight)
        self.assertLessEqual(dist, 11.0)
        self.assertGreater(dist, 0.0)


# ============================================================
# Run
# ============================================================
if __name__ == '__main__':
    print('=' * 60)
    print(' Localization Unit Tests')
    print('=' * 60)
    unittest.main(verbosity=2)