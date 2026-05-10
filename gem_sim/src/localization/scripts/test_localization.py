#!/usr/bin/env python3
"""
test_localization.py — Unit tests for the FusionKF.

No ROS2 required. Run from src/localization/:
    python3 test_localization.py
"""

import sys
import os
import math
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

# Stub geometry_msgs so tests run without ROS2
import types
gm = types.ModuleType('geometry_msgs')
msg_mod = types.ModuleType('geometry_msgs.msg')
class Quaternion:
    def __init__(self): self.w = self.x = self.y = self.z = 0.0
msg_mod.Quaternion = Quaternion
gm.msg = msg_mod
sys.modules['geometry_msgs']     = gm
sys.modules['geometry_msgs.msg'] = msg_mod

from kalman_filter import FusionKF
from gps_utils     import latlon_to_xy, euclidean_distance


# =============================================================================
# 1. FusionKF — initial state
# =============================================================================
class TestFusionKFInit(unittest.TestCase):

    def test_initial_position_zero(self):
        kf = FusionKF()
        x, y = kf.get_position()
        self.assertAlmostEqual(x, 0.0, places=6)
        self.assertAlmostEqual(y, 0.0, places=6)

    def test_initial_velocity_zero(self):
        kf = FusionKF()
        vx, vy = kf.get_velocity()
        self.assertAlmostEqual(vx, 0.0, places=6)
        self.assertAlmostEqual(vy, 0.0, places=6)

    def test_initial_yaw_zero(self):
        kf = FusionKF()
        self.assertAlmostEqual(kf.get_yaw(), 0.0, places=6)

    def test_initial_speed_zero(self):
        kf = FusionKF()
        self.assertAlmostEqual(kf.get_speed(), 0.0, places=6)


# =============================================================================
# 2. Predict — yaw from gyro
# =============================================================================
class TestFusionKFPredictYaw(unittest.TestCase):

    def test_constant_gyro_integrates_yaw(self):
        """1 rad/s for 1 second → yaw ≈ 1 rad."""
        kf = FusionKF()
        for _ in range(100):
            kf.predict(ax=0, ay=0, gyro_z=1.0, dt=0.01)
        self.assertAlmostEqual(kf.get_yaw(), 1.0, delta=0.02)

    def test_zero_gyro_no_yaw_change(self):
        kf = FusionKF()
        for _ in range(50):
            kf.predict(ax=0, ay=0, gyro_z=0.0, dt=0.01)
        self.assertAlmostEqual(kf.get_yaw(), 0.0, delta=1e-6)

    def test_yaw_rate_set_from_gyro(self):
        kf = FusionKF()
        kf.predict(ax=0, ay=0, gyro_z=0.5, dt=0.01)
        self.assertAlmostEqual(kf.get_yaw_rate(), 0.5, delta=1e-6)

    def test_yaw_wraps_past_pi(self):
        """Yaw should stay in [-π, π] when spinning continuously."""
        kf = FusionKF()
        for _ in range(400):
            kf.predict(ax=0, ay=0, gyro_z=1.0, dt=0.01)
        self.assertGreaterEqual(kf.get_yaw(), -math.pi)
        self.assertLessEqual(kf.get_yaw(),    math.pi)


# =============================================================================
# 3. Predict — velocity and position from acceleration
# =============================================================================
class TestFusionKFPredictPosition(unittest.TestCase):

    def test_forward_acceleration_increases_vx(self):
        """ax=1 m/s² for 1 s at yaw=0 should give vx ≈ 1 m/s."""
        kf = FusionKF()
        for _ in range(100):
            kf.predict(ax=1.0, ay=0.0, gyro_z=0.0, dt=0.01)
        vx, vy = kf.get_velocity()
        self.assertAlmostEqual(vx, 1.0, delta=0.05)
        self.assertAlmostEqual(vy, 0.0, delta=0.05)

    def test_forward_accel_advances_position(self):
        """ax=1 m/s² for 2 s at yaw=0 → x ≈ 0.5*(1)*(2²) = 2 m."""
        kf = FusionKF()
        for _ in range(200):
            kf.predict(ax=1.0, ay=0.0, gyro_z=0.0, dt=0.01)
        x, y = kf.get_position()
        self.assertAlmostEqual(x, 2.0, delta=0.1)
        self.assertAlmostEqual(y, 0.0, delta=0.1)

    def test_lateral_accel_at_yaw_90(self):
        """
        At yaw=π/2 (facing north), ax=1 should advance y, not x.
        """
        kf = FusionKF()
        kf.x[4, 0] = math.pi / 2   # set yaw to north
        for _ in range(100):
            kf.predict(ax=1.0, ay=0.0, gyro_z=0.0, dt=0.01)
        x, y = kf.get_position()
        self.assertAlmostEqual(x, 0.0, delta=0.15)
        self.assertGreater(y, 0.3)  # 0.5*a*t^2 = 0.5*1*1^2 = 0.5 m

    def test_covariance_grows_without_gps(self):
        """P should grow during predict steps (no GPS update)."""
        kf = FusionKF()
        p0 = kf.P[0, 0]
        for _ in range(50):
            kf.predict(ax=0, ay=0, gyro_z=0.0, dt=0.01)
        self.assertGreater(kf.P[0, 0], p0)


# =============================================================================
# 4. GPS update
# =============================================================================
class TestFusionKFGPS(unittest.TestCase):

    def test_gps_snaps_position_on_first_fix(self):
        """
        With high initial P, first GPS fix should snap position close to GPS.
        """
        kf = FusionKF()
        kf.predict(ax=0, ay=0, gyro_z=0.0, dt=0.1)
        kf.update_gps(5.0, 3.0)
        x, y = kf.get_position()
        self.assertAlmostEqual(x, 5.0, delta=0.5)
        self.assertAlmostEqual(y, 3.0, delta=0.5)

    def test_gps_pulls_position_toward_measurement(self):
        """GPS update at (4, 4) should move estimate from (0, 0) toward it."""
        kf = FusionKF()
        x_before, _ = kf.get_position()
        kf.update_gps(4.0, 4.0)
        x_after, _ = kf.get_position()
        self.assertGreater(x_after, x_before)

    def test_gps_outlier_rejected(self):
        """GPS jump beyond threshold should be ignored."""
        kf = FusionKF()
        kf.update_gps(1.0, 1.0)   # small update accepted
        x_before, y_before = kf.get_position()
        kf.update_gps(500.0, 500.0)   # huge jump — should be rejected
        x_after, y_after = kf.get_position()
        self.assertAlmostEqual(x_before, x_after, delta=0.1)
        self.assertAlmostEqual(y_before, y_after, delta=0.1)

    def test_repeated_gps_converges(self):
        """Repeated GPS at same point within threshold should converge."""
        kf = FusionKF()
        for _ in range(20):
            kf.predict(ax=0, ay=0, gyro_z=0.0, dt=0.1)
            kf.update_gps(3.0, 4.0)
        x, y = kf.get_position()
        self.assertAlmostEqual(x, 3.0, delta=0.5)
        self.assertAlmostEqual(y, 4.0, delta=0.5)

    def test_covariance_decreases_after_gps(self):
        """P[0,0] should shrink after GPS update."""
        kf = FusionKF()
        kf.predict(ax=0, ay=0, gyro_z=0.0, dt=0.1)
        p_before = kf.P[0, 0]
        kf.update_gps(0.0, 0.0)
        self.assertLess(kf.P[0, 0], p_before)


# =============================================================================
# 5. GPS utils
# =============================================================================
class TestGPSUtils(unittest.TestCase):

    def test_datum_maps_to_origin(self):
        lat0, lon0 = 40.1164, -88.2434
        x, y = latlon_to_xy(lat0, lon0, lat0, lon0)
        self.assertAlmostEqual(x, 0.0, places=6)
        self.assertAlmostEqual(y, 0.0, places=6)

    def test_north_displacement(self):
        lat0, lon0 = 40.1164, -88.2434
        _, y = latlon_to_xy(lat0 + 0.001, lon0, lat0, lon0)
        self.assertAlmostEqual(y, 111.195, delta=0.5)

    def test_east_displacement_positive_x(self):
        lat0, lon0 = 40.1164, -88.2434
        x, _ = latlon_to_xy(lat0, lon0 + 0.001, lat0, lon0)
        self.assertGreater(x, 0.0)

    def test_euclidean_distance(self):
        self.assertAlmostEqual(euclidean_distance(0, 0, 3, 4), 5.0, places=6)


# =============================================================================
# 6. Full pipeline — straight drive with GPS corrections
# =============================================================================
class TestFullPipeline(unittest.TestCase):

    def test_straight_drive_east_with_gps(self):
        """
        Simulate 2 m/s east for 10 s with GPS corrections every 10 IMU ticks.
        Final position should be within 2 m of 20 m east.
        """
        kf  = FusionKF()
        dt  = 0.01
        gps_every = 10

        for i in range(1000):
            kf.predict(ax=0.0, ay=0.0, gyro_z=0.0, dt=dt)

            # Inject velocity directly via GPS-like position steps
            if i % gps_every == 0:
                true_x = 2.0 * (i * dt)
                kf.update_gps(true_x + np.random.normal(0, 0.3),
                              np.random.normal(0, 0.3))

        x, y = kf.get_position()
        self.assertAlmostEqual(x, 20.0, delta=2.0)
        self.assertAlmostEqual(y,  0.0, delta=2.0)

    def test_turning_drive(self):
        """
        Turning at 0.1 rad/s while moving. Just checks no crash and
        position is plausible (moved some distance, didn't teleport).
        """
        kf = FusionKF()
        for _ in range(500):
            kf.predict(ax=0.5, ay=0.0, gyro_z=0.1, dt=0.01)

        x, y  = kf.get_position()
        dist  = math.hypot(x, y)
        self.assertGreater(dist, 0.5)    # moved somewhere
        self.assertLess(dist, 100.0)     # didn't explode

    def test_yaw_consistent_through_gps_updates(self):
        """GPS updates should not affect yaw estimate."""
        kf = FusionKF()
        for _ in range(10):
            kf.predict(ax=0, ay=0, gyro_z=0.2, dt=0.01)
        yaw_before = kf.get_yaw()
        kf.update_gps(1.0, 0.0)
        self.assertAlmostEqual(kf.get_yaw(), yaw_before, delta=0.01)


# =============================================================================
if __name__ == '__main__':
    print('=' * 55)
    print(' FusionKF Unit Tests')
    print('=' * 55)
    unittest.main(verbosity=2)