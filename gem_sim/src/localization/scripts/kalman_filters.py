#!/usr/bin/env python3
"""
kalman_filters.py — Three decoupled Kalman Filters for GEM localization.

  YawKF   : 1-D KF for yaw + yaw_rate  (IMU gyro + VO yaw update)
  LocalKF : 2-D KF for body-frame vx, vy  (IMU accel + VO displacement)
             Also accumulates world-frame x, y by integrating with current yaw.
  GlobalKF: 2-D KF for world-frame x, y  (receives GPS position updates)
"""

import numpy as np


# ---------------------------------------------------------------------------
# YawKF
# ---------------------------------------------------------------------------
class YawKF:
    """
    State: [yaw, yaw_rate]
    Predict with IMU gyro_z.
    Update with VO-derived yaw estimate.
    """

    def __init__(self):
        self.x = np.zeros((2, 1))          # [yaw; yaw_rate]
        self.P = np.eye(2) * 0.1

        # Process noise — gyro drift
        self.Q = np.diag([1e-4, 1e-3])

        # Measurement noise for VO yaw update
        self.R_vo = np.array([[0.05]])

    # -- prediction ----------------------------------------------------------
    def predict(self, gyro_z: float, dt: float):
        """Propagate yaw using gyro angular velocity.
        
        gyro_z is a direct measurement of angular velocity, not an increment.
        So yaw_rate is set to gyro_z directly; yaw integrates yaw_rate*dt.
        """
        # A: yaw carries forward, yaw_rate is reset each step (overridden by B)
        A = np.array([[1.0, 0.0],
                      [0.0, 0.0]])
        B = np.array([[dt],
                      [1.0]])

        self.x = A @ self.x + B * gyro_z
        self.x[0, 0] = self._wrap(self.x[0, 0])
        self.P = A @ self.P @ A.T + self.Q

    # -- update --------------------------------------------------------------
    def update_vo(self, vo_yaw: float):
        """Correct yaw estimate using accumulated VO heading."""
        H = np.array([[1.0, 0.0]])
        z = np.array([[self._wrap(vo_yaw - self.x[0, 0])]])  # innovation (wrapped)

        S = H @ self.P @ H.T + self.R_vo
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ z
        self.x[0, 0] = self._wrap(self.x[0, 0])
        self.P = (np.eye(2) - K @ H) @ self.P

    # -- properties ----------------------------------------------------------
    @property
    def yaw(self) -> float:
        return float(self.x[0, 0])

    @property
    def yaw_rate(self) -> float:
        return float(self.x[1, 0])

    @staticmethod
    def _wrap(angle: float) -> float:
        """Wrap angle to [-π, π]."""
        import math
        return math.atan2(math.sin(angle), math.cos(angle))


# ---------------------------------------------------------------------------
# LocalKF
# ---------------------------------------------------------------------------
class LocalKF:
    """
    State: [vx_body, vy_body]  (body-frame velocities)
    Predict with IMU linear acceleration.
    Update with VO velocity / displacement.

    world_x, world_y are maintained by numerical integration (not KF state)
    using the current yaw from YawKF.
    """

    def __init__(self):
        self.x = np.zeros((2, 1))          # [vx_body; vy_body]
        self.P = np.eye(2) * 0.5

        # Process noise — IMU accel noise
        self.Q = np.diag([0.05, 0.05])

        # Measurement noise for VO velocity update
        self.R_vo = np.diag([0.1, 0.1])

        # World-frame position (integrated, not in KF state)
        self.world_x: float = 0.0
        self.world_y: float = 0.0

    # -- prediction ----------------------------------------------------------
    def predict(self, ax: float, ay: float, dt: float):
        """Propagate body-frame velocity using IMU linear acceleration."""
        A = np.eye(2)
        B = np.eye(2) * dt

        self.x = A @ self.x + B @ np.array([[ax], [ay]])
        self.P = A @ self.P @ A.T + self.Q

    def integrate_position(self, yaw: float, dt: float):
        """
        Rotate body-frame velocity into world frame and integrate position.
        Called after predict() every IMU cycle.
        """
        import math
        vx_b = float(self.x[0, 0])
        vy_b = float(self.x[1, 0])
        vx_w = math.cos(yaw) * vx_b - math.sin(yaw) * vy_b
        vy_w = math.sin(yaw) * vx_b + math.cos(yaw) * vy_b
        self.world_x += vx_w * dt
        self.world_y += vy_w * dt

    # -- update --------------------------------------------------------------
    def update_vo(self, dx: float, dy: float, dt: float):
        """Correct body-frame velocity using VO displacement delta."""
        if dt <= 0:
            return
        vx_meas = dx / dt
        vy_meas = dy / dt

        H = np.eye(2)
        z = np.array([[vx_meas], [vy_meas]])
        S = H @ self.P @ H.T + self.R_vo
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(2) - K @ H) @ self.P

    # -- properties ----------------------------------------------------------
    @property
    def speed(self) -> float:
        """Magnitude of body-frame velocity (used for VO scale)."""
        import math
        return math.hypot(float(self.x[0, 0]), float(self.x[1, 0]))


# ---------------------------------------------------------------------------
# GlobalKF
# ---------------------------------------------------------------------------
class GlobalKF:
    """
    State: [x_world, y_world, vx_world, vy_world]
    Predict with world-frame velocity from LocalKF.
    Update with GPS absolute position.
    """

    def __init__(self):
        self.x = np.zeros((4, 1))          # [x; y; vx; vy]
        self.P = np.diag([100.0, 100.0, 1.0, 1.0])  # high pos uncertainty → K≈1 on first GPS

        # Process noise — small model uncertainty
        self.Q = np.diag([0.01, 0.01, 0.1, 0.1])

        # GPS measurement noise (~2–5 m CEP for a typical GNSS receiver)
        self.R_gps = np.diag([1.0, 1.0])   # trust GPS (good GNSS receiver)

        # Reject GPS jumps larger than this many metres
        self.gps_reject_threshold: float = 8.0

    # -- prediction ----------------------------------------------------------
    def predict(self, vx_w: float, vy_w: float, dt: float):
        """Constant-velocity prediction using world-frame velocity from LocalKF."""
        A = np.array([[1, 0, dt,  0],
                      [0, 1,  0, dt],
                      [0, 0,  1,  0],
                      [0, 0,  0,  1]])

        # Override velocity states with measured world velocity
        self.x[2, 0] = vx_w
        self.x[3, 0] = vy_w
        self.x = A @ self.x
        self.P = A @ self.P @ A.T + self.Q

    # -- update --------------------------------------------------------------
    def update_gps(self, gps_x: float, gps_y: float):
        """Correct position using GPS fix."""
        import math
        # Outlier rejection
        dx = gps_x - float(self.x[0, 0])
        dy = gps_y - float(self.x[1, 0])
        if math.hypot(dx, dy) > self.gps_reject_threshold:
            return

        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        z = np.array([[gps_x], [gps_y]])
        S = H @ self.P @ H.T + self.R_gps
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P

    # -- properties ----------------------------------------------------------
    @property
    def position(self):
        """Returns (x, y) world-frame position."""
        return float(self.x[0, 0]), float(self.x[1, 0])