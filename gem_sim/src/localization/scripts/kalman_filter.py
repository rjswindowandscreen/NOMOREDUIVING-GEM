#!/usr/bin/env python3
"""
kalman_filter.py — Kalman Filter fusing IMU, GPS, and INS velocity.

State: [x, y, vx, vy, yaw, yaw_rate]

Predict   : IMU accelerometer + gyro         (~50 Hz)
update_gps: GPS position from /navsatfix     (~40 Hz)
update_vel: INS velocity from /twist_ins     (~40 Hz)
"""

import math
import numpy as np


class FusionKF:

    def __init__(self):

        # state vector: [x, y, vx, vy, yaw, yaw_rate]
        self.x = np.zeros((6, 1))

        # covariance — high initial position uncertainty so first GPS snaps into place
        self.P = np.diag([100.0, 100.0, 1.0, 1.0, 0.1, 0.1])

        # process noise — how much we trust the IMU model per step
        self.Q = np.diag([0.01, 0.01, 0.5, 0.5, 0.001, 0.01])

        # GPS position measurement noise (metres^2)
        self.R_gps = np.diag([1.0, 1.0])

        # reject GPS fixes that jump more than this (multipath / bad fix)
        self.gps_reject_threshold = 8.0


    def predict(self, ax, ay, gyro_z, dt):
        """
        Predict step — called every IMU message (~50 Hz).

        ax, ay   : body-frame acceleration in m/s^2
                   pass in bias-corrected values from localization_node
        gyro_z   : bias-corrected yaw rate from gyro in rad/s
        dt       : time since last predict in seconds
        """

        # rotate body-frame acceleration into world frame using current yaw
        # done here so the matrices below stay linear
        yaw = float(self.x[4, 0])
        ax_world = math.cos(yaw) * ax - math.sin(yaw) * ay
        ay_world = math.sin(yaw) * ax + math.cos(yaw) * ay

        # state transition: position integrates velocity, yaw integrates yaw_rate
        # yaw_rate row is zeroed — set directly from gyro via B @ u below
        A = np.array([
            [1, 0, dt, 0,  0,  0],
            [0, 1,  0, dt, 0,  0],
            [0, 0,  1,  0, 0,  0],
            [0, 0,  0,  1, 0,  0],
            [0, 0,  0,  0, 1, dt],
            [0, 0,  0,  0, 0,  0],
        ])

        # control input: [ax_world, ay_world, gyro_z] → state
        B = np.array([
            [ 0,   0,  0],
            [ 0,   0,  0],
            [dt,   0,  0],
            [ 0,  dt,  0],
            [ 0,   0,  0],
            [ 0,   0,  1],
        ])

        u = np.array([[ax_world], [ay_world], [gyro_z]])

        self.x = A @ self.x + B @ u
        self.P = A @ self.P @ A.T + self.Q

        # keep yaw in [-pi, pi]
        self.x[4, 0] = math.atan2(math.sin(self.x[4, 0]), math.cos(self.x[4, 0]))


    def update_gps(self, gps_x, gps_y):
        """
        Update step — GPS position from /navsatfix (~40 Hz).

        gps_x, gps_y : position in metres relative to datum, from latlon_to_xy()
        """

        dx = gps_x - float(self.x[0, 0])
        dy = gps_y - float(self.x[1, 0])
        if math.hypot(dx, dy) > self.gps_reject_threshold:
            return

        # H selects [x, y] from state
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ])

        z = np.array([[gps_x], [gps_y]])
        S = H @ self.P @ H.T + self.R_gps
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(6) - K @ H) @ self.P

        self.x[4, 0] = math.atan2(math.sin(self.x[4, 0]), math.cos(self.x[4, 0]))


    def update_velocity(self, vx, vy, cov_vx=0.1, cov_vy=0.1):
        """
        Update step — INS velocity from /twist_ins (~40 Hz).

        Also used for ZUPT: pass vx=0, vy=0 with tight covariance when
        the car is confirmed stationary to actively correct velocity drift.

        vx, vy       : world-frame velocity in m/s (ENU frame, x=east y=north)
        cov_vx/cov_vy: velocity measurement variance — use tight values (0.01)
                       for ZUPT, looser values (from message covariance) when moving
        """

        # clamp covariance to a sensible minimum so we don't over-trust
        cov_vx = max(cov_vx, 0.001)
        cov_vy = max(cov_vy, 0.001)

        # H selects [vx, vy] from state
        H = np.array([
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ])

        z = np.array([[vx], [vy]])
        R = np.diag([cov_vx, cov_vy])

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(6) - K @ H) @ self.P

        self.x[4, 0] = math.atan2(math.sin(self.x[4, 0]), math.cos(self.x[4, 0]))


    def get_position(self):
        return float(self.x[0, 0]), float(self.x[1, 0])

    def get_velocity(self):
        return float(self.x[2, 0]), float(self.x[3, 0])

    def get_yaw(self):
        return float(self.x[4, 0])

    def get_yaw_rate(self):
        return float(self.x[5, 0])

    def get_speed(self):
        vx, vy = self.get_velocity()
        return math.hypot(vx, vy)