#!/usr/bin/env python3
"""
gps_utils.py — Coordinate conversion and geometry helpers.
"""

import math
from geometry_msgs.msg import Quaternion

_R_EARTH = 6_371_000.0   # WGS-84 mean Earth radius in metres


def latlon_to_xy(lat: float, lon: float, lat0: float, lon0: float):
    """
    Convert (lat, lon) to local Cartesian (x, y) in metres
    relative to datum (lat0, lon0).

    x = east, y = north
    Accurate to < 1 m within ~1 km — sufficient for GEM campus route.
    """
    d_lat   = math.radians(lat - lat0)
    d_lon   = math.radians(lon - lon0)
    lat_mid = math.radians((lat + lat0) / 2.0)

    x = _R_EARTH * d_lon * math.cos(lat_mid)
    y = _R_EARTH * d_lat
    return x, y


def euler_to_quat(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> Quaternion:
    """Convert yaw/pitch/roll (radians) to a ROS Quaternion."""
    cy = math.cos(yaw   * 0.5)
    sy = math.sin(yaw   * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll  * 0.5)
    sr = math.sin(roll  * 0.5)

    q   = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)