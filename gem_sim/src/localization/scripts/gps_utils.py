#!/usr/bin/env python3
"""
gps_utils.py — Coordinate conversion and geometry helpers.
"""

import math
from geometry_msgs.msg import Quaternion


# Earth radius in metres (WGS-84 mean)
_R_EARTH = 6_371_000.0


def latlon_to_xy(lat: float, lon: float, lat0: float, lon0: float):
    """
    Convert (lat, lon) to a local Cartesian (x, y) in metres
    relative to datum (lat0, lon0).

    Uses equirectangular approximation — accurate to <1 m within ~1 km radius,
    which is more than sufficient for the GEM campus route.

    Returns
    -------
    x : east  (metres)
    y : north (metres)
    """
    d_lat = math.radians(lat - lat0)
    d_lon = math.radians(lon - lon0)
    lat_mid = math.radians((lat + lat0) / 2.0)

    x = _R_EARTH * d_lon * math.cos(lat_mid)   # east
    y = _R_EARTH * d_lat                         # north
    return x, y


def xy_to_latlon(x: float, y: float, lat0: float, lon0: float):
    """
    Inverse of latlon_to_xy. Converts local (x, y) back to (lat, lon).
    Useful for debugging / visualisation.
    """
    lat_mid = math.radians(lat0)
    d_lat = y / _R_EARTH
    d_lon = x / (_R_EARTH * math.cos(lat_mid))

    lat = lat0 + math.degrees(d_lat)
    lon = lon0 + math.degrees(d_lon)
    return lat, lon


def euler_to_quat(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> Quaternion:
    """
    Convert Euler angles (yaw, pitch, roll) in radians to a ROS Quaternion.
    GEM is a ground vehicle so pitch and roll are assumed zero by default.
    """
    cy = math.cos(yaw   * 0.5)
    sy = math.sin(yaw   * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll  * 0.5)
    sr = math.sin(roll  * 0.5)

    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """2-D Euclidean distance between two world-frame points."""
    return math.hypot(x2 - x1, y2 - y1)