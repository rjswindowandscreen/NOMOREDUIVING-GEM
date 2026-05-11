"""
stanley_math.py — Pure controller math. No ROS, no numpy, no matplotlib.

Importable from both the ROS controller (stanley_controller.py) and the
standalone Windows test sim (test_stanley_sim.py).

Sign conventions (standard Stanley):
    xte > 0  → reference path is to the LEFT of vehicle  → steer LEFT (+)
    he  > 0  → vehicle heading is to the RIGHT of path tangent → steer LEFT (+)
"""

import math


def stanley(xte: float, he: float, speed: float,
            k: float = 0.3, max_steer: float = 0.61) -> float:
    """Stanley lateral controller. Returns road-wheel steer angle in radians."""
    v_safe = max(speed, 0.5)        # avoid div-by-zero and huge gains at standstill
    s = he + math.atan(k * xte / v_safe)
    return max(-max_steer, min(max_steer, s))


def rate_limit(target: float, current: float, max_rate: float, dt: float) -> float:
    """Clamp |target - current| / dt to max_rate. Used for accel/decel limiting."""
    return max(current - max_rate * dt, min(current + max_rate * dt, target))


def update_speed(curr_v: float, dt: float, *,
                 max_speed: float, max_accel: float, stop_decel: float,
                 obs_blocked: bool = False, obs_near: bool = False,
                 near_goal: bool = False, lane_stale: bool = False) -> float:
    """Pick a target speed from stop conditions, then rate-limit toward it."""
    if lane_stale or near_goal or obs_blocked:
        return rate_limit(0.0,             curr_v, stop_decel, dt)
    if obs_near:
        return rate_limit(max_speed * 0.5, curr_v, max_accel,  dt)
    return     rate_limit(max_speed,       curr_v, max_accel,  dt)
