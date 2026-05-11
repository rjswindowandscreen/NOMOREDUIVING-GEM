"""
test_stanley_sim.py — Standalone Stanley + slow-speed simulator.  NO ROS.

Drop on Windows, install numpy + matplotlib, then:
    python test_stanley_sim.py

What it does:
  - Models a kinematic bicycle (wheelbase 1.75 m, like a GEM).
  - Runs your Stanley lateral controller against a known synthetic lane.
  - Runs your rate-limited longitudinal controller (slow speed + obstacle stop).
  - Plots trajectory, XTE, speed, and steering over time for each scenario.

When you're happy with the behaviour here, lift `stanley()`, `rate_limit()`,
and `update_speed()` into a `stanley_math.py` module and import them from your
ROS node — they take only floats, no rclpy.

Sign conventions used here (STANDARD Stanley):
  XTE > 0  → reference path is to the LEFT of the vehicle  → steer LEFT
  HE  > 0  → vehicle heading is to the RIGHT of path tangent → steer LEFT
  steer > 0 → road wheels point LEFT (CCW yaw, ROS right-hand-rule)

>>> WARNING <<<
Your perception node's /lane_error sign convention may differ from this.
Verify on a straight section before trusting the controller on the real car:
  drift the car slightly RIGHT of the lane; check whether the published XTE
  is POSITIVE or NEGATIVE.  If your controller steers AWAY from the lane,
  negate xte (and possibly he) before passing to stanley().
"""

import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Import the same math functions the ROS controller uses, so this sim and
# the real controller can never disagree.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))
from stanley_math import stanley, rate_limit, update_speed  # noqa: E402, F401


# ─────────────────────────────────────────────────────────────────────────────
# Kinematic bicycle model (rear-axle reference, front-axle steering)
# ─────────────────────────────────────────────────────────────────────────────

class Bicycle:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, L=1.75):
        self.x, self.y, self.yaw, self.v, self.L = x, y, yaw, v, L

    def step(self, steer: float, v_cmd: float, dt: float):
        # In the real car, speed tracking has its own dynamics; here v_cmd is
        # already rate-limited upstream, so we let velocity follow it directly.
        self.v = v_cmd
        self.x   += self.v * math.cos(self.yaw) * dt
        self.y   += self.v * math.sin(self.yaw) * dt
        self.yaw += (self.v / self.L) * math.tan(steer) * dt
        self.yaw  = math.atan2(math.sin(self.yaw), math.cos(self.yaw))


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic "perception" — produces XTE/HE from a known reference path.
# In the real system, this is replaced by your /lane_error subscription.
# ─────────────────────────────────────────────────────────────────────────────

def make_lane(kind: str, n: int = 600) -> np.ndarray:
    """Return reference path as (n, 2) array of (x, y) in metres."""
    s = np.linspace(0, 50, n)
    if kind == 'straight':
        return np.column_stack([s, np.zeros_like(s)])
    if kind == 'sine':
        return np.column_stack([s, 2.0 * np.sin(s / 5.0)])
    if kind == 'arc':
        R = 20.0
        theta = s / R
        return np.column_stack([R * np.sin(theta), R * (1 - np.cos(theta))])
    raise ValueError(kind)


def perceive(car: Bicycle, lane: np.ndarray):
    """Compute (XTE, HE) at the FRONT AXLE relative to the lane polyline."""
    # Stanley operates on front-axle position, not the rear-axle (vehicle origin).
    fx = car.x + car.L * math.cos(car.yaw)
    fy = car.y + car.L * math.sin(car.yaw)

    # Nearest polyline vertex (good enough for dense paths)
    d = np.hypot(lane[:, 0] - fx, lane[:, 1] - fy)
    i = int(np.argmin(d))
    j = min(i + 1, len(lane) - 1)

    # Path tangent direction
    tx, ty = lane[j] - lane[i]
    if tx == 0 and ty == 0:
        tx, ty = lane[i] - lane[max(i - 1, 0)]
    norm = math.hypot(tx, ty)
    tx, ty = tx / norm, ty / norm

    # XTE: signed perpendicular distance.  +ve = lane is LEFT of vehicle.
    px, py = fx - lane[i, 0], fy - lane[i, 1]
    cross = tx * py - ty * px              # >0 → car LEFT of path → lane RIGHT of car
    xte = -cross                            # flip to Stanley convention

    # HE: shortest signed angle from car heading to path tangent
    path_yaw = math.atan2(ty, tx)
    he = math.atan2(math.sin(path_yaw - car.yaw),
                    math.cos(path_yaw - car.yaw))
    return xte, he


# ─────────────────────────────────────────────────────────────────────────────
# Scenarios
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = [
    dict(name='straight lane, start +1.5 m offset',
         lane='straight', start=(0.0,  1.5,  0.0)),
    dict(name='straight lane, heading 20 deg off',
         lane='straight', start=(0.0,  0.0,  math.radians(20))),
    dict(name='sine wave lane',
         lane='sine',     start=(0.0,  0.0,  0.0)),
    dict(name='gentle left arc (R=20 m)',
         lane='arc',      start=(0.0, -1.0,  0.0)),
    dict(name='straight + obstacle at x = 12 m',
         lane='straight', start=(0.0,  0.0,  0.0), obstacle_at=12.0),
]


# ─────────────────────────────────────────────────────────────────────────────
# Tunables — match what you plan to use on the real vehicle
# ─────────────────────────────────────────────────────────────────────────────

CFG = dict(
    dt          = 0.02,    # 50 Hz control loop
    duration    = 30.0,    # seconds simulated
    # Speed
    max_speed   = 0.8,     # m/s — walking pace
    max_accel   = 0.2,     # m/s² — acceleration limit
    stop_decel  = 0.6,     # m/s² — harder decel when obstacle blocks
    # Stanley
    stanley_k   = 0.3,
    max_steer   = 0.61,    # rad at the road wheel
    # Obstacle thresholds (purely longitudinal — Stanley does not avoid laterally)
    stop_dist   = 5.0,
    slow_dist   = 9.0,
)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation loop
# ─────────────────────────────────────────────────────────────────────────────

def run(scenario: dict, cfg: dict = CFG):
    lane = make_lane(scenario['lane'])
    car  = Bicycle(*scenario['start'])
    obs_x = scenario.get('obstacle_at')

    log = {k: [] for k in ('t', 'x', 'y', 'yaw', 'v', 'steer', 'xte', 'he')}
    steps = int(cfg['duration'] / cfg['dt'])

    for k in range(steps):
        t = k * cfg['dt']
        xte, he = perceive(car, lane)

        # Obstacle in front of car (along +x for these scenarios)?
        obs_blocked = obs_near = False
        if obs_x is not None:
            d = obs_x - car.x
            if   0 < d < cfg['stop_dist']: obs_blocked = True
            elif 0 < d < cfg['slow_dist']: obs_near    = True

        v_cmd = update_speed(car.v, cfg['dt'],
                             max_speed=cfg['max_speed'],
                             max_accel=cfg['max_accel'],
                             stop_decel=cfg['stop_decel'],
                             obs_blocked=obs_blocked,
                             obs_near=obs_near)
        s_cmd = stanley(xte, he, car.v,
                        k=cfg['stanley_k'], max_steer=cfg['max_steer'])

        car.step(s_cmd, v_cmd, cfg['dt'])

        log['t'].append(t)
        log['x'].append(car.x);   log['y'].append(car.y)
        log['yaw'].append(car.yaw); log['v'].append(car.v)
        log['steer'].append(s_cmd); log['xte'].append(xte); log['he'].append(he)

    return lane, log, obs_x


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot(scenario, lane, log, obs_x):
    fig = plt.figure(figsize=(13, 8))
    gs  = fig.add_gridspec(2, 3, height_ratios=[1.6, 1], hspace=0.4, wspace=0.32)
    ax_xy = fig.add_subplot(gs[0, :])
    ax_x  = fig.add_subplot(gs[1, 0])
    ax_v  = fig.add_subplot(gs[1, 1])
    ax_s  = fig.add_subplot(gs[1, 2])

    fig.suptitle(scenario['name'], fontsize=12)

    ax_xy.plot(lane[:, 0], lane[:, 1], 'k--', lw=1,  label='lane (reference)')
    ax_xy.plot(log['x'],   log['y'],   'b-',  lw=1.5, label='car trajectory')
    ax_xy.plot(log['x'][0], log['y'][0], 'go', ms=8, label='start')
    ax_xy.plot(log['x'][-1], log['y'][-1], 'b*', ms=10, label='end')
    if obs_x is not None:
        ax_xy.plot(obs_x, 0, 'rx', ms=12, mew=3, label='obstacle')
    ax_xy.set_aspect('equal'); ax_xy.grid(alpha=0.3); ax_xy.legend(loc='upper right')
    ax_xy.set_xlabel('x [m]'); ax_xy.set_ylabel('y [m]')

    ax_x.plot(log['t'], log['xte'], 'r-')
    ax_x.axhline(0, color='gray', lw=0.5)
    ax_x.set_xlabel('t [s]'); ax_x.set_ylabel('XTE [m]'); ax_x.grid(alpha=0.3)
    ax_x.set_title('cross-track error')

    ax_v.plot(log['t'], log['v'], 'g-')
    ax_v.axhline(CFG['max_speed'], color='gray', lw=0.5, ls='--')
    ax_v.set_xlabel('t [s]'); ax_v.set_ylabel('speed [m/s]'); ax_v.grid(alpha=0.3)
    ax_v.set_title('vehicle speed')

    ax_s.plot(log['t'], np.degrees(log['steer']), 'm-')
    ax_s.axhline(0, color='gray', lw=0.5)
    ax_s.set_xlabel('t [s]'); ax_s.set_ylabel('steer [deg]'); ax_s.grid(alpha=0.3)
    ax_s.set_title('steering command')


def main():
    for sc in SCENARIOS:
        lane, log, obs_x = run(sc)
        plot(sc, lane, log, obs_x)
        # Quick summary line per scenario
        xte = np.array(log['xte'])
        print(f"[{sc['name']:42s}]  "
              f"final XTE = {xte[-1]:+.3f} m,  "
              f"|XTE|_max = {np.max(np.abs(xte)):.3f} m,  "
              f"final speed = {log['v'][-1]:.2f} m/s")
    plt.show()


if __name__ == '__main__':
    main()
