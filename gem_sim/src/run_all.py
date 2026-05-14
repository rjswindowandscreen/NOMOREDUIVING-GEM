#!/usr/bin/env python3
"""
run_all.py — Master launcher for the GEM autonomous system.

Lives at gem_sim/src/run_all.py  (sibling of control/, perception/, localization/)

Sequence
────────
Phase 1: Park-to-drive
    RRT drives from current position to the lane entry point.
    park_control.py publishes /parking_done (Bool True) when finished.

Phase 2: Lane following
    Stanley tracks the lane via perception until the GPS goal is reached
    (real) or the process is stopped (sim).

Usage
─────
    python3 src/run_all.py --mode sim
    python3 src/run_all.py --mode real

All goals are hardcoded as clearly labelled variables below — edit them
directly when you have real coordinates from the rosbag.
"""

import os
import signal
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

# ── Paths ──────────────────────────────────────────────────────────────────────
THIS_DIR         = os.path.dirname(os.path.abspath(__file__))
CTRL_SCRIPTS_DIR = os.path.join(THIS_DIR, 'control',     'scripts')
LOC_SCRIPT       = os.path.join(THIS_DIR, 'localization', 'run_localization.py')

# Import topics — gives us PARKING_DONE for the monitor node
sys.path.insert(0, THIS_DIR)
from topics import PARKING_DONE

# ══ HARDCODED GOALS — edit these ══════════════════════════════════════════════
#
# Park-to-drive RRT goal (lane entry point, local ENU frame).
# Replace with real coordinates extracted from your rosbag.
PARK_GOAL_X   = -20.0   # metres east  from starting datum
PARK_GOAL_Y   =   0.0   # metres north from starting datum
PARK_GOAL_YAW =   0.0   # heading (radians, 0 = east)

# Lane-following GPS destination (real mode only — ignored in sim).
# Replace with the actual GPS coordinates of your route endpoint.
GPS_GOAL_LAT    =  40.1164
GPS_GOAL_LON    = -88.2434
GPS_GOAL_RADIUS =   2.0     # metres — stop when within this distance

# ══════════════════════════════════════════════════════════════════════════════


# ── Parking monitor ────────────────────────────────────────────────────────────

class ParkingMonitor(Node):
    """Minimal node that watches /parking_done and sets a flag."""

    def __init__(self):
        super().__init__('run_all_monitor')
        self.parking_done = False
        self.create_subscription(Bool, PARKING_DONE, self._cb, 10)

    def _cb(self, msg):
        if msg.data:
            self.parking_done = True
            self.get_logger().info(
                'Parking complete — transitioning to lane following.'
            )


# ── Subprocess helpers ─────────────────────────────────────────────────────────

def _kill(proc, name):
    if proc is not None and proc.poll() is None:
        print(f'[run_all] Stopping {name}...')
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='GEM autonomous system launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Edit PARK_GOAL_X/Y/YAW and GPS_GOAL_LAT/LON at the top of this\n'
            'file before running on the real car.'
        ),
    )
    parser.add_argument(
        '--mode', choices=['sim', 'real'], default='sim',
        help=(
            'sim  — Gazebo odom, no GPS stop, no localization, no PACMod.\n'
            'real — /odometry/global, GPS stop, localization + PACMod bridge.'
        ),
    )
    args = parser.parse_args()
    mode = args.mode

    print()
    print('╔══════════════════════════════════════════╗')
    print('║         GEM Autonomous System            ║')
    print('╚══════════════════════════════════════════╝')
    print(f'  mode        : {mode}')
    print(f'  RRT goal    : ({PARK_GOAL_X:.2f}, {PARK_GOAL_Y:.2f}, {PARK_GOAL_YAW:.2f} rad)')
    if mode == 'real':
        print(f'  GPS goal    : ({GPS_GOAL_LAT:.6f}, {GPS_GOAL_LON:.6f})  ±{GPS_GOAL_RADIUS:.1f} m')
    else:
        print('  GPS goal    : N/A — sim mode runs until Ctrl+C')
    print()

    # ── ROS2 monitor ──────────────────────────────────────────────────────
    rclpy.init()
    monitor = ParkingMonitor()

    procs = {}

    def shutdown_all():
        for name, p in list(procs.items()):
            _kill(p, name)
        try:
            monitor.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()

    def on_signal(sig, frame):
        print('\n[run_all] Interrupted — shutting down all processes.')
        shutdown_all()
        sys.exit(0)

    signal.signal(signal.SIGINT,  on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    # ── Real-only infrastructure: PACMod bridge + localization ────────────
    if mode == 'real':
        print('[run_all] Starting pacmod_bridge...')
        procs['pacmod_bridge'] = subprocess.Popen(
            [sys.executable, os.path.join(CTRL_SCRIPTS_DIR, 'pacmod_bridge.py')]
        )
        print('[run_all] Starting localization...')
        procs['localization'] = subprocess.Popen(
            [sys.executable, LOC_SCRIPT]
        )
        time.sleep(1.5)   # let both initialise before commands flow

    # ── Perception ────────────────────────────────────────────────────────
    # sim  → perception/scripts/lane_detect.py  (Gazebo camera, sim model)
    # real → perception_real/run_perception.py  (real car model + camera)
    if mode == 'sim':
        perc_cmd = [
            sys.executable,
            os.path.join(THIS_DIR, 'perception', 'scripts', 'lane_detect.py'),
            '--ros-args',
            '-p', f'mode:={mode}',
        ]
        perc_cwd = os.path.join(THIS_DIR, 'perception')
        print('[run_all] Starting perception (sim)...')
    else:
        perc_cmd = [
            sys.executable,
            os.path.join(THIS_DIR, 'perception_real', 'run_perception.py'),
        ]
        perc_cwd = os.path.join(THIS_DIR, 'perception_real')
        print('[run_all] Starting perception (real)...')

    procs['perception'] = subprocess.Popen(perc_cmd, cwd=perc_cwd)
    time.sleep(1.0)   # let perception warm up before control starts

    # ══ Phase 1: Park-to-drive ═════════════════════════════════════════════
    print('[run_all] ══ Phase 1: Park-to-drive ══')
    procs['park_control'] = subprocess.Popen([
        sys.executable,
        os.path.join(CTRL_SCRIPTS_DIR, 'park_control.py'),
        '--ros-args',
        '-p', f'mode:={mode}',
        '-p', f'goal_x:={PARK_GOAL_X}',
        '-p', f'goal_y:={PARK_GOAL_Y}',
        '-p', f'goal_yaw:={PARK_GOAL_YAW}',
        '-p', 'drive_speed:=0.8',
    ])
    print('[run_all] Waiting for /parking_done...')

    while not monitor.parking_done:
        rclpy.spin_once(monitor, timeout_sec=0.1)
        if procs['park_control'].poll() is not None:
            rc = procs['park_control'].returncode
            print(f'[run_all] ERROR: park_control exited unexpectedly (rc={rc}).')
            shutdown_all()
            sys.exit(1)

    print('[run_all] Parking done — settling 0.5 s...')
    time.sleep(0.5)
    _kill(procs.pop('park_control'), 'park_control')

    # ══ Phase 2: Lane following ════════════════════════════════════════════
    print('\n[run_all] ══ Phase 2: Lane following ══')
    procs['stanley_controller'] = subprocess.Popen([
        sys.executable,
        os.path.join(CTRL_SCRIPTS_DIR, 'stanley_controller.py'),
        '--ros-args',
        '-p', f'mode:={mode}',
        '-p', f'goal_lat:={GPS_GOAL_LAT}',
        '-p', f'goal_lon:={GPS_GOAL_LON}',
        '-p', f'goal_radius:={GPS_GOAL_RADIUS}',
    ])

    if mode == 'sim':
        print('[run_all] Sim — press Ctrl+C to stop.')
    else:
        print('[run_all] Real — stops automatically at GPS goal.')

    try:
        procs['stanley_controller'].wait()
        print('\n[run_all] Lane following complete.')
    except KeyboardInterrupt:
        pass

    shutdown_all()
    print('[run_all] All done.')


if __name__ == '__main__':
    main()