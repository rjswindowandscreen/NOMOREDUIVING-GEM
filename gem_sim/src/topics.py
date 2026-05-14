#!/usr/bin/env python3
"""
topics.py — Single source of truth for every ROS2 topic name in the project.

Lives at gem_sim/src/topics.py — sibling of control/, localization/, perception/.

Convention
──────────
  SIM_<name>   topic name used in Gazebo simulation
  REAL_<name>  topic name used on the real GEM car
  <name>       topic is the same in both modes

Import in any script
─────────────────────
  import sys, os
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
  from topics import get_topics, ACKERMANN_CMD, ...

  t = get_topics('sim')   # or 'real'
  camera_topic = t['camera']
  odom_topic   = t['odom']
"""


# ── Topics that differ between sim and real ────────────────────────────────────

# Camera image feed
SIM_CAMERA  = '/camera/image_raw'               # Gazebo camera plugin
REAL_CAMERA = '/zed/zed_node/rgb/image_rect_color'  # ZED camera rectified RGB

# Vehicle odometry / pose
SIM_ODOM  = 'odom'                # Gazebo ground-truth odometry
REAL_ODOM = '/odometry/global'    # our localization node output (KF fused)


# ── Topics that are the same in both modes ─────────────────────────────────────

# Perception output (perception → control)
LANE_ERROR = '/lane_error'     # Float32MultiArray [XTE (m), HE (rad)]
OBSTACLES  = '/obstacles'      # Float32MultiArray [x, y, area, x, y, area, ...]
LANE_MASK  = '/lane_mask'      # Image — intermediate perception output

# Control output (control → sim/PACMod)
ACKERMANN_CMD = '/ackermann_cmd'   # AckermannDrive — sim eats it; pacmod_bridge translates for real

# Internal coordination
PARKING_DONE = '/parking_done'    # Bool — park_control publishes True when at lane entry


# ── Sensor topics (real car only — not present in sim) ────────────────────────

IMU       = '/imu'          # sensor_msgs/Imu        from Ouster or onboard IMU
NAVSATFIX = '/navsatfix'    # sensor_msgs/NavSatFix  from Septentrio GNSS
TWIST_INS = '/twist_ins'    # geometry_msgs/TwistWithCovarianceStamped from Septentrio INS


# ── PACMod topics (real car only) ─────────────────────────────────────────────

PACMOD_SPEED_RPT    = 'pacmod/vehicle_speed_rpt'  # in  — VehicleSpeedRpt
PACMOD_GLOBAL_RPT   = 'pacmod/global_rpt'          # in  — GlobalRpt
PACMOD_SHIFT_RPT    = 'pacmod/shift_rpt'           # in  — SystemRptInt
PACMOD_GLOBAL_CMD   = 'pacmod/global_cmd'          # out — GlobalCmd  (enable/disable)
PACMOD_SHIFT_CMD    = 'pacmod/shift_cmd'           # out — SystemCmdInt (gear)
PACMOD_ACCEL_CMD    = 'pacmod/accel_cmd'           # out — SystemCmdFloat (throttle 0–1)
PACMOD_BRAKE_CMD    = 'pacmod/brake_cmd'           # out — SystemCmdFloat (brake 0–1)
PACMOD_STEERING_CMD = 'pacmod/steering_cmd'        # out — PositionWithSpeed (steer wheel rad)


# ── Helper ─────────────────────────────────────────────────────────────────────

def get_topics(mode: str) -> dict:
    """
    Return a dict of topic names resolved for the given mode.

    Parameters
    ----------
    mode : 'sim' or 'real'

    Returns
    -------
    dict with keys:
        camera, odom,
        lane_error, obstacles, lane_mask,
        ackermann_cmd, parking_done,
        imu, navsatfix, twist_ins,
        pacmod_speed_rpt, pacmod_global_rpt, pacmod_shift_rpt,
        pacmod_global_cmd, pacmod_shift_cmd,
        pacmod_accel_cmd, pacmod_brake_cmd, pacmod_steering_cmd
    """
    if mode not in ('sim', 'real'):
        raise ValueError(f"mode must be 'sim' or 'real', got '{mode!r}'")

    return {
        # mode-dependent
        'camera':  SIM_CAMERA  if mode == 'sim' else REAL_CAMERA,
        'odom':    SIM_ODOM    if mode == 'sim' else REAL_ODOM,

        # same in both
        'lane_error':    LANE_ERROR,
        'obstacles':     OBSTACLES,
        'lane_mask':     LANE_MASK,
        'ackermann_cmd': ACKERMANN_CMD,
        'parking_done':  PARKING_DONE,

        # real-only sensors (safe to subscribe in sim — just never fires)
        'imu':      IMU,
        'navsatfix': NAVSATFIX,
        'twist_ins': TWIST_INS,

        # PACMod (real only)
        'pacmod_speed_rpt':    PACMOD_SPEED_RPT,
        'pacmod_global_rpt':   PACMOD_GLOBAL_RPT,
        'pacmod_shift_rpt':    PACMOD_SHIFT_RPT,
        'pacmod_global_cmd':   PACMOD_GLOBAL_CMD,
        'pacmod_shift_cmd':    PACMOD_SHIFT_CMD,
        'pacmod_accel_cmd':    PACMOD_ACCEL_CMD,
        'pacmod_brake_cmd':    PACMOD_BRAKE_CMD,
        'pacmod_steering_cmd': PACMOD_STEERING_CMD,
    }