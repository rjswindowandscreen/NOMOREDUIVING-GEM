#!/usr/bin/env python3
"""
pacmod_bridge.py — Translate /ackermann_cmd into pacmod2 commands.

Real-vehicle only. In sim, the topics it publishes have no listeners (harmless).

State machine:
    WAIT_FOR_GLOBAL → SHIFT_TO_DRIVE → ENABLE → RUNNING ↔ OVERRIDDEN

At RUNNING, every 33 ms the bridge publishes:
  - pacmod/global_cmd   (enable=true)
  - pacmod/shift_cmd    (SHIFT_FORWARD)
  - pacmod/accel_cmd    (throttle command, 0..MAX_THROTTLE)
  - pacmod/brake_cmd    (brake command, 0..MAX_BRAKE)
  - pacmod/steering_cmd (column angle = road_wheel * STEERING_RATIO)

If global_rpt.override_active goes true → drop to OVERRIDDEN: publish zeros.
"""

import enum
import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDrive
from pacmod2_msgs.msg import (
    GlobalCmd, GlobalRpt,
    SystemCmdInt, SystemCmdFloat, SystemRptInt,
    PositionWithSpeed, VehicleSpeedRpt,
)


# ── CONFIG ────────────────────────────────────────────────────────────────────

HEARTBEAT_HZ        = 30.0

# Steering ratio (column → road wheel). PLACEHOLDER — verify with team.
STEERING_RATIO      = 17.0
STEERING_RATE_LIMIT = 1.5      # rad/s at the column

# Throttle / brake split
SPEED_DEADBAND = 0.05          # m/s
THROTTLE_KP    = 0.5
BRAKE_KP       = 0.6
CREEP_ACCEL    = 0.21          # baseline accel command (from game_control)
MAX_THROTTLE   = 0.35
MAX_BRAKE      = 0.4

# pacmod2_msgs/SystemCmdInt constants
SHIFT_PARK    = 0
SHIFT_FORWARD = 3


class State(enum.Enum):
    WAIT_FOR_GLOBAL = 0
    SHIFT_TO_DRIVE  = 1
    ENABLE          = 2
    RUNNING         = 3
    OVERRIDDEN      = 4


class PacmodBridge(Node):
    def __init__(self):
        super().__init__('pacmod_bridge')

        self.state = State.WAIT_FOR_GLOBAL

        # Latest /ackermann_cmd
        self.cmd_speed = 0.0
        self.cmd_steer = 0.0

        # Reports
        self.measured_speed      = 0.0
        self.global_rpt_enabled  = False
        self.global_rpt_override = False
        self.shift_rpt_output    = None

        # Subs
        self.create_subscription(
            AckermannDrive,   '/ackermann_cmd',           self._ackermann_cb, 10)
        self.create_subscription(
            VehicleSpeedRpt,  'pacmod/vehicle_speed_rpt', self._speed_cb,     10)
        self.create_subscription(
            GlobalRpt,        'pacmod/global_rpt',        self._global_cb,    10)
        self.create_subscription(
            SystemRptInt,     'pacmod/shift_rpt',         self._shift_cb,     10)

        # Pubs
        self.global_cmd_pub = self.create_publisher(GlobalCmd,          'pacmod/global_cmd',   10)
        self.shift_cmd_pub  = self.create_publisher(SystemCmdInt,       'pacmod/shift_cmd',    10)
        self.accel_cmd_pub  = self.create_publisher(SystemCmdFloat,     'pacmod/accel_cmd',    10)
        self.brake_cmd_pub  = self.create_publisher(SystemCmdFloat,     'pacmod/brake_cmd',    10)
        self.steer_cmd_pub  = self.create_publisher(PositionWithSpeed,  'pacmod/steering_cmd', 10)

        # Heartbeat
        self.create_timer(1.0 / HEARTBEAT_HZ, self._heartbeat)
        self.get_logger().info(f'pacmod_bridge up, heartbeat {HEARTBEAT_HZ:.0f} Hz')

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _ackermann_cb(self, msg):
        self.cmd_speed = msg.speed
        self.cmd_steer = msg.steering_angle

    def _speed_cb(self, msg):
        if msg.vehicle_speed_valid:
            self.measured_speed = msg.vehicle_speed

    def _global_cb(self, msg):
        self.global_rpt_enabled  = msg.enabled
        self.global_rpt_override = msg.override_active

        if self.state == State.WAIT_FOR_GLOBAL:
            self.state = State.SHIFT_TO_DRIVE
            self.get_logger().info('global_rpt seen → SHIFT_TO_DRIVE')

        # Latch override
        if msg.override_active and self.state == State.RUNNING:
            self.state = State.OVERRIDDEN
            self.get_logger().warn('OVERRIDE active — disengaging autonomy')

    def _shift_cb(self, msg):
        self.shift_rpt_output = msg.output

    # ── Heartbeat ────────────────────────────────────────────────────────────

    def _heartbeat(self):
        now = self.get_clock().now().to_msg()

        # State transitions
        if self.state == State.SHIFT_TO_DRIVE and self.shift_rpt_output == SHIFT_FORWARD:
            self.state = State.ENABLE
            self.get_logger().info('shift confirmed → ENABLE')
        elif self.state == State.ENABLE and self.global_rpt_enabled:
            self.state = State.RUNNING
            self.get_logger().info('pacmod enabled → RUNNING')

        # global_cmd — always
        g = GlobalCmd()
        g.header.stamp     = now
        g.enable           = self.state in (State.ENABLE, State.RUNNING)
        g.clear_override   = False
        g.ignore_override  = False
        self.global_cmd_pub.publish(g)

        # shift_cmd — always
        s = SystemCmdInt()
        s.header.stamp = now
        s.command = SHIFT_FORWARD if self.state in (
            State.SHIFT_TO_DRIVE, State.ENABLE, State.RUNNING
        ) else SHIFT_PARK
        self.shift_cmd_pub.publish(s)

        # accel / brake / steer — zero unless RUNNING
        accel = brake = 0.0
        steer_pos = 0.0
        if self.state == State.RUNNING:
            accel, brake = self._split_throttle_brake(self.cmd_speed, self.measured_speed)
            steer_pos    = self.cmd_steer * STEERING_RATIO

        a = SystemCmdFloat(); a.header.stamp = now; a.command = float(accel)
        b = SystemCmdFloat(); b.header.stamp = now; b.command = float(brake)
        self.accel_cmd_pub.publish(a)
        self.brake_cmd_pub.publish(b)

        st = PositionWithSpeed()
        st.header.stamp           = now
        st.angular_position       = float(steer_pos)
        st.angular_velocity_limit = float(STEERING_RATE_LIMIT)
        self.steer_cmd_pub.publish(st)

    @staticmethod
    def _split_throttle_brake(target_v, measured_v):
        err = target_v - measured_v
        if err >  SPEED_DEADBAND:
            return min(CREEP_ACCEL + THROTTLE_KP * err, MAX_THROTTLE), 0.0
        if err < -SPEED_DEADBAND:
            return 0.0, min(-BRAKE_KP * err, MAX_BRAKE)
        return CREEP_ACCEL, 0.0


def main():
    rclpy.init()
    node = PacmodBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
