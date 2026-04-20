#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt


def euler_to_quaternion(euler):
    roll, pitch, yaw = euler
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [x, y, z, w]


def quaternion_to_euler(quaternion):
    x, y, z, w = quaternion
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.copysign(np.pi / 2, sinp) if abs(sinp) >= 1 else np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return [roll, pitch, yaw]


def visualization(node, cur_time, speed=None, xte=None, he=None, interval=30.0):
    """Save plots of speed, XTE, HE vs time periodically."""
    if not hasattr(node, 'start_time'):
        raise AttributeError('node must have attribute start_time')
    if not hasattr(node, 'prev_plot_time') or node.prev_plot_time is None:
        node.prev_plot_time = node.start_time

    for attr, default in [
        ('speed_times', []), ('speeds', []),
        ('xte_times', []),   ('xte_vals', []),
        ('he_times', []),    ('he_vals', []),
    ]:
        if not hasattr(node, attr):
            setattr(node, attr, default)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(base_dir, '..', 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    for attr, fname in [
        ('plot_file',     'speed_plot.png'),
        ('xte_plot_file', 'xte_plot.png'),
        ('he_plot_file',  'he_plot.png'),
    ]:
        if not hasattr(node, attr):
            setattr(node, attr, os.path.join(plots_dir, fname))

    elapsed = (cur_time - node.start_time).nanoseconds / 1e9

    if speed is not None:
        node.speed_times.append(elapsed)
        node.speeds.append(float(speed))
    if xte is not None:
        node.xte_times.append(elapsed)
        node.xte_vals.append(float(xte))
    if he is not None:
        node.he_times.append(elapsed)
        node.he_vals.append(float(he))

    since_last = (cur_time - node.prev_plot_time).nanoseconds / 1e9
    if since_last < float(interval):
        return

    if node.speed_times:
        plt.figure(figsize=(8, 4))
        plt.plot(node.speed_times, node.speeds, '-b')
        plt.xlabel('Time (s)'); plt.ylabel('Speed (m/s)')
        plt.title('Speed vs Time'); plt.grid(True); plt.tight_layout()
        plt.savefig(node.plot_file); plt.close()

    if node.xte_times:
        plt.figure(figsize=(8, 4))
        plt.plot(node.xte_times, node.xte_vals, '-r')
        plt.axhline(y=1.3,  color='k', linestyle='--', linewidth=1)
        plt.axhline(y=-1.3, color='k', linestyle='--', linewidth=1)
        plt.xlabel('Time (s)'); plt.ylabel('XTE (m)')
        plt.title('Cross Track Error vs Time'); plt.grid(True); plt.tight_layout()
        plt.savefig(node.xte_plot_file); plt.close()

    if node.he_times:
        plt.figure(figsize=(8, 4))
        plt.plot(node.he_times, node.he_vals, '-g')
        plt.xlabel('Time (s)'); plt.ylabel('HE (deg)')
        plt.title('Heading Error vs Time'); plt.grid(True); plt.tight_layout()
        plt.savefig(node.he_plot_file); plt.close()

    node.prev_plot_time = cur_time
