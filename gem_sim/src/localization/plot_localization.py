#!/usr/bin/env python3
"""
plot_localization.py — Subscribe to /odometry/global and /navsatfix,
record the run, then save a trajectory plot and metrics to
src/localization/data/

Usage:
    # Terminal 1
    python3 run_localization.py

    # Terminal 2 — drive around, Ctrl+C to stop and save
    python3 plot_localization.py

Output in src/localization/data/:
    trajectory_<timestamp>.png
    metrics_<timestamp>.csv
    summary_<timestamp>.txt
"""

import os
import sys
import math
import csv
import datetime
import rclpy
from rclpy.node import Node

from nav_msgs.msg    import Odometry
from sensor_msgs.msg import NavSatFix

# ros_topics.py is in the same folder as this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ros_topics import GPS_TOPIC, ODOM_TOPIC

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.collections as mc
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print('WARNING: matplotlib not found.  pip3 install matplotlib --break-system-packages')


# ── Output directory ──────────────────────────────────────────────────────────
THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(THIS_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

R_EARTH = 6_371_000.0


def latlon_to_xy(lat, lon, lat0, lon0):
    d_lat   = math.radians(lat - lat0)
    d_lon   = math.radians(lon - lon0)
    lat_mid = math.radians((lat + lat0) / 2.0)
    x = R_EARTH * d_lon * math.cos(lat_mid)
    y = R_EARTH * d_lat
    return x, y


# ── ROS2 node ─────────────────────────────────────────────────────────────────
class LocalizationLogger(Node):

    def __init__(self):
        super().__init__('localization_logger')

        self.kf_path   = []
        self.kf_record = []
        self.gps_raw   = []
        self.lat0      = None
        self.lon0      = None

        self.create_subscription(Odometry,  ODOM_TOPIC, self.odom_cb, 10)
        self.create_subscription(NavSatFix, GPS_TOPIC,  self.gps_cb,  10)

        self.get_logger().info(f'Logging {ODOM_TOPIC} and {GPS_TOPIC}')
        self.get_logger().info('Press Ctrl+C to stop and save.')

    # -------------------------------------------------------------------------
    def odom_cb(self, msg):
        x  = msg.pose.pose.position.x
        y  = msg.pose.pose.position.y
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        yr = msg.twist.twist.angular.z
        q  = msg.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        t   = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        self.kf_path.append((x, y))
        self.kf_record.append({
            'time':     t,
            'kf_x':     x,
            'kf_y':     y,
            'vx':       vx,
            'vy':       vy,
            'speed':    math.hypot(vx, vy),
            'yaw_deg':  math.degrees(yaw),
            'yaw_rate': yr,
        })

    def gps_cb(self, msg):
        if msg.status.status < 0:
            return
        if self.lat0 is None:
            self.lat0 = msg.latitude
            self.lon0 = msg.longitude
            return
        gx, gy = latlon_to_xy(msg.latitude, msg.longitude, self.lat0, self.lon0)
        self.gps_raw.append((gx, gy))

    # -------------------------------------------------------------------------
    def save(self):
        if not self.kf_record:
            print('No data recorded — nothing to save.')
            return

        print(f'\nRecorded {len(self.kf_record)} odom messages, '
              f'{len(self.gps_raw)} GPS fixes.')
        print(f'Saving to {DATA_DIR}/')

        self._save_csv()
        self._save_summary()
        if HAS_MPL:
            self._save_plot()
        else:
            print('  (skipping plot — matplotlib not installed)')

    # -------------------------------------------------------------------------
    def _save_csv(self):
        path  = os.path.join(DATA_DIR, f'metrics_{TIMESTAMP}.csv')
        n_gps = len(self.gps_raw)
        n_kf  = len(self.kf_record)
        rows  = []

        for i, (gx, gy) in enumerate(self.gps_raw):
            kf_idx = min(int(i / max(n_gps - 1, 1) * (n_kf - 1)), n_kf - 1)
            rec    = self.kf_record[kf_idx]
            rows.append({
                'gps_fix_num':  i + 1,
                'time':         round(rec['time'],     3),
                'kf_x':         round(rec['kf_x'],     3),
                'kf_y':         round(rec['kf_y'],     3),
                'gps_x':        round(gx, 3),
                'gps_y':        round(gy, 3),
                'error_m':      round(math.hypot(rec['kf_x'] - gx, rec['kf_y'] - gy), 3),
                'speed_mps':    round(rec['speed'],    3),
                'yaw_deg':      round(rec['yaw_deg'],  2),
                'yaw_rate_rps': round(rec['yaw_rate'], 4),
            })

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f'  CSV saved:      {path}')

    # -------------------------------------------------------------------------
    def _save_summary(self):
        path  = os.path.join(DATA_DIR, f'summary_{TIMESTAMP}.txt')
        n_gps = len(self.gps_raw)
        n_kf  = len(self.kf_record)

        errors = []
        for i, (gx, gy) in enumerate(self.gps_raw):
            kf_idx = min(int(i / max(n_gps - 1, 1) * (n_kf - 1)), n_kf - 1)
            rec = self.kf_record[kf_idx]
            errors.append(math.hypot(rec['kf_x'] - gx, rec['kf_y'] - gy))

        total_dist = 0.0
        for i in range(1, len(self.kf_path)):
            total_dist += math.hypot(
                self.kf_path[i][0] - self.kf_path[i-1][0],
                self.kf_path[i][1] - self.kf_path[i-1][1]
            )

        duration = self.kf_record[-1]['time'] - self.kf_record[0]['time']
        speeds   = [r['speed'] for r in self.kf_record]
        mean_err = sum(errors) / len(errors) if errors else 0
        max_err  = max(errors) if errors else 0
        min_err  = min(errors) if errors else 0

        lines = [
            '=' * 50,
            '  LOCALIZATION RUN SUMMARY',
            f'  {TIMESTAMP}',
            '=' * 50,
            '',
            f'  IMU topic         : {ODOM_TOPIC}',
            f'  GPS topic         : {GPS_TOPIC}',
            '',
            f'  Duration          : {duration:.1f} s',
            f'  Odom messages     : {len(self.kf_record)}',
            f'  GPS fixes         : {n_gps}',
            f'  Distance driven   : {total_dist:.2f} m',
            '',
            '  -- KF vs GPS error --',
            f'  Mean error        : {mean_err:.3f} m',
            f'  Max error         : {max_err:.3f} m',
            f'  Min error         : {min_err:.3f} m',
            '',
            '  -- Speed --',
            f'  Mean speed        : {sum(speeds)/len(speeds):.3f} m/s',
            f'  Max speed         : {max(speeds):.3f} m/s',
            '',
            '  -- Final position --',
            f'  x = {self.kf_record[-1]["kf_x"]:.3f} m  (east)',
            f'  y = {self.kf_record[-1]["kf_y"]:.3f} m  (north)',
            '',
            '=' * 50,
        ]

        with open(path, 'w') as f:
            f.write('\n'.join(lines))

        print()
        for line in lines:
            print(line)
        print(f'\n  Summary saved:  {path}')

    # -------------------------------------------------------------------------
    def _save_plot(self):
        path = os.path.join(DATA_DIR, f'trajectory_{TIMESTAMP}.png')

        kf_xs  = [p[0] for p in self.kf_path]
        kf_ys  = [p[1] for p in self.kf_path]
        gps_xs = [p[0] for p in self.gps_raw]
        gps_ys = [p[1] for p in self.gps_raw]
        times  = [r['time'] - self.kf_record[0]['time'] for r in self.kf_record]
        speeds = [r['speed'] for r in self.kf_record]
        yaws   = [r['yaw_deg'] for r in self.kf_record]
        n_kf   = len(self.kf_record)
        n_gps  = len(self.gps_raw)

        errors, err_times = [], []
        for i, (gx, gy) in enumerate(self.gps_raw):
            kf_idx = min(int(i / max(n_gps - 1, 1) * (n_kf - 1)), n_kf - 1)
            rec = self.kf_record[kf_idx]
            errors.append(math.hypot(rec['kf_x'] - gx, rec['kf_y'] - gy))
            err_times.append(rec['time'] - self.kf_record[0]['time'])

        fig = plt.figure(figsize=(14, 12))
        fig.suptitle(f'GEM Localization — {TIMESTAMP}', fontsize=13, fontweight='normal', y=0.98)

        gs      = fig.add_gridspec(2, 3, height_ratios=[1.6, 1], hspace=0.38, wspace=0.32)
        ax_traj = fig.add_subplot(gs[0, :])
        ax_err  = fig.add_subplot(gs[1, 0])
        ax_spd  = fig.add_subplot(gs[1, 1])
        ax_yaw  = fig.add_subplot(gs[1, 2])

        # ── Trajectory ────────────────────────────────────────────────────
        ax = ax_traj

        t_arr  = np.array(times)
        x_arr  = np.array(kf_xs)
        y_arr  = np.array(kf_ys)
        t_norm = (t_arr - t_arr.min()) / max(t_arr.max() - t_arr.min(), 1e-6)

        points = np.array([x_arr, y_arr]).T.reshape(-1, 1, 2)
        segs   = np.concatenate([points[:-1], points[1:]], axis=1)
        lc     = mc.LineCollection(segs, cmap='viridis', linewidth=2.5, zorder=3)
        lc.set_array(t_norm[:-1])
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, pad=0.01, fraction=0.02)
        cbar.set_label('time  (s)', fontsize=9)
        cbar.set_ticks([0, 0.5, 1.0])
        cbar.set_ticklabels(['0', f'{t_arr.max()/2:.0f}', f'{t_arr.max():.0f}'])

        # heading arrows
        yaw_rads   = [math.radians(y) for y in yaws]
        arrow_step = max(1, len(kf_xs) // 20)
        for i in range(0, len(kf_xs) - arrow_step, arrow_step):
            dx = math.cos(yaw_rads[i]) * 0.4
            dy = math.sin(yaw_rads[i]) * 0.4
            ax.annotate('', xy=(kf_xs[i]+dx, kf_ys[i]+dy),
                        xytext=(kf_xs[i], kf_ys[i]),
                        arrowprops=dict(arrowstyle='->', color='#1D9E75',
                                        lw=1.2, mutation_scale=10),
                        zorder=5)

        # GPS fixes + error lines
        if gps_xs:
            for i, (gx, gy) in enumerate(self.gps_raw):
                kf_idx = min(int(i / max(n_gps - 1, 1) * (n_kf - 1)), n_kf - 1)
                kx, ky = kf_xs[kf_idx], kf_ys[kf_idx]
                ax.plot([gx, kx], [gy, ky], color='#EF9F27', linewidth=0.7,
                        alpha=0.45, zorder=2)
            ax.scatter(gps_xs, gps_ys, color='#EF9F27', s=22, zorder=4,
                       label=f'GPS fix  ({GPS_TOPIC})', edgecolors='none', alpha=0.85)

        # distance markers every 10 m
        dist_acc  = 0.0
        next_mark = 10.0
        for i in range(1, len(kf_xs)):
            dist_acc += math.hypot(kf_xs[i]-kf_xs[i-1], kf_ys[i]-kf_ys[i-1])
            if dist_acc >= next_mark:
                ax.plot(kf_xs[i], kf_ys[i], '|', color='gray',
                        markersize=8, markeredgewidth=1.2, zorder=5)
                ax.text(kf_xs[i]+0.3, kf_ys[i]+0.3,
                        f'{next_mark:.0f}m', fontsize=7.5, color='gray', zorder=6)
                next_mark += 10.0

        # start / end
        ax.plot(kf_xs[0],  kf_ys[0],  'o', color='#1D9E75', markersize=11, zorder=6)
        ax.plot(kf_xs[-1], kf_ys[-1], '*', color='#533AB7', markersize=14, zorder=6)
        ax.text(kf_xs[0]   + 0.4, kf_ys[0]   + 0.4, 'Start', fontsize=8.5, color='#1D9E75')
        ax.text(kf_xs[-1]  + 0.4, kf_ys[-1]  + 0.4, 'End',   fontsize=8.5, color='#533AB7')

        x_pad = max((max(kf_xs) - min(kf_xs)) * 0.1, 2)
        y_pad = max((max(kf_ys) - min(kf_ys)) * 0.3, 5)
        ax.set_xlim(min(kf_xs) - x_pad, max(kf_xs) + x_pad)
        ax.set_ylim(min(kf_ys) - y_pad, max(kf_ys) + y_pad)
        ax.set_xlabel('x  (metres east)',  fontsize=10)
        ax.set_ylabel('y  (metres north)', fontsize=10)
        mean_e = sum(errors)/len(errors) if errors else 0
        ax.set_title(
            f'Path  —  {mean_e:.2f} m mean GPS error  |  '
            f'{max(errors):.2f} m max  |  '
            f'{sum(1 for s in speeds if s > 0.1) / (len(speeds)+1e-9) * 100:.0f}% moving',
            fontsize=9, color='#555'
        )
        ax.grid(True, linewidth=0.5, alpha=0.4)

        # compass rose
        cx = max(kf_xs) + x_pad * 0.6
        cy = min(kf_ys) - y_pad * 0.5
        r  = x_pad * 0.35
        ax.annotate('', xy=(cx, cy+r), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
        ax.text(cx, cy+r+0.3, 'N', ha='center', va='bottom', fontsize=8, color='gray')
        ax.annotate('', xy=(cx+r, cy), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
        ax.text(cx+r+0.3, cy, 'E', ha='left', va='center', fontsize=8, color='gray')

        handles = [
            plt.Line2D([0], [0], color='#1D9E75', lw=2, label=f'KF path  ({ODOM_TOPIC})'),
            plt.scatter([], [], color='#EF9F27', s=18, label=f'GPS fix  ({GPS_TOPIC})'),
        ]
        ax.legend(handles=handles, fontsize=8.5, loc='upper left',
                  framealpha=0.7, handlelength=1.5)

        # ── Error over time ────────────────────────────────────────────────
        ax = ax_err
        if errors:
            ax.plot(err_times, errors, color='#D85A30', linewidth=1.5)
            ax.fill_between(err_times, 0, errors, alpha=0.15, color='#D85A30')
            ax.axhline(mean_e, color='#D85A30', linewidth=1, linestyle='--',
                       alpha=0.7, label=f'mean {mean_e:.2f} m')
            ax.legend(fontsize=8)
        ax.set_xlabel('time  (s)', fontsize=9)
        ax.set_ylabel('error  (m)', fontsize=9)
        ax.set_title('KF vs GPS error', fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(True, linewidth=0.5, alpha=0.4)

        # ── Speed ──────────────────────────────────────────────────────────
        ax = ax_spd
        ax.plot(times, speeds, color='#378ADD', linewidth=1.5)
        ax.set_xlabel('time  (s)', fontsize=9)
        ax.set_ylabel('speed  (m/s)', fontsize=9)
        ax.set_title('Speed', fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(True, linewidth=0.5, alpha=0.4)

        # ── Yaw ────────────────────────────────────────────────────────────
        ax = ax_yaw
        ax.plot(times, yaws, color='#7F77DD', linewidth=1.5)
        for ref in [0, 90, -90, 180, -180]:
            ax.axhline(ref, color='gray', linewidth=0.5, linestyle='--', alpha=0.35)
        ax.set_xlabel('time  (s)', fontsize=9)
        ax.set_ylabel('yaw  (degrees)', fontsize=9)
        ax.set_title('Heading', fontsize=10)
        ax.grid(True, linewidth=0.5, alpha=0.4)

        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Plot saved:     {path}')


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = LocalizationLogger()

    print('Press Ctrl+C to stop recording and save.')

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass

    print('\nStopping...')
    node.save()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()