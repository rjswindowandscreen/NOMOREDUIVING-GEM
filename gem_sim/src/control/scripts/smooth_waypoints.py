#!/usr/bin/env python3
"""
Smooths recorded highbay waypoints using a moving average.
Run this AFTER waypoint_recorder.py has saved highbay_waypoints.py.

Usage:
    python3 src/control/scripts/smooth_waypoints.py
"""
import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from highbay_waypoints import WayPoints
except ImportError:
    print('ERROR: highbay_waypoints.py not found. Run waypoint_recorder.py first.')
    sys.exit(1)


def smooth(waypoints, window=5):
    """Moving average smoothing."""
    pts = np.array(waypoints)
    n   = len(pts)
    smoothed = []
    for i in range(n):
        # Wrap around for loop closure
        indices = [(i + j - window // 2) % n for j in range(window)]
        avg = pts[indices].mean(axis=0)
        smoothed.append((float(avg[0]), float(avg[1])))
    return smoothed


def resample(waypoints, target_spacing=0.5):
    """
    Resample waypoints to have uniform spacing.
    target_spacing: desired distance between waypoints in meters.
    """
    resampled = [waypoints[0]]
    accumulated = 0.0
    for i in range(1, len(waypoints)):
        dx = waypoints[i][0] - waypoints[i-1][0]
        dy = waypoints[i][1] - waypoints[i-1][1]
        seg = math.hypot(dx, dy)
        accumulated += seg
        if accumulated >= target_spacing:
            resampled.append(waypoints[i])
            accumulated = 0.0
    return resampled


def main():
    raw = WayPoints().getWayPoints()
    print(f'Loaded {len(raw)} raw waypoints')

    # Step 1: smooth out driving wobble
    smoothed = smooth(raw, window=7)
    print(f'Smoothed with window=7')

    # Step 2: resample to uniform 0.5m spacing
    resampled = resample(smoothed, target_spacing=0.5)
    print(f'Resampled to {len(resampled)} waypoints at 0.5m spacing')

    # Step 3: close the loop
    resampled.append(resampled[0])

    # Save back to highbay_waypoints.py
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'highbay_waypoints.py'
    )

    lines = []
    lines.append('#!/usr/bin/env python3')
    lines.append('"""')
    lines.append('Highbay track waypoints — recorded by driving and smoothed.')
    lines.append('"""')
    lines.append('')
    lines.append('class WayPoints:')
    lines.append('    def __init__(self):')
    lines.append('        self.waypoints = [')
    for x, y in resampled:
        lines.append(f'            ({x:.4f}, {y:.4f}),')
    lines.append('        ]')
    lines.append('')
    lines.append('    def getWayPoints(self):')
    lines.append('        return self.waypoints')
    lines.append('')

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f'Saved smoothed waypoints to {out_path}')


if __name__ == '__main__':
    main()