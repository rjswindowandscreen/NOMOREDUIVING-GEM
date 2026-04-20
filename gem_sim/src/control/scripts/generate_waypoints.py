#!/usr/bin/env python3
"""
Generate highbay waypoints from the highbay_track.dae mesh geometry.

This script extracts the Lane_Left and Lane_Right geometry from the Gazebo
mesh, averages them to compute the lane centerline, and saves the result
as highbay_waypoints.py.

This is far more accurate than recording waypoints by driving — the
waypoints are mathematically centered in the lane and perfectly ordered.

Usage:
    python3 src/control/scripts/generate_waypoints.py
"""

import os
import sys
import xml.etree.ElementTree as ET
import numpy as np


# Path to the highbay_track DAE mesh
# Assumes gem_sim/src/gem_simulator is a sibling of gem_sim/src/control
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
CONTROL_DIR = os.path.dirname(SCRIPTS_DIR)
GEM_SIM_DIR = os.path.join(os.path.dirname(CONTROL_DIR), 'gem_simulator')
DAE_PATH = os.path.join(
    GEM_SIM_DIR,
    'gem_gazebo', 'models', 'highbay_track', 'meshes', 'highbay_track.dae'
)
OUTPUT_PATH = os.path.join(SCRIPTS_DIR, 'highbay_waypoints.py')


def get_vertices(root, geometry_name, ns):
    """Extract vertex positions from a named geometry in the DAE file."""
    for g in root.findall('.//c:geometry', ns):
        if geometry_name in g.get('id', '') or geometry_name in g.get('name', ''):
            positions = g.find('.//c:float_array', ns)
            if positions is not None:
                vals = list(map(float, positions.text.strip().split()))
                return np.array(vals).reshape(-1, 3)
    return None


def quad_centers(vertices):
    """Average every group of 4 vertices (one quad face) to get face centers."""
    centers = []
    for i in range(0, len(vertices) - 3, 4):
        centers.append(vertices[i:i+4].mean(axis=0))
    return np.array(centers)


def main():
    if not os.path.exists(DAE_PATH):
        print(f'ERROR: Could not find highbay_track.dae at:\n  {DAE_PATH}')
        print('Make sure gem_simulator is present at gem_sim/src/gem_simulator/')
        sys.exit(1)

    print(f'Reading mesh from:\n  {DAE_PATH}')

    tree = ET.parse(DAE_PATH)
    root = tree.getroot()
    ns = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}

    left_verts  = get_vertices(root, 'Lane_Left',  ns)
    right_verts = get_vertices(root, 'Lane_Right', ns)

    if left_verts is None or right_verts is None:
        print('ERROR: Could not find Lane_Left or Lane_Right geometry in mesh.')
        sys.exit(1)

    left_centers  = quad_centers(left_verts)
    right_centers = quad_centers(right_verts)

    min_len    = min(len(left_centers), len(right_centers))
    centerline = (left_centers[:min_len] + right_centers[:min_len]) / 2

    # Verify it's a closed loop
    first = centerline[0, :2]
    last  = centerline[-1, :2]
    gap   = np.linalg.norm(first - last)
    print(f'Track gap (last to first point): {gap:.3f}m')

    # Verify clockwise direction
    pts  = centerline[:, :2]
    n    = len(pts)
    area = sum(
        (pts[i][0] * pts[(i+1)%n][1] - pts[(i+1)%n][0] * pts[i][1])
        for i in range(n)
    )
    direction = 'clockwise' if area < 0 else 'counter-clockwise'
    print(f'Track direction: {direction}')

    # Spacing stats
    dists = [np.linalg.norm(pts[i+1] - pts[i]) for i in range(len(pts)-1)]
    print(f'Waypoint spacing: min={min(dists):.2f}m  max={max(dists):.2f}m  avg={np.mean(dists):.2f}m')
    print(f'Total waypoints: {len(centerline)}')

    # Write output file
    lines = []
    lines.append('#!/usr/bin/env python3')
    lines.append('"""')
    lines.append('Highbay track centerline waypoints.')
    lines.append('Auto-generated from highbay_track.dae mesh geometry.')
    lines.append('Do not edit manually — regenerate with generate_waypoints.py.')
    lines.append('"""')
    lines.append('')
    lines.append('class WayPoints:')
    lines.append('    def __init__(self):')
    lines.append('        self.waypoints = [')
    for pt in centerline:
        lines.append(f'            ({pt[0]:.4f}, {pt[1]:.4f}),')
    # Close the loop
    lines.append(f'            ({centerline[0][0]:.4f}, {centerline[0][1]:.4f}),')
    lines.append('        ]')
    lines.append('')
    lines.append('    def getWayPoints(self):')
    lines.append('        return self.waypoints')
    lines.append('')

    with open(OUTPUT_PATH, 'w') as f:
        f.write('\n'.join(lines))

    print(f'\nSaved to:\n  {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
