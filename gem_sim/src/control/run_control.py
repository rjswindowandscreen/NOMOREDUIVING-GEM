#!/usr/bin/env python3
"""
Run control module.
Usage: python3 src/control/run_control.py

Make sure the sim is already running:
    ros2 launch gem_launch gem_init.launch.py

And waypoints have been recorded:
    python3 src/control/scripts/waypoint_recorder.py
"""

import subprocess
import sys
import os

CONTROL_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(CONTROL_DIR, 'scripts')

SCRIPTS = [
    'main.py',
]


def main():
    processes = []

    print('[control] Starting control module...')

    for script in SCRIPTS:
        script_path = os.path.join(SCRIPTS_DIR, script)
        print(f'[control] Starting {script}...')
        p = subprocess.Popen(
            [sys.executable, script_path],
            cwd=CONTROL_DIR
        )
        processes.append((script, p))

    print('[control] Running. Press Ctrl+C to stop.\n')

    try:
        while True:
            for script, p in processes:
                if p.poll() is not None:
                    print(f'\n[control] {script} exited. Shutting down...')
                    for _, other in processes:
                        other.terminate()
                    sys.exit(1)
    except KeyboardInterrupt:
        print('\n[control] Shutting down...')
        for _, p in processes:
            p.terminate()
        for _, p in processes:
            p.wait()
        print('[control] Done.')


if __name__ == '__main__':
    main()
