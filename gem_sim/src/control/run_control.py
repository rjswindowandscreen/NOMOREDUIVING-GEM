#!/usr/bin/env python3
"""
Run control module.
Usage:
    python3 src/control/run_control.py            # sim
    python3 src/control/run_control.py --real     # real GEM (also runs pacmod_bridge)

Make sure the sim is already running (sim only):
    ros2 launch gem_launch gem_init.launch.py
"""

import argparse
import os
import subprocess
import sys

CONTROL_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(CONTROL_DIR, 'scripts')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', action='store_true',
                        help='Also launch pacmod_bridge for real-vehicle operation')
    args = parser.parse_args()

    scripts = ['main.py']
    if args.real:
        scripts.append('pacmod_bridge.py')

    processes = []

    print('[control] Starting control module...')

    for script in scripts:
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
