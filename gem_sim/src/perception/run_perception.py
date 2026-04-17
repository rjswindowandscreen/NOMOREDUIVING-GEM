#!/usr/bin/env python3
"""
Run perception module.
Usage: python3 src/perception/run_perception.py
"""
import subprocess
import sys
import os

# Always run scripts relative to this file's location (perception/)
PERCEPTION_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(PERCEPTION_DIR, "scripts")

SCRIPTS = [
    "drive.py",
    "lane_detect.py",
]

def main():
    processes = []

    print("[perception] Starting all scripts...")

    for script in SCRIPTS:
        script_path = os.path.join(SCRIPTS_DIR, script)
        print(f"[perception] Starting {script}...")
        p = subprocess.Popen(
            [sys.executable, script_path],
            cwd=PERCEPTION_DIR  # sets working directory to perception/
        )
        processes.append((script, p))

    print("[perception] All scripts running. Press Ctrl+C to stop.\n")

    try:
        while True:
            for script, p in processes:
                if p.poll() is not None:
                    print(f"\n[perception] {script} exited unexpectedly. Shutting down...")
                    for _, other in processes:
                        other.terminate()
                    sys.exit(1)
    except KeyboardInterrupt:
        print("\n[perception] Shutting down...")
        for _, p in processes:
            p.terminate()
        for _, p in processes:
            p.wait()
        print("[perception] Done.")

if __name__ == "__main__":
    main()