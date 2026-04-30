#!/usr/bin/env python3
"""
localization.py — Entry point for the localization module.
Run this file to start the localization node:
    python3 localization.py
"""

import os
import sys

# Add scripts/ folder to path so localization_node can import helpers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from localization import main

if __name__ == '__main__':
    main()