#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from localization import main

if __name__ == '__main__':
    main()