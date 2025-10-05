#!/usr/bin/env python3
"""
Quick launcher script for the OpenBCI PyQt Demo
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main application
from openbci_pyqt_demo import main

if __name__ == "__main__":
    main()