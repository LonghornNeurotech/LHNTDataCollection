#!/usr/bin/env python3
"""
Test script for OpenBCI PyQt Demo
Validates that all components can be imported and initialized properly.
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test PyQt5 imports
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import QTimer
        from PyQt5.QtGui import QIcon
        print("✓ PyQt5 imports successful")
        
        # Test pyqtgraph
        import pyqtgraph as pg
        print("✓ PyQtGraph import successful")
        
        # Test existing modules
        from eeg_processor import EEGProcessor, find_serial_port
        print("✓ EEG processor import successful")
        
        # Test demo components
        from openbci_pyqt_demo import (
            EEGDataWorker, 
            RealTimePlotWidget, 
            ControlPanel, 
            OpenBCIMainWindow
        )
        print("✓ Demo components import successful")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_component_initialization():
    """Test that components can be initialized without errors."""
    print("\nTesting component initialization...")
    
    try:
        # Import components first
        from PyQt5.QtWidgets import QApplication
        from openbci_pyqt_demo import (
            EEGDataWorker, 
            RealTimePlotWidget, 
            ControlPanel
        )
        
        # Create minimal QApplication if needed
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        # Test EEGDataWorker
        worker = EEGDataWorker(use_synthetic=True)
        print("✓ EEGDataWorker initialization successful")
        
        # Test RealTimePlotWidget
        plot_widget = RealTimePlotWidget(num_channels=8, window_size=100)
        print("✓ RealTimePlotWidget initialization successful")
        
        # Test ControlPanel
        control_panel = ControlPanel()
        print("✓ ControlPanel initialization successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Initialization error: {e}")
        return False

def test_port_detection():
    """Test serial port detection."""
    print("\nTesting port detection...")
    
    try:
        from eeg_processor import find_serial_port
        port = find_serial_port()
        if port:
            print(f"✓ Found serial port: {port}")
        else:
            print("! No OpenBCI port detected (this is normal if no board is connected)")
        return True
        
    except Exception as e:
        print(f"✗ Port detection error: {e}")
        return False

def main():
    """Run all tests."""
    print("OpenBCI PyQt Demo - Component Tests")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_component_initialization,
        test_port_detection
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The demo is ready to use.")
        print("\nTo launch the demo:")
        print("  python openbci_pyqt_demo.py")
        print("  or")
        print("  python run_demo.py")
    else:
        print("✗ Some tests failed. Check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)