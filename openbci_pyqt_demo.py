#!/usr/bin/env python3
"""
OpenBCI EEG Data Collection PyQt Demo
=====================================

A comprehensive PyQt application for real-time EEG data collection and visualization
using OpenBCI headsets. This demo integrates with the existing EEGProcessor class
and provides a modern GUI interface for data collection, visualization, and recording.

Features:
- Real-time EEG data visualization
- OpenBCI board connection management
- Data recording and export
- Signal filtering and processing controls
- Multi-channel display
- Status monitoring and logging

Author: GitHub Copilot
Date: October 2025
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

# PyQt imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QSplitter, QTabWidget, QGroupBox, QLabel, 
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QProgressBar, QTextEdit, QTableWidget, QTableWidgetItem,
    QMenuBar, QStatusBar, QToolBar, QAction, QMessageBox,
    QFileDialog, QSlider, QFrame, QSizePolicy, QRadioButton, QButtonGroup
)
from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QObject, QMutex, QWaitCondition
)
from PyQt5.QtGui import (
    QFont, QIcon, QPalette, QColor, QPixmap, QPainter, QBrush
)

# Plotting imports
import pyqtgraph as pg
import pyqtgraph.exporters

# Import the existing EEG processor
from eeg_processor import EEGProcessor, find_serial_port


class StoredDataManager:
    """
    Manager for loading and serving stored EEG data from .pkl files.
    Expects pickle file with data sampled at 125 Hz in 1-second segments.
    """
    
    def __init__(self, pkl_file_path: str = None):
        self.pkl_file_path = pkl_file_path
        self.samples = []  # List of 1-second samples
        self.num_channels = 0
        self.sampling_rate = 125
        self.current_sample_idx = 0
        
        if pkl_file_path and os.path.exists(pkl_file_path):
            self.load_data(pkl_file_path)
    
    def load_data(self, pkl_file_path: str):
        """
        Load EEG data from pickle file.
        Expected format: List of numpy arrays with shape (channels, 125)
        or a single array with shape (num_samples, channels, 125)
        """
        try:
            with open(pkl_file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Handle different data formats
            if isinstance(data, list):
                self.samples = data
            elif isinstance(data, np.ndarray):
                if data.ndim == 3:  # (num_samples, channels, time_points)
                    self.samples = [data[i] for i in range(data.shape[0])]
                elif data.ndim == 2:  # (channels, time_points) - split into 1-sec chunks
                    num_samples = data.shape[1] // self.sampling_rate
                    self.samples = [
                        data[:, i*self.sampling_rate:(i+1)*self.sampling_rate]
                        for i in range(num_samples)
                    ]
            elif isinstance(data, dict):
                # Handle dictionary format with 'samples' or 'data' key
                if 'samples' in data:
                    self.samples = data['samples']
                elif 'data' in data:
                    raw_data = data['data']
                    if isinstance(raw_data, np.ndarray) and raw_data.ndim == 2:
                        num_samples = raw_data.shape[1] // self.sampling_rate
                        self.samples = [
                            raw_data[:, i*self.sampling_rate:(i+1)*self.sampling_rate]
                            for i in range(num_samples)
                        ]
            
            if self.samples:
                # Get number of channels from first sample
                first_sample = self.samples[0]
                if first_sample.ndim == 2:
                    self.num_channels = first_sample.shape[0]
                elif first_sample.ndim == 1:
                    self.num_channels = 1
                    self.samples = [s.reshape(1, -1) for s in self.samples]
                
                print(f"Loaded {len(self.samples)} samples with {self.num_channels} channels")
            else:
                raise ValueError("No valid samples found in pickle file")
                
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            raise
    
    def get_sample(self, index: int) -> np.ndarray:
        """
        Get a specific 1-second sample by index.
        Returns array with shape (1, channels, 125) for consistency with live data.
        """
        if not self.samples:
            return np.zeros((1, self.num_channels or 8, self.sampling_rate))
        
        index = max(0, min(index, len(self.samples) - 1))
        sample = self.samples[index]
        
        # Ensure shape is (1, channels, time_points)
        if sample.ndim == 2:
            sample = sample[np.newaxis, :, :]
        
        return sample
    
    def get_num_samples(self) -> int:
        """Return the number of available samples."""
        return len(self.samples)


class EEGDataWorker(QObject):
    """
    Worker thread for continuous EEG data acquisition and processing.
    Runs in a separate thread to prevent GUI blocking.
    """
    
    # Signals for communicating with the main thread
    data_ready = pyqtSignal(np.ndarray)  # Emits new EEG data
    connection_status = pyqtSignal(bool, str)  # Connection status and message
    error_occurred = pyqtSignal(str)  # Error messages
    
    def __init__(self, use_synthetic=False):
        super().__init__()
        self.eeg_processor: Optional[EEGProcessor] = None
        self.use_synthetic = use_synthetic
        self.running = False
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        
    def start_acquisition(self):
        """Start the EEG data acquisition process."""
        try:
            self.eeg_processor = EEGProcessor()
            self.running = True
            self.connection_status.emit(True, "Connected to OpenBCI board")
            
            # Start the acquisition loop
            self.run_acquisition()
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to start acquisition: {str(e)}")
            self.connection_status.emit(False, f"Connection failed: {str(e)}")
    
    def stop_acquisition(self):
        """Stop the EEG data acquisition process."""
        self.running = False
        if self.eeg_processor:
            try:
                self.eeg_processor.stop()
                self.connection_status.emit(False, "Disconnected from OpenBCI board")
            except Exception as e:
                self.error_occurred.emit(f"Error stopping acquisition: {str(e)}")
            finally:
                self.eeg_processor = None
    
    def run_acquisition(self):
        """Main acquisition loop - runs continuously while connected."""
        while self.running and self.eeg_processor:
            try:
                # Get recent EEG data (14 seconds by default)
                data = self.eeg_processor.get_recent_data(duration=1)  # Get 1 second of data
                
                if data is not None and data.shape[-1] > 0:
                    # Convert tensor to numpy if needed
                    if hasattr(data, 'numpy'):
                        data = data.numpy()
                    
                    # Emit the data to the main thread
                    self.data_ready.emit(data)
                
                # Brief pause to prevent overwhelming the GUI
                time.sleep(0.1)  # 100ms update rate
                
            except Exception as e:
                self.error_occurred.emit(f"Data acquisition error: {str(e)}")
                break


class RealTimePlotWidget(pg.GraphicsLayoutWidget):
    """
    Custom widget for real-time EEG data plotting using PyQtGraph.
    Displays multiple EEG channels with auto-scaling and scrolling.
    """
    
    def __init__(self, num_channels=8, window_size=1000):
        super().__init__()
        
        self.num_channels = num_channels
        self.window_size = window_size
        self.sampling_rate = 125  # Hz (default for OpenBCI)
        
        # Data storage
        self.data_buffer = np.zeros((num_channels, window_size))
        self.time_axis = np.linspace(0, window_size/self.sampling_rate, window_size)
        
        # Channel information
        self.channel_names = [f"Ch{i+1}" for i in range(num_channels)]
        self.channel_colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
            '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'
        ]
        
        # Plot setup
        self.plots = []
        self.curves = []
        self.setup_plots()
        
        # Auto-scaling parameters
        self.auto_scale = True
        self.y_scale_factor = 50  # Scaling factor for display
        
    def setup_plots(self):
        """Set up the individual channel plots."""
        for i in range(self.num_channels):
            # Create subplot
            plot = self.addPlot(row=i, col=0)
            plot.setLabel('left', self.channel_names[i], units='μV')
            plot.setLabel('bottom', 'Time', units='s')
            plot.showGrid(x=True, y=True, alpha=0.3)
            
            # Configure plot appearance
            plot.setYRange(-100, 100)  # Initial range
            plot.setXRange(0, self.window_size/self.sampling_rate)
            
            # Create curve for this channel
            curve = plot.plot(
                self.time_axis, 
                self.data_buffer[i], 
                pen=pg.mkPen(color=self.channel_colors[i % len(self.channel_colors)], width=2)
            )
            
            self.plots.append(plot)
            self.curves.append(curve)
    
    def update_data(self, new_data):
        """
        Update the plots with new EEG data.
        
        Args:
            new_data: numpy array of shape (1, channels, samples) or (channels, samples)
        """
        try:
            # Handle different input shapes
            if new_data.ndim == 3:
                new_data = new_data[0]  # Remove batch dimension
            
            if new_data.ndim != 2:
                return
            
            channels, samples = new_data.shape
            channels = min(channels, self.num_channels)
            
            if samples == 0:
                return
            
            # Shift existing data and add new samples
            if samples >= self.window_size:
                # If new data is larger than window, take the last window_size samples
                self.data_buffer = new_data[:channels, -self.window_size:]
            else:
                # Shift existing data left and append new data
                self.data_buffer[:channels, :-samples] = self.data_buffer[:channels, samples:]
                self.data_buffer[:channels, -samples:] = new_data[:channels, :]
            
            # Update curves
            for i in range(channels):
                # Apply scaling and offset for better visualization
                display_data = self.data_buffer[i] * self.y_scale_factor
                self.curves[i].setData(self.time_axis, display_data)
                
                # Auto-scale Y axis if enabled
                if self.auto_scale:
                    data_range = np.ptp(display_data)  # Peak-to-peak range
                    if data_range > 0:
                        margin = data_range * 0.1
                        self.plots[i].setYRange(
                            np.min(display_data) - margin,
                            np.max(display_data) + margin
                        )
                        
        except Exception as e:
            print(f"Error updating plot data: {e}")
    
    def set_auto_scale(self, enabled):
        """Enable or disable auto-scaling."""
        self.auto_scale = enabled
    
    def set_y_scale_factor(self, factor):
        """Set the Y-axis scaling factor."""
        self.y_scale_factor = factor
    
    def clear_plots(self):
        """Clear all plot data."""
        self.data_buffer.fill(0)
        for i in range(self.num_channels):
            self.curves[i].setData(self.time_axis, self.data_buffer[i])


class ControlPanel(QGroupBox):
    """
    Control panel widget containing connection controls, recording controls,
    and signal processing settings.
    """
    
    # Signals
    connect_requested = pyqtSignal()
    disconnect_requested = pyqtSignal()
    start_recording = pyqtSignal(str)  # filename
    stop_recording = pyqtSignal()
    settings_changed = pyqtSignal(dict)  # settings dictionary
    mode_changed = pyqtSignal(str)  # 'live' or 'stored'
    sample_selected = pyqtSignal(int)  # sample index
    load_pkl_file = pyqtSignal(str)  # pkl file path
    
    def __init__(self):
        super().__init__("Control Panel")
        self.recording = False
        self.connected = False
        self.current_mode = 'live'
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the control panel UI."""
        layout = QVBoxLayout()
        
        # ============= MODE SELECTION =============
        mode_group = QGroupBox("Data Source Mode")
        mode_layout = QVBoxLayout()
        
        # Radio buttons for mode selection
        self.mode_button_group = QButtonGroup()
        self.live_radio = QRadioButton("Live EEG Device")
        self.stored_radio = QRadioButton("Stored Data (.pkl)")
        
        self.mode_button_group.addButton(self.live_radio)
        self.mode_button_group.addButton(self.stored_radio)
        self.live_radio.setChecked(True)
        
        self.live_radio.toggled.connect(self.on_mode_changed)
        
        mode_layout.addWidget(self.live_radio)
        mode_layout.addWidget(self.stored_radio)
        
        # Load pickle file button
        load_pkl_layout = QHBoxLayout()
        self.load_pkl_btn = QPushButton("Load .pkl File")
        self.load_pkl_btn.clicked.connect(self.on_load_pkl_clicked)
        self.load_pkl_btn.setEnabled(False)
        load_pkl_layout.addWidget(self.load_pkl_btn)
        mode_layout.addLayout(load_pkl_layout)
        
        self.pkl_file_label = QLabel("No file loaded")
        self.pkl_file_label.setWordWrap(True)
        self.pkl_file_label.setStyleSheet("font-size: 9px; color: gray;")
        mode_layout.addWidget(self.pkl_file_label)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # Connection controls (only for live mode)
        self.conn_group = QGroupBox("Connection")
        conn_layout = QVBoxLayout()
        
        # Port selection
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("Port:"))
        self.port_combo = QComboBox()
        self.refresh_ports()
        port_layout.addWidget(self.port_combo)
        
        refresh_btn = QPushButton("🔄")
        refresh_btn.setMaximumWidth(30)
        refresh_btn.clicked.connect(self.refresh_ports)
        port_layout.addWidget(refresh_btn)
        conn_layout.addLayout(port_layout)
        
        # Connection buttons
        btn_layout = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.on_connect_clicked)
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.on_disconnect_clicked)
        self.disconnect_btn.setEnabled(False)
        
        btn_layout.addWidget(self.connect_btn)
        btn_layout.addWidget(self.disconnect_btn)
        conn_layout.addLayout(btn_layout)
        
        # Board type selection
        board_layout = QHBoxLayout()
        board_layout.addWidget(QLabel("Board:"))
        self.board_combo = QComboBox()
        self.board_combo.addItems(["Cyton (8 channels)", "Cyton+Daisy (16 channels)", "Synthetic"])
        board_layout.addWidget(self.board_combo)
        conn_layout.addLayout(board_layout)
        
        self.conn_group.setLayout(conn_layout)
        layout.addWidget(self.conn_group)
        
        # Recording controls
        rec_group = QGroupBox("Recording")
        rec_layout = QVBoxLayout()
        
        # Recording info
        self.recording_label = QLabel("Status: Not recording")
        rec_layout.addWidget(self.recording_label)
        
        # Recording buttons
        rec_btn_layout = QHBoxLayout()
        self.start_rec_btn = QPushButton("Start Recording")
        self.start_rec_btn.clicked.connect(self.on_start_recording)
        self.start_rec_btn.setEnabled(False)
        
        self.stop_rec_btn = QPushButton("Stop Recording")
        self.stop_rec_btn.clicked.connect(self.on_stop_recording)
        self.stop_rec_btn.setEnabled(False)
        
        rec_btn_layout.addWidget(self.start_rec_btn)
        rec_btn_layout.addWidget(self.stop_rec_btn)
        rec_layout.addLayout(rec_btn_layout)
        
        rec_group.setLayout(rec_layout)
        layout.addWidget(rec_group)
        
        # Signal processing settings
        proc_group = QGroupBox("Signal Processing")
        proc_layout = QGridLayout()
        
        # Filter settings
        proc_layout.addWidget(QLabel("Low Cut (Hz):"), 0, 0)
        self.lowcut_spin = QDoubleSpinBox()
        self.lowcut_spin.setRange(0.1, 50.0)
        self.lowcut_spin.setValue(5.0)
        self.lowcut_spin.setSingleStep(0.1)
        proc_layout.addWidget(self.lowcut_spin, 0, 1)
        
        proc_layout.addWidget(QLabel("High Cut (Hz):"), 1, 0)
        self.highcut_spin = QDoubleSpinBox()
        self.highcut_spin.setRange(10.0, 100.0)
        self.highcut_spin.setValue(35.0)
        self.highcut_spin.setSingleStep(0.1)
        proc_layout.addWidget(self.highcut_spin, 1, 1)
        
        proc_layout.addWidget(QLabel("Notch (Hz):"), 2, 0)
        self.notch_spin = QDoubleSpinBox()
        self.notch_spin.setRange(50.0, 60.0)
        self.notch_spin.setValue(60.0)
        self.notch_spin.setSingleStep(0.1)
        proc_layout.addWidget(self.notch_spin, 2, 1)
        
        # Display settings
        proc_layout.addWidget(QLabel("Y Scale:"), 3, 0)
        self.scale_spin = QSpinBox()
        self.scale_spin.setRange(1, 1000)
        self.scale_spin.setValue(50)
        proc_layout.addWidget(self.scale_spin, 3, 1)
        
        # Auto-scale checkbox
        self.autoscale_check = QCheckBox("Auto Scale")
        self.autoscale_check.setChecked(True)
        proc_layout.addWidget(self.autoscale_check, 4, 0, 1, 2)
        
        proc_group.setLayout(proc_layout)
        layout.addWidget(proc_group)
        
        # ============= SAMPLE SELECTOR (for stored mode) =============
        self.sample_group = QGroupBox("Sample Selection")
        sample_layout = QVBoxLayout()
        
        # Sample info label
        self.sample_info_label = QLabel("No samples loaded")
        sample_layout.addWidget(self.sample_info_label)
        
        # Sample selector slider
        slider_layout = QVBoxLayout()
        self.sample_slider = QSlider(Qt.Horizontal)
        self.sample_slider.setMinimum(0)
        self.sample_slider.setMaximum(13)  # Default to 14 samples (0-13)
        self.sample_slider.setValue(0)
        self.sample_slider.setTickPosition(QSlider.TicksBelow)
        self.sample_slider.setTickInterval(1)
        self.sample_slider.valueChanged.connect(self.on_sample_changed)
        slider_layout.addWidget(QLabel("Sample:"))
        slider_layout.addWidget(self.sample_slider)
        
        # Current sample label
        self.current_sample_label = QLabel("Sample 1 of 14")
        self.current_sample_label.setAlignment(Qt.AlignCenter)
        slider_layout.addWidget(self.current_sample_label)
        
        sample_layout.addLayout(slider_layout)
        
        self.sample_group.setLayout(sample_layout)
        self.sample_group.setEnabled(False)  # Disabled until pkl file is loaded
        layout.addWidget(self.sample_group)
        
        # Add stretch to push controls to top
        layout.addStretch()
        
        self.setLayout(layout)
    
    def on_mode_changed(self, checked):
        """Handle mode change between live and stored."""
        if checked:  # Live radio button was toggled on
            self.current_mode = 'live'
            self.conn_group.setEnabled(True)
            self.load_pkl_btn.setEnabled(False)
            self.sample_group.setEnabled(False)
            self.mode_changed.emit('live')
        else:  # Stored radio button
            self.current_mode = 'stored'
            self.conn_group.setEnabled(False)
            self.load_pkl_btn.setEnabled(True)
            # Sample group will be enabled after file is loaded
            self.mode_changed.emit('stored')
    
    def on_load_pkl_clicked(self):
        """Handle load pickle file button click."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Pickle File",
            "",
            "Pickle Files (*.pkl);;All Files (*)"
        )
        
        if filename:
            self.load_pkl_file.emit(filename)
    
    def set_pkl_file_loaded(self, filename: str, num_samples: int):
        """Update UI after pickle file is loaded."""
        self.pkl_file_label.setText(f"Loaded: {os.path.basename(filename)}")
        self.sample_info_label.setText(f"{num_samples} samples available")
        
        # Update slider range
        self.sample_slider.setMaximum(max(0, num_samples - 1))
        self.sample_slider.setValue(0)
        
        # Enable sample selector
        self.sample_group.setEnabled(True)
        
        self.update_sample_label()
    
    def on_sample_changed(self, value):
        """Handle sample slider change."""
        self.update_sample_label()
        self.sample_selected.emit(value)
    
    def update_sample_label(self):
        """Update the current sample label."""
        current = self.sample_slider.value() + 1
        total = self.sample_slider.maximum() + 1
        self.current_sample_label.setText(f"Sample {current} of {total}")
    
    def refresh_ports(self):
        """Refresh the list of available serial ports."""
        self.port_combo.clear()
        ports = []
        
        # Try to find the OpenBCI port automatically
        auto_port = find_serial_port()
        if auto_port:
            ports.append(f"Auto: {auto_port}")
        
        # Add some common port options
        import serial.tools.list_ports
        available_ports = list(serial.tools.list_ports.comports())
        for port in available_ports:
            ports.append(port.device)
        
        if not ports:
            ports.append("No ports found")
        
        self.port_combo.addItems(ports)
    
    def on_connect_clicked(self):
        """Handle connect button click."""
        self.connect_requested.emit()
    
    def on_disconnect_clicked(self):
        """Handle disconnect button click."""
        self.disconnect_requested.emit()
    
    def on_start_recording(self):
        """Handle start recording button click."""
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eeg_recording_{timestamp}.csv"
        
        # Get save location from user
        filename, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Recording As", 
            filename,
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if filename:
            self.start_recording.emit(filename)
    
    def on_stop_recording(self):
        """Handle stop recording button click."""
        self.stop_recording.emit()
    
    def set_connection_status(self, connected, message=""):
        """Update the connection status."""
        self.connected = connected
        self.connect_btn.setEnabled(not connected)
        self.disconnect_btn.setEnabled(connected)
        self.start_rec_btn.setEnabled(connected and not self.recording)
    
    def set_recording_status(self, recording, filename=""):
        """Update the recording status."""
        self.recording = recording
        if recording:
            self.recording_label.setText(f"Status: Recording to {filename}")
            self.start_rec_btn.setEnabled(False)
            self.stop_rec_btn.setEnabled(True)
        else:
            self.recording_label.setText("Status: Not recording")
            self.start_rec_btn.setEnabled(self.connected)
            self.stop_rec_btn.setEnabled(False)


class OpenBCIMainWindow(QMainWindow):
    """
    Main application window for the OpenBCI EEG data collection interface.
    Supports both live EEG device and stored .pkl file data sources.
    """
    
    def __init__(self):
        super().__init__()
        
        # Application state
        self.eeg_worker: Optional[EEGDataWorker] = None
        self.worker_thread: Optional[QThread] = None
        self.recording = False
        self.recording_data = []
        self.recording_filename = ""
        
        # Stored data mode
        self.current_mode = 'live'  # 'live' or 'stored'
        self.stored_data_manager: Optional[StoredDataManager] = None
        
        # Setup UI
        self.setup_ui()
        self.setup_menus()
        self.setup_status_bar()
        
        # Window properties
        self.setWindowTitle("OpenBCI EEG Data Collection - Demo")
        self.setGeometry(100, 100, 1400, 900)
        
        # Status
        self.update_status("Ready")
    
    def setup_ui(self):
        """Set up the main user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left panel for controls
        left_panel = QVBoxLayout()
        
        # Control panel
        self.control_panel = ControlPanel()
        self.control_panel.connect_requested.connect(self.connect_to_board)
        self.control_panel.disconnect_requested.connect(self.disconnect_from_board)
        self.control_panel.start_recording.connect(self.start_recording)
        self.control_panel.stop_recording.connect(self.stop_recording)
        self.control_panel.mode_changed.connect(self.on_mode_changed)
        self.control_panel.sample_selected.connect(self.on_sample_selected)
        self.control_panel.load_pkl_file.connect(self.load_pkl_file)
        
        left_panel.addWidget(self.control_panel)
        
        # Log display
        log_group = QGroupBox("System Log")
        log_layout = QVBoxLayout()
        self.log_display = QTextEdit()
        self.log_display.setMaximumHeight(200)
        self.log_display.setReadOnly(True)
        log_layout.addWidget(self.log_display)
        
        # Clear log button
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self.log_display.clear)
        log_layout.addWidget(clear_log_btn)
        
        log_group.setLayout(log_layout)
        left_panel.addWidget(log_group)
        
        # Create left panel widget
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setMaximumWidth(300)
        
        # Right panel for plots
        self.plot_widget = RealTimePlotWidget(num_channels=8, window_size=1000)
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(self.plot_widget)
        splitter.setStretchFactor(0, 0)  # Left panel fixed width
        splitter.setStretchFactor(1, 1)  # Plot area expandable
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
    
    def setup_menus(self):
        """Set up the application menus."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        # Export action
        export_action = QAction("Export Data...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_data)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        # Clear plots action
        clear_action = QAction("Clear Plots", self)
        clear_action.triggered.connect(self.plot_widget.clear_plots)
        view_menu.addAction(clear_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        # About action
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_status_bar(self):
        """Set up the status bar."""
        self.status_bar = self.statusBar()
        
        # Connection status
        self.connection_label = QLabel("Disconnected")
        self.connection_label.setStyleSheet("color: red;")
        self.status_bar.addPermanentWidget(self.connection_label)
        
        # Recording status
        self.recording_label = QLabel("Not Recording")
        self.status_bar.addPermanentWidget(self.recording_label)
        
        # Data rate
        self.data_rate_label = QLabel("0 Hz")
        self.status_bar.addPermanentWidget(self.data_rate_label)
    
    def connect_to_board(self):
        """Connect to the OpenBCI board."""
        if self.eeg_worker is not None:
            return
        
        try:
            # Create worker and thread
            self.eeg_worker = EEGDataWorker()
            self.worker_thread = QThread()
            
            # Move worker to thread
            self.eeg_worker.moveToThread(self.worker_thread)
            
            # Connect signals
            self.eeg_worker.data_ready.connect(self.on_data_received)
            self.eeg_worker.connection_status.connect(self.on_connection_status)
            self.eeg_worker.error_occurred.connect(self.on_error_occurred)
            
            # Connect thread signals
            self.worker_thread.started.connect(self.eeg_worker.start_acquisition)
            
            # Start thread
            self.worker_thread.start()
            
            self.log_message("Attempting to connect to OpenBCI board...")
            
        except Exception as e:
            self.log_message(f"Failed to start connection: {e}")
            QMessageBox.critical(self, "Connection Error", f"Failed to connect: {e}")
    
    def disconnect_from_board(self):
        """Disconnect from the OpenBCI board."""
        if self.eeg_worker is None:
            return
        
        try:
            # Stop recording if active
            if self.recording:
                self.stop_recording()
            
            # Stop worker
            self.eeg_worker.stop_acquisition()
            
            # Stop thread
            if self.worker_thread:
                self.worker_thread.quit()
                self.worker_thread.wait(3000)  # Wait up to 3 seconds
                
                if self.worker_thread.isRunning():
                    self.worker_thread.terminate()
                    self.worker_thread.wait()
            
            # Clean up
            self.eeg_worker = None
            self.worker_thread = None
            
            self.log_message("Disconnected from OpenBCI board")
            
        except Exception as e:
            self.log_message(f"Error during disconnection: {e}")
    
    def on_data_received(self, data):
        """Handle new EEG data from the worker thread."""
        try:
            # Update the plot
            self.plot_widget.update_data(data)
            
            # Record data if recording is active
            if self.recording and data is not None:
                # Convert to numpy if needed
                if hasattr(data, 'numpy'):
                    data_np = data.numpy()
                else:
                    data_np = data
                
                # Store data with timestamp
                timestamp = time.time()
                if data_np.ndim == 3:
                    data_np = data_np[0]  # Remove batch dimension
                
                # Store each sample with its timestamp
                for i in range(data_np.shape[1]):
                    sample_data = {
                        'timestamp': timestamp + i / 125.0,  # Assuming 125 Hz sampling rate
                        **{f'ch{j+1}': data_np[j, i] for j in range(data_np.shape[0])}
                    }
                    self.recording_data.append(sample_data)
            
        except Exception as e:
            self.log_message(f"Error processing data: {e}")
    
    def on_connection_status(self, connected, message):
        """Handle connection status updates."""
        if connected:
            self.connection_label.setText("Connected")
            self.connection_label.setStyleSheet("color: green;")
            self.control_panel.set_connection_status(True)
        else:
            self.connection_label.setText("Disconnected")
            self.connection_label.setStyleSheet("color: red;")
            self.control_panel.set_connection_status(False)
        
        self.log_message(message)
    
    def on_error_occurred(self, error_message):
        """Handle error messages from the worker thread."""
        self.log_message(f"ERROR: {error_message}")
        QMessageBox.warning(self, "EEG System Error", error_message)
    
    def start_recording(self, filename):
        """Start recording EEG data to file."""
        if self.recording:
            return
        
        try:
            self.recording = True
            self.recording_filename = filename
            self.recording_data = []
            
            self.control_panel.set_recording_status(True, filename)
            self.recording_label.setText("Recording")
            self.recording_label.setStyleSheet("color: red;")
            
            self.log_message(f"Started recording to: {filename}")
            
        except Exception as e:
            self.log_message(f"Failed to start recording: {e}")
            self.recording = False
    
    def stop_recording(self):
        """Stop recording and save data to file."""
        if not self.recording:
            return
        
        try:
            self.recording = False
            
            # Save data to CSV
            if self.recording_data:
                df = pd.DataFrame(self.recording_data)
                df.to_csv(self.recording_filename, index=False)
                self.log_message(f"Saved {len(self.recording_data)} samples to {self.recording_filename}")
            else:
                self.log_message("No data to save")
            
            # Update UI
            self.control_panel.set_recording_status(False)
            self.recording_label.setText("Not Recording")
            self.recording_label.setStyleSheet("color: black;")
            
            # Clear recording data
            self.recording_data = []
            self.recording_filename = ""
            
        except Exception as e:
            self.log_message(f"Failed to save recording: {e}")
    
    def log_message(self, message):
        """Add a message to the log display."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_display.append(formatted_message)
        
        # Auto-scroll to bottom
        cursor = self.log_display.textCursor()
        cursor.movePosition(cursor.End)
        self.log_display.setTextCursor(cursor)
    
    def update_status(self, message):
        """Update the status bar message."""
        self.status_bar.showMessage(message)
    
    def on_mode_changed(self, mode: str):
        """Handle mode change between live and stored data."""
        self.current_mode = mode
        
        if mode == 'live':
            self.log_message("Switched to Live EEG Device mode")
            # Clear stored data if any
            self.stored_data_manager = None
        else:  # stored mode
            self.log_message("Switched to Stored Data mode")
            # Disconnect from live board if connected
            if self.eeg_worker is not None:
                self.disconnect_from_board()
    
    def load_pkl_file(self, filename: str):
        """Load a pickle file containing stored EEG data."""
        try:
            self.log_message(f"Loading pickle file: {filename}")
            
            # Create stored data manager
            self.stored_data_manager = StoredDataManager(filename)
            
            num_samples = self.stored_data_manager.get_num_samples()
            num_channels = self.stored_data_manager.num_channels
            
            self.log_message(f"Loaded {num_samples} samples with {num_channels} channels")
            
            # Update control panel
            self.control_panel.set_pkl_file_loaded(filename, num_samples)
            
            # Update plot widget if needed
            if num_channels != self.plot_widget.num_channels:
                # Recreate plot widget with correct number of channels
                # For now, just log a message
                self.log_message(f"Note: Plot configured for {self.plot_widget.num_channels} channels, data has {num_channels}")
            
            # Display first sample
            self.on_sample_selected(0)
            
        except Exception as e:
            self.log_message(f"Error loading pickle file: {e}")
            QMessageBox.critical(self, "Load Error", f"Failed to load pickle file:\n{str(e)}")
    
    def on_sample_selected(self, sample_index: int):
        """Handle sample selection in stored data mode."""
        if self.current_mode != 'stored' or self.stored_data_manager is None:
            return
        
        try:
            # Get the selected sample
            sample_data = self.stored_data_manager.get_sample(sample_index)
            
            # Update the plot
            self.plot_widget.update_data(sample_data)
            
            self.log_message(f"Displaying sample {sample_index + 1}")
            
        except Exception as e:
            self.log_message(f"Error displaying sample: {e}")
    
    def export_data(self):
        """Export current buffer data to file."""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, 
                "Export Data", 
                f"eeg_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if filename:
                # Get current buffer data
                buffer_data = self.plot_widget.data_buffer
                
                # Create DataFrame
                df = pd.DataFrame(
                    buffer_data.T,
                    columns=[f'ch{i+1}' for i in range(buffer_data.shape[0])]
                )
                df['time'] = np.linspace(0, len(df)/125.0, len(df))
                
                # Save to file
                df.to_csv(filename, index=False)
                self.log_message(f"Exported data to: {filename}")
                
        except Exception as e:
            self.log_message(f"Failed to export data: {e}")
            QMessageBox.critical(self, "Export Error", f"Failed to export data: {e}")
    
    def show_about(self):
        """Show the about dialog."""
        QMessageBox.about(
            self,
            "About OpenBCI EEG Demo",
            "OpenBCI EEG Data Collection Demo\n\n"
            "A PyQt-based application for real-time EEG data\n"
            "collection and visualization using OpenBCI hardware\n"
            "or stored .pkl data files.\n\n"
            "Features:\n"
            "• Real-time multi-channel EEG visualization\n"
            "• Stored data playback from .pkl files\n"
            "• Sample-by-sample navigation (125 Hz, 1-sec)\n"
            "• Data recording and export\n"
            "• Signal processing controls\n"
            "• Connection management\n\n"
            "Built with PyQt5 and PyQtGraph"
        )
    
    def closeEvent(self, event):
        """Handle application close event."""
        if self.recording:
            reply = QMessageBox.question(
                self,
                "Recording Active",
                "Recording is currently active. Stop recording and exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.stop_recording()
            else:
                event.ignore()
                return
        
        # Disconnect from board
        self.disconnect_from_board()
        
        # Accept the close event
        event.accept()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("OpenBCI EEG Demo")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Longhorn Neurotech")
    
    # Create and show main window
    window = OpenBCIMainWindow()
    window.show()
    
    # Run the application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()