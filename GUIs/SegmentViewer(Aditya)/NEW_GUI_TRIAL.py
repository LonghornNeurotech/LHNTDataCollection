"""
Minimal EEG Segment Viewer
Displays segmented EEG data with channel selection and window navigation
"""

import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QCheckBox, QSlider, QSpinBox, QPushButton, QLabel, QGroupBox,
    QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QRadioButton, QButtonGroup,
    QFileDialog, QMessageBox, QComboBox, QLineEdit, QTabWidget, QMenu, QWidgetAction, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QPalette, QColor, QKeySequence
from PyQt5.QtWidgets import QShortcut
import platform
import time
from datetime import datetime
import os 
from pylsl import StreamInfo, StreamOutlet, local_clock

# Serial port imports
try:
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("Warning: pyserial not available. Install with: pip install pyserial")

# Signal processing imports
try:
    from scipy.signal import butter, lfilter, lfilter_zi, iirnotch
    from scipy.ndimage import uniform_filter1d
    from scipy import signal as sp_signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Install with: pip install scipy")

# LSL imports
try:
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False
    print("Warning: pylsl not available. Install with: pip install pylsl")

# BrainFlow imports
try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False
    print("Warning: BrainFlow not available. Install with: pip install brainflow")

# Import your data loading functions
from conversions import get_gdf_array, get_pkl_array


def find_headset_port():
    """
    Automatically search for headset on typical COM ports (COM3, COM4) and others.
    Returns the port name if found, None otherwise.
    """
    if not SERIAL_AVAILABLE:
        print("Warning: pyserial not available for port scanning")
        return None

    system = platform.system()
    ports = list(serial.tools.list_ports.comports())
    print(f"Scanning available ports: {[p.device for p in ports]}")

    # Priority ports for Windows
    priority_ports = ['COM3', 'COM4']

    # Check priority ports first
    for port_name in priority_ports:
        for port in ports:
            if port.device == port_name:
                if 'CH340' in port.description or 'USB' in port.description or 'Serial' in port.description:
                    print(f"Found headset on priority port: {port.device}")
                    return port.device

    # Check all other ports
    for port in ports:
        if port.device not in priority_ports:
            if 'CH340' in port.description or 'USB' in port.description:
                print(f"Found potential headset: {port.device}")
                if system == "Darwin":  # macOS
                    if any(identifier in port.device.lower() for identifier in ["usbserial", "cu.usbmodem", "tty.usbserial"]):
                        return port.device
                elif system == "Windows":
                    if "com" in port.device.lower():
                        return port.device
                elif system == "Linux":
                    if "ttyUSB" in port.device or "ttyACM" in port.device:
                        return port.device

    print("No headset detected on any port.")
    return None


class SettingsDialog(QDialog):
    """Dialog for configuring application settings"""
    def __init__(self, current_mode, current_theme, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(400)

        layout = QVBoxLayout()

        # Theme selection
        theme_group = QGroupBox("Theme")
        theme_layout = QVBoxLayout()

        self.theme_button_group = QButtonGroup()

        self.light_theme_radio = QRadioButton("Light Mode")
        self.dark_theme_radio = QRadioButton("Dark Mode")

        self.theme_button_group.addButton(self.light_theme_radio)
        self.theme_button_group.addButton(self.dark_theme_radio)

        if current_theme == 'light':
            self.light_theme_radio.setChecked(True)
        else:
            self.dark_theme_radio.setChecked(True)

        theme_layout.addWidget(self.light_theme_radio)
        theme_layout.addWidget(self.dark_theme_radio)
        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group)

        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def get_mode(self):
        """Return selected display mode"""
        return 'stacked'

    def get_theme(self):
        """Return selected theme"""
        return 'light' if self.light_theme_radio.isChecked() else 'dark'


class SegmentViewer(QMainWindow):
    def __init__(self, window_size_sec=10.0, sampling_rate=125):
        super().__init__()
        # Enable OpenGL acceleration and antialiasing for smoother rendering
        pg.setConfigOptions(useOpenGL=True, antialias=True)

        self.raw_data = None  # No data loaded initially
        self.window_size_sec = window_size_sec
        self.sampling_rate = sampling_rate
        self.display_mode = 'stacked'  # 'overlay' or 'stacked'
        self.stacked_plot_item = None  # Created in setup_stacked_mode()
        self.file_loaded = False
        self.current_filename = ""
        self.theme = 'dark'  # 'light' or 'dark'

        # Initialize with no segmentation
        self.segmented = None
        self.num_windows = 0
        self.num_channels = 0
        self.num_samples = 0
        self.current_window = 0
        self.active_channels = set()
        self.last_stacked_channels = set()  # Track which channels are in stacked layout
        self.channel_names = []  # Channel names from file metadata (empty = use Ch1, Ch2, ...)

        # Track last processed horizontal zoom value to avoid expensive updates
        self.last_horizontal_zoom_value = None

        # Vertical bounds mode: 'iqr' or 'minmax'
        self.vertical_bounds_mode = 'iqr'

        # Y-Range for stacked mode (amplitude scale per channel) - mne-lsl style
        # This is the vertical spacing between channels AND the amplitude range shown
        self.yRange = 100.0  # Default 100 uV scale
        # Total vertical span for auto-spacing: divided among active channels
        self.total_y_span = 1600.0
        # Amplitude scale factor for stacked mode (1.0 = normal, >1 = zoomed in)
        # This scales signal amplitude WITHIN each channel lane without changing lane spacing
        self.channel_amplitude_scale = 1.0

        # Available yRange presets for combo box
        self.yRange_presets = {
            '1 uV': 1.0, '10 uV': 10.0, '50 uV': 50.0, '100 uV': 100.0,
            '200 uV': 200.0, '500 uV': 500.0, '1 mV': 1000.0, '5 mV': 5000.0,
            '10 mV': 10000.0, '50 mV': 50000.0, '100 mV': 100000.0
        }

        # Channel colors (theme-dependent, generated by generate_channel_colors)
        self.channel_colors = None
        self.generate_channel_colors()

        # Pre-computed channel means for fast stacked mode rendering
        self.channel_means = None

        # Flag to prevent double data updates during autoplay
        self._skip_range_update = False

        # Autoplay state
        self.autoplay_active = False
        self.autoplay_speed = 1.0  # 1x = real-time
        self.autoplay_timer = QTimer()
        self.autoplay_timer.timeout.connect(self.autoplay_step)
        self.autoplay_current_time = 0.0  # Current time position in seconds

        # Streaming mode variables
        self.mode = 'file'  # 'file' or 'stream'
        self.streaming_active = False
        self.stream_first_update = True  # Flag for initial auto-range on first streaming update
        self.board = None
        self.board_id = None
        self.eeg_channels = None
        self.eeg_outlet = None
        self.marker_outlet = None
        self.stream_timer = QTimer()
        self.stream_timer.timeout.connect(self.update_stream_data)
        self.stream_buffer = None
        self.stream_time_axis = None
        self.stream_start_time = 0
        self.patient_id = ""
        self.trial_number = 1

        # Signal processing parameters
        self.lowcut = 5.0
        self.highcut = 35.0
        self.notch_freq = 60.0
        self.magnitude_scale = 100  # microvolts
        # OpenBCI-style smooth: moving average window in seconds applied to display
        self.smooth_seconds = 0.12  # Default 120ms window
        self.smoothing_enabled = True

        # Pre-calculated filter coefficients (will be computed when sampling rate is known)
        self.filter_coeffs_valid = False
        self.bandpass_b = None
        self.bandpass_a = None
        self.notch_b = None
        self.notch_a = None
        # Persistent filter states per channel for seamless chunk-to-chunk filtering
        self.bandpass_zi = {}  # channel_idx -> filter state
        self.notch_zi = {}    # channel_idx -> filter state

        # Streaming plot item references (reused to avoid memory leak)
        self.streaming_plot_items = {}  # Channel index -> PlotDataItem
        self.fft_plot_items = {}  # Channel index -> PlotDataItem
        self.band_power_bar_item = None  # Single BarGraphItem for band power

        # FFT and band power tracking
        self.fft_data = {}
        self.band_power_data = {
            'delta': {},  # 0.5-4 Hz
            'theta': {},  # 4-8 Hz
            'alpha': {},  # 8-13 Hz
            'beta': {},   # 13-30 Hz
            'gamma': {}   # 30-50 Hz
        }
        # Throttle FFT/band power updates (these are expensive, no need at 60Hz)
        self.fft_update_counter = 0
        self.fft_update_interval = 3  # Update FFT every 3rd frame (~20Hz)

        # Smoothing for FFT and band power
        # Higher alpha = more responsive to current data; frequency-domain smoothing handles visual smoothness
        self.fft_smoothing_alpha = 0.2
        self.band_power_smoothing_alpha = 0.2
        self.smoothed_fft = {}  # Stores smoothed FFT for each channel
        self.smoothed_band_power = {}  # Stores smoothed band power
        self.band_power_y_max = 0.0  # Stable Y-axis max for band power
        
        self.setWindowTitle("EEG Viewer - No File Loaded")
        self.setGeometry(100, 100, 1600, 900)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout_container = QHBoxLayout(main_widget)

        # Left panel for controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(350)
        left_panel_layout = QVBoxLayout(left_panel)

        # Mode selection
        mode_group = QGroupBox("Mode Selection")
        mode_layout = QVBoxLayout()
        self.file_mode_radio = QRadioButton("File Mode")
        self.stream_mode_radio = QRadioButton("Streaming Mode")
        self.file_mode_radio.setChecked(True)
        self.file_mode_radio.toggled.connect(self.on_mode_changed)
        mode_layout.addWidget(self.file_mode_radio)
        mode_layout.addWidget(self.stream_mode_radio)
        mode_group.setLayout(mode_layout)
        left_panel_layout.addWidget(mode_group)

        # File loading section
        self.file_group = QGroupBox("File Loading")
        file_layout = QVBoxLayout()
        load_file_btn = QPushButton("Load File")
        load_file_btn.clicked.connect(self.load_file)
        file_layout.addWidget(load_file_btn)

        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("color: gray; font-style: italic;")
        file_layout.addWidget(self.file_label)
        self.file_group.setLayout(file_layout)
        left_panel_layout.addWidget(self.file_group)

        # Streaming controls section
        self.stream_group = QGroupBox("Streaming Controls")
        stream_layout = QVBoxLayout()

        # Auto-connect button
        self.connect_btn = QPushButton("Auto-Connect Headset")
        self.connect_btn.clicked.connect(self.auto_connect_headset)
        stream_layout.addWidget(self.connect_btn)

        self.connection_status = QLabel("Not connected")
        self.connection_status.setStyleSheet("color: red;")
        stream_layout.addWidget(self.connection_status)

        # Patient ID and Trial number
        patient_layout = QHBoxLayout()
        patient_layout.addWidget(QLabel("Patient ID:"))
        self.patient_id_input = QLineEdit()
        self.patient_id_input.setPlaceholderText("e.g., P001")
        patient_layout.addWidget(self.patient_id_input)
        stream_layout.addLayout(patient_layout)

        trial_layout = QHBoxLayout()
        trial_layout.addWidget(QLabel("Trial:"))
        self.trial_spinbox = QSpinBox()
        self.trial_spinbox.setRange(1, 9999)
        self.trial_spinbox.setValue(1)
        trial_layout.addWidget(self.trial_spinbox)
        stream_layout.addLayout(trial_layout)

        # LSL Stream controls
        stream_layout.addWidget(QLabel("LSL Streams:"))
        self.start_eeg_stream_btn = QPushButton("Start EEG Stream")
        self.start_eeg_stream_btn.clicked.connect(self.start_eeg_stream)
        self.start_eeg_stream_btn.setEnabled(False)
        stream_layout.addWidget(self.start_eeg_stream_btn)

        self.start_marker_stream_btn = QPushButton("Start Marker Stream")
        self.start_marker_stream_btn.clicked.connect(self.start_marker_stream)
        self.start_marker_stream_btn.setEnabled(False)
        stream_layout.addWidget(self.start_marker_stream_btn)

        self.eeg_stream_status = QLabel("EEG Stream: Inactive")
        self.eeg_stream_status.setStyleSheet("color: gray;")
        stream_layout.addWidget(self.eeg_stream_status)

        self.marker_stream_status = QLabel("Marker Stream: Inactive")
        self.marker_stream_status.setStyleSheet("color: gray;")
        stream_layout.addWidget(self.marker_stream_status)

        # Start/Stop visualization
        self.start_viz_btn = QPushButton("Start Visualization")
        self.start_viz_btn.clicked.connect(self.toggle_streaming_visualization)
        self.start_viz_btn.setEnabled(False)
        stream_layout.addWidget(self.start_viz_btn)

        # Motor imagery task button (placeholder)
        self.motor_imagery_btn = QPushButton("Motor Imagery Task")
        self.motor_imagery_btn.clicked.connect(self.motor_imagery_task_placeholder)
        self.motor_imagery_btn.setEnabled(False)
        self.motor_imagery_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        stream_layout.addWidget(self.motor_imagery_btn)

        self.stream_group.setLayout(stream_layout)
        self.stream_group.setEnabled(False)
        left_panel_layout.addWidget(self.stream_group)

        # Signal processing controls
        self.signal_group = QGroupBox("Signal Processing")
        signal_layout = QVBoxLayout()

        # Magnitude scaling
        mag_layout = QHBoxLayout()
        mag_layout.addWidget(QLabel("Magnitude (µV):"))
        self.magnitude_combo = QComboBox()
        self.magnitude_combo.addItems(['5', '10', '15', '25', '50', '100', '200'])
        self.magnitude_combo.setCurrentText('200')
        self.magnitude_combo.currentTextChanged.connect(self.on_magnitude_changed)
        mag_layout.addWidget(self.magnitude_combo)
        signal_layout.addLayout(mag_layout)

        # OpenBCI-style Smooth control (time-based moving average window)
        smooth_layout = QHBoxLayout()
        self.smooth_checkbox = QCheckBox("Smooth:")
        self.smooth_checkbox.setChecked(True)
        self.smooth_checkbox.stateChanged.connect(self.on_smooth_checkbox_changed)
        smooth_layout.addWidget(self.smooth_checkbox)
        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setMinimum(1)    # 10 ms
        self.smooth_slider.setMaximum(20)   # 200 ms
        self.smooth_slider.setValue(12)     # Default 120 ms
        self.smooth_slider.valueChanged.connect(self.on_smooth_slider_changed)
        smooth_layout.addWidget(self.smooth_slider)
        self.smooth_label = QLabel("120 ms")
        self.smooth_label.setMinimumWidth(50)
        smooth_layout.addWidget(self.smooth_label)
        signal_layout.addLayout(smooth_layout)

        self.signal_group.setLayout(signal_layout)
        self.signal_group.setEnabled(False)
        left_panel_layout.addWidget(self.signal_group)

        left_panel_layout.addStretch()
        main_layout_container.addWidget(left_panel)

        # Right side: Main content area
        right_panel = QWidget()
        self.main_layout = QVBoxLayout(right_panel)
        main_layout_container.addWidget(right_panel, stretch=1)
        
        # Channel selection dropdown (initially disabled but visible)
        self.channel_group = QGroupBox("Channel Selection")
        channel_layout = QHBoxLayout()

        # "Select All" checkbox
        self.select_all_checkbox = QCheckBox("Select All")
        self.select_all_checkbox.stateChanged.connect(self.toggle_all_channels)
        channel_layout.addWidget(self.select_all_checkbox)

        # Channel dropdown button
        self.channel_dropdown_btn = QPushButton("Select Channels")
        self.channel_dropdown_menu = QMenu(self)
        self.channel_dropdown_btn.setMenu(self.channel_dropdown_menu)
        channel_layout.addWidget(self.channel_dropdown_btn)

        # Label showing selected channels
        self.selected_channels_label = QLabel("No channels selected")
        channel_layout.addWidget(self.selected_channels_label)
        channel_layout.addStretch()

        self.channel_checkboxes = []
        self.channel_group.setLayout(channel_layout)
        self.channel_group.setEnabled(False)  # Disabled until file loaded
        self.main_layout.addWidget(self.channel_group)

        # Create tabs for different visualizations
        self.viz_tabs = QTabWidget()

        # Main EEG plot tab
        main_plot_tab = QWidget()
        main_plot_layout = QVBoxLayout(main_plot_tab)
        main_plot_layout.setContentsMargins(0, 0, 0, 0)

        # Create a container for plot area with zoom sliders
        plot_and_zoom_widget = QWidget()
        plot_and_zoom_layout = QHBoxLayout(plot_and_zoom_widget)
        plot_and_zoom_layout.setContentsMargins(0, 0, 0, 0)
        
        # Set minimum height for plot area to ensure controls are accessible
        plot_and_zoom_widget.setMinimumHeight(400)
        
        # Vertical zoom controls on the left
        vertical_zoom_widget = QWidget()
        vertical_zoom_layout = QVBoxLayout(vertical_zoom_widget)
        vertical_zoom_layout.setContentsMargins(0, 0, 0, 0)
        
        # Vertical zoom slider (Y-axis zoom)
        self.vertical_zoom_slider = QSlider(Qt.Vertical)
        self.vertical_zoom_slider.setMinimum(10)  # 10% zoom (zoomed out)
        self.vertical_zoom_slider.setMaximum(500)  # 500% zoom (zoomed in)
        self.vertical_zoom_slider.setValue(100)  # 100% = normal
        self.vertical_zoom_slider.setTickPosition(QSlider.TicksRight)
        self.vertical_zoom_slider.setTickInterval(50)
        self.vertical_zoom_slider.valueChanged.connect(self.on_vertical_zoom_slider_changed)
        self.vertical_zoom_slider.setEnabled(False)  # Disabled until file loaded
        vertical_zoom_layout.addWidget(self.vertical_zoom_slider, stretch=1)
        
        # Vertical zoom spinbox for manual input (with decimal support)
        self.vertical_zoom_spinbox = QDoubleSpinBox()
        self.vertical_zoom_spinbox.setRange(0.01, 10000.0)  # Allow much wider range than slider, including tiny values
        self.vertical_zoom_spinbox.setSingleStep(1.0)
        self.vertical_zoom_spinbox.setDecimals(2)  # Allow 2 decimal places
        self.vertical_zoom_spinbox.setValue(100.0)
        self.vertical_zoom_spinbox.setSuffix("%")
        self.vertical_zoom_spinbox.setMaximumWidth(80)
        self.vertical_zoom_spinbox.setEnabled(False)  # Disabled until file loaded
        self.vertical_zoom_spinbox.valueChanged.connect(self.on_vertical_zoom_spinbox_changed)
        vertical_zoom_layout.addWidget(self.vertical_zoom_spinbox)
        
        # Vertical bounds mode dropdown
        self.bounds_mode_combo = QComboBox()
        self.bounds_mode_combo.addItems(["IQR Bounds", "Min-Max Bounds"])
        self.bounds_mode_combo.setCurrentIndex(0)  # Start with IQR
        self.bounds_mode_combo.setMaximumWidth(120)
        self.bounds_mode_combo.setToolTip("Select vertical axis bounds calculation method")
        self.bounds_mode_combo.currentIndexChanged.connect(self.on_bounds_mode_changed)
        self.bounds_mode_combo.setEnabled(False)  # Disabled until file loaded
        vertical_zoom_layout.addWidget(self.bounds_mode_combo)
        
        plot_and_zoom_layout.addWidget(vertical_zoom_widget)
        
        # Right side: plot container + horizontal zoom slider
        plot_column_widget = QWidget()
        plot_column_layout = QVBoxLayout(plot_column_widget)
        plot_column_layout.setContentsMargins(0, 0, 0, 0)
        
        # Plot container (will be replaced when switching modes)
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.plot_layout.setSpacing(0)  # Remove spacing between plots for tight stacking
        self.plot_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        plot_column_layout.addWidget(self.plot_container, stretch=1)  # Give plot area most of the space
        
        # Horizontal zoom slider (X-axis/time zoom) on the bottom
        horizontal_zoom_widget = QWidget()
        horizontal_zoom_layout = QHBoxLayout(horizontal_zoom_widget)
        horizontal_zoom_layout.setContentsMargins(0, 0, 0, 0)
        
        horizontal_zoom_layout.addWidget(QLabel("Window Length:"))
        self.horizontal_zoom_slider = QSlider(Qt.Horizontal)
        self.horizontal_zoom_slider.setMinimum(10)  # 0.1 seconds minimum
        self.horizontal_zoom_slider.setMaximum(600)  # 60.0 seconds maximum
        self.horizontal_zoom_slider.setValue(int(self.window_size_sec * 10))  # Scale by 10 for finer control
        self.horizontal_zoom_slider.setTickPosition(QSlider.TicksBelow)
        self.horizontal_zoom_slider.setTickInterval(50)
        self.horizontal_zoom_slider.valueChanged.connect(self.on_horizontal_zoom_changed)
        self.horizontal_zoom_slider.setEnabled(False)  # Disabled until file loaded
        self.horizontal_zoom_slider.setMaximumWidth(300)  # Shrink slider width
        horizontal_zoom_layout.addWidget(self.horizontal_zoom_slider)
        
        # Window size spinbox (replaces label)
        self.window_size_spinbox = QDoubleSpinBox()
        self.window_size_spinbox.setRange(0.1, 60.0)
        self.window_size_spinbox.setSingleStep(0.1)
        self.window_size_spinbox.setValue(self.window_size_sec)
        self.window_size_spinbox.setDecimals(1)
        self.window_size_spinbox.setSuffix(" s")
        self.window_size_spinbox.setMaximumWidth(80)
        self.window_size_spinbox.valueChanged.connect(self.on_window_size_spinbox_changed)
        self.window_size_spinbox.setEnabled(False)  # Disabled until file loaded
        horizontal_zoom_layout.addWidget(self.window_size_spinbox)
        
        # Sampling rate spinbox
        horizontal_zoom_layout.addWidget(QLabel("  Sampling Rate:"))
        self.sampling_rate_spinbox = QSpinBox()
        self.sampling_rate_spinbox.setRange(1, 10000)
        self.sampling_rate_spinbox.setSingleStep(1)
        self.sampling_rate_spinbox.setValue(self.sampling_rate)
        self.sampling_rate_spinbox.setSuffix(" Hz")
        self.sampling_rate_spinbox.setMaximumWidth(100)
        self.sampling_rate_spinbox.valueChanged.connect(self.on_sampling_rate_changed)
        self.sampling_rate_spinbox.setEnabled(False)  # Disabled until file loaded
        horizontal_zoom_layout.addWidget(self.sampling_rate_spinbox)
        
        # Play/Pause button (replaces autoplay toggle)
        self.play_pause_btn = QPushButton("▶ Play")
        self.play_pause_btn.clicked.connect(self.on_spacebar)
        self.play_pause_btn.setEnabled(False)  # Disabled until file loaded
        self.play_pause_btn.setMaximumWidth(130)
        self.play_pause_btn.setFocusPolicy(Qt.ClickFocus)
        horizontal_zoom_layout.addWidget(self.play_pause_btn)

        # Playback speed
        horizontal_zoom_layout.addWidget(QLabel("Speed:"))
        self.playback_speed_spinbox = QDoubleSpinBox()
        self.playback_speed_spinbox.setRange(0.1, 10.0)
        self.playback_speed_spinbox.setSingleStep(0.1)
        self.playback_speed_spinbox.setValue(1.0)
        self.playback_speed_spinbox.setDecimals(1)
        self.playback_speed_spinbox.setSuffix("x")
        self.playback_speed_spinbox.setMaximumWidth(80)
        self.playback_speed_spinbox.valueChanged.connect(self.on_playback_speed_changed)
        horizontal_zoom_layout.addWidget(self.playback_speed_spinbox)

        horizontal_zoom_layout.addStretch()  # Push everything to the left
        
        plot_column_layout.addWidget(horizontal_zoom_widget)
        
        plot_and_zoom_layout.addWidget(plot_column_widget, stretch=1)

        main_plot_layout.addWidget(plot_and_zoom_widget, stretch=1)
        self.viz_tabs.addTab(main_plot_tab, "EEG Data")

        # FFT tab
        fft_tab = QWidget()
        fft_layout = QVBoxLayout(fft_tab)
        self.fft_plot_widget = pg.PlotWidget()
        self.fft_plot_widget.setLabel('bottom', 'Frequency (Hz)')
        self.fft_plot_widget.setLabel('left', 'Power (dB)')
        self.fft_plot_widget.setTitle('Real-time FFT')
        self.fft_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        fft_layout.addWidget(self.fft_plot_widget)
        self.viz_tabs.addTab(fft_tab, "FFT")

        # Band Power tab
        band_power_tab = QWidget()
        band_power_layout = QVBoxLayout(band_power_tab)
        self.band_power_plot_widget = pg.PlotWidget()
        self.band_power_plot_widget.setLabel('bottom', 'Frequency Band')
        self.band_power_plot_widget.setLabel('left', 'Power')
        self.band_power_plot_widget.setTitle('Real-time Band Power')
        self.band_power_plot_widget.showGrid(x=False, y=True, alpha=0.3)
        band_power_layout.addWidget(self.band_power_plot_widget)
        self.viz_tabs.addTab(band_power_tab, "Band Power")

        self.main_layout.addWidget(self.viz_tabs, stretch=1)
        
        # Create initial "no file" message
        self.no_file_label = QLabel("No file loaded\n\nClick 'Load File' to begin")
        self.no_file_label.setAlignment(Qt.AlignCenter)
        self.no_file_label.setStyleSheet("font-size: 24px; color: gray;")
        self.plot_layout.addWidget(self.no_file_label)
        
        # Window navigation (file mode only - hidden in streaming mode)
        self.nav_widget = QWidget()
        nav_layout = QHBoxLayout(self.nav_widget)

        # Slider
        nav_layout.addWidget(QLabel("Window:"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.on_slider_changed)
        nav_layout.addWidget(self.slider)

        # SpinBox for direct input
        self.spinbox = QSpinBox()
        self.spinbox.setMinimum(0)
        self.spinbox.setMaximum(0)
        self.spinbox.setValue(0)
        self.spinbox.valueChanged.connect(self.on_spinbox_changed)
        nav_layout.addWidget(self.spinbox)

        # Previous/Next buttons
        self.prev_btn = QPushButton("◄ Prev")
        self.prev_btn.clicked.connect(self.prev_window)
        nav_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next ►")
        self.next_btn.clicked.connect(self.next_window)
        nav_layout.addWidget(self.next_btn)

        # Window info label
        self.info_label = QLabel("No file loaded")
        nav_layout.addWidget(self.info_label)

        self.nav_widget.setEnabled(False)  # Disabled until file loaded
        self.main_layout.addWidget(self.nav_widget)

        # Tools widget (always visible in both file and streaming modes)
        self.tools_widget = QWidget()
        tools_layout = QHBoxLayout(self.tools_widget)
        tools_layout.setContentsMargins(0, 0, 0, 0)

        tools_layout.addStretch()

        # Auto-fit button
        self.autofit_btn = QPushButton("Auto-Fit")
        self.autofit_btn.clicked.connect(self.auto_fit_plot)
        tools_layout.addWidget(self.autofit_btn)

        # Settings button
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(self.open_settings_dialog)
        tools_layout.addWidget(self.settings_btn)

        self.main_layout.addWidget(self.tools_widget)

        # Apply initial theme
        self.apply_theme()

        # Buttons accept click focus (so clicking them deselects spinboxes) but
        # the QShortcut below intercepts spacebar before any button can process it
        for btn in self.findChildren(QPushButton):
            btn.setFocusPolicy(Qt.ClickFocus)

        # Global spacebar shortcut for play/pause or start/stop streaming
        spacebar_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        spacebar_shortcut.activated.connect(self.on_spacebar)

        # Set aspect ratio constraints to preserve display ratio
        self.setMinimumSize(QSize(1200, 675))  # 16:9 ratio
        self.resize(1600, 900)  # Default 16:9 size

    def load_file(self):
        """Open file dialog to load .gdf or .pkl file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open EEG File",
            "",
            "EEG Files (*.gdf *.pkl);;GDF Files (*.gdf);;Pickle Files (*.pkl);;All Files (*)"
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            # Load data based on file extension
            if file_path.endswith('.gdf'):
                self.raw_data, file_metadata = get_gdf_array(file_path)
            elif file_path.endswith('.pkl'):
                self.raw_data, file_metadata = get_pkl_array(file_path)
            else:
                QMessageBox.warning(self, "Error", "Unsupported file type. Please select a .gdf or .pkl file.")
                return

            # Apply metadata from file (sampling rate, channel names)
            if 'sfreq' in file_metadata:
                self.sampling_rate = int(file_metadata['sfreq'])
                self.sampling_rate_spinbox.blockSignals(True)
                self.sampling_rate_spinbox.setValue(self.sampling_rate)
                self.sampling_rate_spinbox.blockSignals(False)
            if 'ch_names' in file_metadata:
                self.channel_names = file_metadata['ch_names']
            else:
                self.channel_names = []

            # Reset smoothing buffers and plot items to prevent shape mismatch with previous stream/file
            self.smoothed_fft = {}
            self.smoothed_band_power = {}
            self.band_power_y_max = 0.0
            self.fft_plot_items.clear()
            self.band_power_bar_item = None

            # Update UI with loaded file
            self.current_filename = file_path.split('/')[-1]
            self.file_label.setText(f"Loaded: {self.current_filename}")
            self.file_label.setStyleSheet("color: green; font-weight: bold;")
            self.file_loaded = True
            
            # Prepare data and calculate window parameters
            self.prepare_data()
            self.current_window = 0

            # Update window length slider/spinbox maximum based on file duration
            file_duration_sec = self.raw_data.shape[1] / self.sampling_rate
            self.horizontal_zoom_slider.setMaximum(int(file_duration_sec * 10))  # Scale by 10
            self.window_size_spinbox.setMaximum(file_duration_sec)

            # Initialize horizontal zoom tracking
            self.last_horizontal_zoom_value = int(self.window_size_sec * 10)
            
            # Update window title
            self.setWindowTitle(f"Segment Viewer - {self.current_filename} - {self.num_windows} windows, {self.num_channels} channels")
            
            # Hide "no file" label before setting up plots (which will delete it)
            if hasattr(self, 'no_file_label') and self.no_file_label is not None:
                try:
                    self.no_file_label.setVisible(False)
                except RuntimeError:
                    pass  # Widget already deleted
            
            # Setup plot based on current display mode (this will clear plot_layout)
            if self.display_mode == 'overlay':
                self.setup_overlay_mode()
            else:
                self.setup_stacked_mode()
            
            # Setup channel checkboxes in dropdown
            self.channel_dropdown_menu.clear()
            self.channel_checkboxes = []
            self.active_channels.clear()  # Reset active channels

            for i in range(self.num_channels):
                cb = QCheckBox(self.get_channel_name(i))
                cb.stateChanged.connect(lambda state, ch=i: self.toggle_channel(ch, state))
                self.channel_checkboxes.append(cb)

                # Add checkbox to menu
                action = QWidgetAction(self.channel_dropdown_menu)
                action.setDefaultWidget(cb)
                self.channel_dropdown_menu.addAction(action)

            # Enable channel group
            self.channel_group.setEnabled(True)

            # Enable zoom sliders and spinboxes
            self.vertical_zoom_slider.setEnabled(True)
            self.vertical_zoom_spinbox.setEnabled(True)
            self.bounds_mode_combo.setEnabled(True)
            self.horizontal_zoom_slider.setEnabled(True)
            self.window_size_spinbox.setEnabled(True)
            self.sampling_rate_spinbox.setEnabled(True)
            self.play_pause_btn.setEnabled(True)  # Enable play button
            
            # Update navigation controls
            self.slider.setMaximum(self.num_windows - 1)
            self.spinbox.setMaximum(self.num_windows - 1)
            self.slider.setValue(0)
            self.spinbox.setValue(0)
            self.info_label.setText(f"Window 0 / {self.num_windows-1}")
            
            # Enable navigation
            self.nav_widget.setEnabled(True)
            
            # Initialize with first channel selected
            self.channel_checkboxes[0].setChecked(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")
    
    
    def setup_overlay_mode(self):
        """Setup single plot widget for overlay display"""
        # Clear existing plots
        while self.plot_layout.count():
            child = self.plot_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Create single plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('bottom', 'Time (seconds)')
        self.plot_widget.setLabel('left', 'Amplitude')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()
        
        # Disable default mouse wheel behavior and add custom zoom
        self.plot_widget.setMouseEnabled(x=True, y=True)
        view_box = self.plot_widget.getViewBox()
        view_box.setMouseMode(pg.ViewBox.RectMode)
        
        # Install custom wheel event using a closure that properly captures view_box
        def make_wheel_handler(vb):
            return lambda event: self.custom_wheel_event(vb, event)
        view_box.wheelEvent = make_wheel_handler(view_box)
        
        # Connect range change signal for dynamic updates
        self.plot_widget.sigRangeChanged.connect(self.on_overlay_range_changed)
        
        self.plot_layout.addWidget(self.plot_widget)
        self.plot_widgets = [self.plot_widget]  # Store in list for consistency
        
        # Dictionary to store plot items for each channel
        self.overlay_plot_items = {}
    
    def setup_stacked_mode(self):
        """Setup single plot widget with vertically offset channels (mne-lsl style)"""
        # Clear existing plots
        while self.plot_layout.count():
            child = self.plot_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Create a custom GraphicsLayoutWidget with wheel scroll signal (like mne-lsl)
        self.plot_widget = pg.GraphicsLayoutWidget()

        # Add the main plot
        self.stacked_plot_item = self.plot_widget.addPlot()
        self.stacked_plot_item.setLabel('bottom', 'Time (s)')
        self.stacked_plot_item.setLabel('left', f'Scale: {self.yRange:.0f} uV')
        self.stacked_plot_item.showGrid(x=True, y=True, alpha=0.3)

        # Disable auto-range for manual control (like mne-lsl)
        self.stacked_plot_item.disableAutoRange()
        self.stacked_plot_item.setMouseEnabled(x=True, y=False)  # Only allow X panning
        self.stacked_plot_item.setMenuEnabled(False)

        # Install wheel event: horizontal scroll = pan time, vertical scroll = amplitude zoom
        view_box = self.stacked_plot_item.getViewBox()
        def wheel_handler(event, axis=None):
            delta = event.delta()
            if delta == 0:
                event.accept()
                return

            # pyqtgraph passes axis=0 for X, axis=1 for Y from AxisItem
            # Also check event orientation for direct wheel events
            if axis == 0:
                orientation = Qt.Horizontal
            elif axis == 1:
                orientation = Qt.Vertical
            elif hasattr(event, 'orientation'):
                orientation = event.orientation()
            else:
                orientation = Qt.Vertical

            if orientation == Qt.Horizontal:
                # Horizontal scroll: pan time axis
                x_range = view_box.viewRange()[0]
                x_size = x_range[1] - x_range[0]
                pan_amount = -delta * 0.1 * x_size / 120.0
                view_box.setXRange(x_range[0] + pan_amount, x_range[1] + pan_amount, padding=0)
            else:
                # Vertical scroll: adjust per-channel amplitude scale
                notches = delta // 120
                if notches != 0:
                    scale_factor = 1.2
                    if notches > 0:
                        self.channel_amplitude_scale *= scale_factor
                    else:
                        self.channel_amplitude_scale /= scale_factor
                    # Clamp to reasonable bounds
                    self.channel_amplitude_scale = max(0.001, min(self.channel_amplitude_scale, 10000))
                    # Sync the vertical zoom slider/spinbox to match
                    slider_val = int(self.channel_amplitude_scale * 100)
                    self.vertical_zoom_slider.blockSignals(True)
                    self.vertical_zoom_spinbox.blockSignals(True)
                    self.vertical_zoom_slider.setValue(max(10, min(500, slider_val)))
                    self.vertical_zoom_spinbox.setValue(slider_val)
                    self.vertical_zoom_slider.blockSignals(False)
                    self.vertical_zoom_spinbox.blockSignals(False)
                    # Redraw — route to correct update function
                    if self.mode == 'stream' and self.streaming_active:
                        self.update_streaming_plots()
                    elif self.file_loaded:
                        self.update_plot_stacked(rebuild_layout=False)
            event.accept()
        view_box.wheelEvent = wheel_handler

        # Connect range change signal for X-axis panning
        self.stacked_plot_item.sigXRangeChanged.connect(self.on_stacked_range_changed)

        self.plot_layout.addWidget(self.plot_widget)
        self.plot_widgets = [self.stacked_plot_item]

        # Dictionary to store plot items for each channel
        self.stacked_plot_items = {}
    
    def generate_channel_colors(self):
        """Generate channel colors based on current theme"""
        np.random.seed(42)
        n = 64
        if self.theme == 'dark':
            # Warm palette (red-biased): bright on dark background
            self.channel_colors = np.column_stack([
                np.random.uniform(180, 255, n),
                np.random.uniform(50, 180, n),
                np.random.uniform(30, 100, n),
            ])
        else:
            # Cool palette (blue-biased): darker on light background
            self.channel_colors = np.column_stack([
                np.random.uniform(20, 100, n),
                np.random.uniform(40, 130, n),
                np.random.uniform(140, 220, n),
            ])

    def prepare_data(self):
        """Prepare data and calculate window parameters and vertical bounds"""
        self.recalculate_windows()

        # Pre-compute per-channel means for fast stacked-mode centering
        self.channel_means = np.mean(self.raw_data, axis=1)

        # Pre-calculate both IQR and min-max bounds for entire dataset (per channel)
        # This is expensive so only do it once when file is loaded
        self.channel_iqr_bounds = {}
        self.channel_minmax_bounds = {}
        
        for ch_idx in range(self.num_channels):
            # Get all data for this channel
            if self.raw_data is None: return
            channel_data = self.raw_data[ch_idx, :]
            
            # IQR bounds
            q1 = np.quantile(channel_data, 0.25)
            q3 = np.quantile(channel_data, 0.75)
            iqr = q3 - q1
            iqr_y_min = q1 - iqr * 5.0
            iqr_y_max = q3 + iqr * 5.0
            self.channel_iqr_bounds[ch_idx] = (iqr_y_min, iqr_y_max)
            
            # Min-Max bounds
            minmax_y_min = np.min(channel_data)
            minmax_y_max = np.max(channel_data)
            self.channel_minmax_bounds[ch_idx] = (minmax_y_min, minmax_y_max)
    
    def get_channel_name(self, ch_idx):
        """Get display name for a channel, using file metadata if available"""
        if self.channel_names and ch_idx < len(self.channel_names):
            return self.channel_names[ch_idx]
        return f"Ch{ch_idx + 1}"

    def compute_channel_spacing(self, num_active):
        """Auto-calculate channel spacing d = total_y_span / (num_active + 1).

        For N channels there are N+1 equal gaps: one above the top channel,
        one between each adjacent pair, and one below the bottom channel.
        """
        if num_active <= 0:
            return self.total_y_span
        return self.total_y_span / (num_active + 1)

    def recalculate_windows(self):
        """Recalculate window parameters (fast, no IQR calculation)"""
        # Calculate window parameters
        self.window_size_samples = int(self.window_size_sec * self.sampling_rate)
        self.stride = int(self.window_size_samples * 0.8)  # 20% overlap
        
        # Calculate number of windows (at least 1 if there's any data)
        total_samples = self.raw_data.shape[1]
        self.num_windows = max(1, (total_samples - self.window_size_samples) // self.stride + 1)
        self.num_channels = self.raw_data.shape[0]
        self.num_samples_per_window = self.window_size_samples
        
        # Create full time axis for entire dataset
        self.full_time_axis = np.arange(total_samples) / self.sampling_rate
    
    def get_window_from_time(self, t_center):
        """Calculate which window index corresponds to a given time point
        
        Args:
            t_center: Time in seconds (typically center of visible range)
            
        Returns:
            Window index (0 to num_windows-1)
        """
        # Convert time to sample index
        sample_idx = int(t_center * self.sampling_rate)
        
        # Calculate which window this sample belongs to
        # Window i starts at sample: i * stride
        # Window i ends at sample: i * stride + window_size_samples
        window_idx = sample_idx // self.stride
        
        # Clamp to valid range
        window_idx = max(0, min(window_idx, self.num_windows - 1))
        
        return window_idx
    
    def update_window_controls_from_view(self):
        """Update slider/spinbox to match the current view range"""
        # Skip window control updates in streaming mode (no window navigation)
        if self.mode == 'stream' or not self.file_loaded:
            return
        
        # Get current view range based on display mode
        if self.display_mode == 'overlay':
            view_range = self.plot_widget.viewRange()[0]  # [x_min, x_max]
        else:
            # In stacked mode, use stacked_plot_item
            if self.stacked_plot_item is None:
                return
            view_range = self.stacked_plot_item.viewRange()[0]
        
        # Calculate center of visible range
        t_center = (view_range[0] + view_range[1]) / 2.0
        
        # Find corresponding window
        new_window = self.get_window_from_time(t_center)
        
        # Update controls without triggering callbacks
        if new_window != self.current_window:
            self.slider.blockSignals(True)
            self.spinbox.blockSignals(True)
            self.current_window = new_window
            self.slider.setValue(new_window)
            self.spinbox.setValue(new_window)
            self.info_label.setText(f"Window {new_window} / {self.num_windows-1}")
            self.slider.blockSignals(False)
            self.spinbox.blockSignals(False)
    
    def get_window_bounds(self, window_idx):
        """Get start and end sample indices for a given window, clamped to valid range"""
        total_samples = len(self.full_time_axis)
        start_sample = max(0, window_idx * self.stride)
        end_sample = min(start_sample + self.window_size_samples, total_samples)
        return start_sample, end_sample
    
    def get_iqr_bounds_for_channels(self, channel_indices):
        """Get the maximum IQR bounds across specified channels"""
        if not channel_indices:
            return (0, 1)  # Default if no channels
        
        # Find channel with greatest range
        max_range = 0
        best_bounds = None
        for ch_idx in channel_indices:
            y_min, y_max = self.channel_iqr_bounds[ch_idx]
            range_size = y_max - y_min
            if range_size > max_range:
                max_range = range_size
                best_bounds = (y_min, y_max)
        
        return best_bounds
    
    def get_minmax_bounds_for_channels(self, channel_indices):
        """Get the maximum min-max bounds across specified channels"""
        if not channel_indices:
            return (0, 1)  # Default if no channels
        
        # Find channel with greatest range
        max_range = 0
        best_bounds = None
        for ch_idx in channel_indices:
            y_min, y_max = self.channel_minmax_bounds[ch_idx]
            range_size = y_max - y_min
            if range_size > max_range:
                max_range = range_size
                best_bounds = (y_min, y_max)
        
        return best_bounds
    
    def get_bounds_for_channels(self, channel_indices):
        """Get bounds based on current vertical bounds mode"""
        if self.vertical_bounds_mode == 'iqr':
            return self.get_iqr_bounds_for_channels(channel_indices)
        else:  # 'minmax'
            return self.get_minmax_bounds_for_channels(channel_indices)
    
    def get_channel_bounds(self, ch_idx):
        """Get bounds for a single channel based on current mode"""
        if self.vertical_bounds_mode == 'iqr':
            return self.channel_iqr_bounds[ch_idx]
        else:  # 'minmax'
            return self.channel_minmax_bounds[ch_idx]
    
    def on_bounds_mode_changed(self, index):
        """Handle bounds mode dropdown selection change
        
        Args:
            index: 0 for IQR, 1 for Min-Max
        """
        if not self.file_loaded:
            return
        
        # Update mode based on selection
        if index == 0:
            self.vertical_bounds_mode = 'iqr'
        else:
            self.vertical_bounds_mode = 'minmax'
        
        # Reapply current zoom level with new bounds
        current_zoom = self.vertical_zoom_spinbox.value()
        self.apply_vertical_zoom(current_zoom)
    
    def custom_wheel_event(self, view_box, event):
        """Custom mouse wheel handler:
        - Respects setMouseEnabled settings
        - Horizontal scroll (left/right): Pan time axis
        - Vertical scroll: Disabled (to prevent accidental zooming)

        Note: pyqtgraph uses QGraphicsSceneWheelEvent, not QWheelEvent, which has
        different methods (delta() and orientation() instead of angleDelta()).
        """
        # Check if x-axis mouse interaction is enabled
        # ViewBox.state['mouseEnabled'] is a list [x_bool, y_bool]
        if not view_box.state['mouseEnabled'][0]:
            event.accept()
            return

        # Get delta value (magnitude of scroll)
        delta = event.delta()
        
        if delta == 0:
            event.accept()
            return
        
        # Check orientation
        if hasattr(event, 'orientation'):
            orientation = event.orientation()
        else:
            orientation = Qt.Vertical
        
        if orientation == Qt.Horizontal:
            # Horizontal scroll: Pan time (X) axis left/right
            # Get current view range
            view_range = view_box.viewRange()
            x_range = view_range[0]  # [x_min, x_max]
            x_size = x_range[1] - x_range[0]
            
            # Pan sensitivity: 10% of visible range per scroll unit
            pan_factor = 0.1
            
            # Positive delta = scroll right = move view left (show earlier data)
            # Negative delta = scroll left = move view right (show later data)
            pan_amount = -delta * pan_factor * x_size / 120.0  # Normalize delta (typically ±120 per scroll step)
            new_x_min = x_range[0] + pan_amount
            new_x_max = x_range[1] + pan_amount
            view_box.setXRange(new_x_min, new_x_max, padding=0)
        # else: ignore vertical scroll
        
        event.accept()
    
    def apply_vertical_zoom(self, value):
        """Apply vertical zoom to the plot (Y-axis zoom)

        Args:
            value: Zoom percentage (any positive value, where 100 = normal)
        """
        if not self.active_channels:
            return

        # Get the zoom factor (1.0 = 100%)
        zoom_factor = value / 100.0

        if self.display_mode == 'overlay':
            if self.mode == 'stream' and self.streaming_active:
                # In streaming mode, zoom based on magnitude scale setting
                base_range = self.magnitude_scale  # microvolts
                y_range = base_range / zoom_factor
                self.plot_widget.setYRange(-y_range, y_range, padding=0)
            elif self.file_loaded:
                # File mode: use pre-calculated channel bounds
                y_min, y_max = self.get_bounds_for_channels(self.active_channels)
                y_center = (y_min + y_max) / 2.0
                y_range = (y_max - y_min) / zoom_factor

                new_y_min = y_center - y_range / 2.0
                new_y_max = y_center + y_range / 2.0
                self.plot_widget.setYRange(new_y_min, new_y_max, padding=0)
        else:
            # Stacked mode: scale signal amplitude within each channel lane
            # Lane spacing stays fixed; signals get bigger/smaller within their lanes
            self.channel_amplitude_scale = zoom_factor

            # Redraw with new scale — route to the correct update function
            if self.stacked_plot_item is not None:
                if self.mode == 'stream' and self.streaming_active:
                    self.update_streaming_plots()
                elif self.file_loaded:
                    self.update_plot_stacked(rebuild_layout=False)
    
    def on_vertical_zoom_slider_changed(self, value):
        """Handle vertical zoom slider changes
        
        Args:
            value: Slider value (10-500)
        """
        # Update spinbox to match slider
        self.vertical_zoom_spinbox.blockSignals(True)
        self.vertical_zoom_spinbox.setValue(value)
        self.vertical_zoom_spinbox.blockSignals(False)
        
        # Apply the zoom
        self.apply_vertical_zoom(value)
    
    def on_vertical_zoom_spinbox_changed(self, value):
        """Handle vertical zoom spinbox changes
        
        Args:
            value: Spinbox value (1-10000)
        """
        # Update slider to match spinbox (clamped to slider range)
        slider_value = int(max(10, min(500, value)))
        self.vertical_zoom_slider.blockSignals(True)
        self.vertical_zoom_slider.setValue(slider_value)
        self.vertical_zoom_slider.blockSignals(False)
        
        # Apply the zoom with the actual spinbox value (not clamped)
        self.apply_vertical_zoom(value)
    
    def sync_navigation_controls(self):
        """
        Updates slider/spinbox max values and values after num_windows has changed.
        This assumes self.recalculate_windows() or self.prepare_data() has 
        already been called.
        """
        # 1. Clamp the internal current_window variable to be valid
        self.current_window = max(0, min(self.current_window, self.num_windows - 1))
            
        # 2. Block signals to prevent infinite loops (e.g., slider changing spinbox)
        self.slider.blockSignals(True)
        self.spinbox.blockSignals(True)
        
        # 3. Set the new maximums
        self.slider.setMaximum(self.num_windows - 1)
        self.spinbox.setMaximum(self.num_windows - 1)
        
        # 4. Re-assert the (potentially clamped) current window value
        #    This is the crucial step that forces the slider to update its visual handle
        self.slider.setValue(self.current_window)
        self.spinbox.setValue(self.current_window)
        
        # 5. Unblock signals
        self.slider.blockSignals(False)
        self.spinbox.blockSignals(False)
        
        # 6. Update labels
        self.info_label.setText(f"Window {self.current_window} / {self.num_windows-1}")
        self.setWindowTitle(f"Segment Viewer - {self.current_filename} - {self.num_windows} windows, {self.num_channels} channels")

    def on_horizontal_zoom_changed(self, value):
        """Handle horizontal zoom slider changes (time window size)

        Args:
            value: Window size in tenths of seconds (10-600, representing 0.1-60.0 seconds)
        """
        # Convert slider value to seconds (slider is scaled by 10 for finer control)
        new_window_size = value / 10.0

        # Update the spinbox
        self.window_size_spinbox.blockSignals(True)
        self.window_size_spinbox.setValue(new_window_size)
        self.window_size_spinbox.blockSignals(False)

        # Update window size
        self.window_size_sec = new_window_size

        # In streaming mode, resize the buffer to match the new window length
        if self.mode == 'stream' and self.streaming_active:
            self.resize_stream_buffer(new_window_size)
            return

        # File mode processing below
        if not self.file_loaded:
            return

        # Stop autoplay if active
        if self.autoplay_active and self.autoplay_timer.isActive():
            self.autoplay_timer.stop()
            self.play_pause_btn.setText("▶ Play")

        # Only process if change is significant (> 1 unit = 0.1 seconds)
        # This avoids expensive recalculations on every tiny slider movement
        if self.last_horizontal_zoom_value is not None:
            if abs(value - self.last_horizontal_zoom_value) < 1:
                return

        # Store this value as the last processed one
        self.last_horizontal_zoom_value = value

        # Get current view center BEFORE changing window size
        if self.display_mode == 'overlay':
            view_range = self.plot_widget.viewRange()[0]
        elif self.stacked_plot_item is not None:
            view_range = self.stacked_plot_item.viewRange()[0]
        else:
            return

        t_center = (view_range[0] + view_range[1]) / 2.0

        # Recalculate windows with new size
        self.recalculate_windows()

        # Update navigation controls and re-calculate windows
        self.sync_navigation_controls()

        # Calculate new view range centered on the same time point
        t_start = t_center - (new_window_size / 2.0)
        t_end = t_center + (new_window_size / 2.0)

        # Clamp to valid time range
        t_start = max(0, t_start)
        t_end = min(self.full_time_axis[-1], t_end)

        # Adjust center if we hit a boundary
        if t_start == 0:
            t_end = min(new_window_size, self.full_time_axis[-1])
        elif t_end == self.full_time_axis[-1]:
            t_start = max(0, self.full_time_axis[-1] - new_window_size)

        # Set the new view range (this will trigger range_changed signal which updates window controls)
        if self.display_mode == 'overlay':
            self.plot_widget.setXRange(t_start, t_end, padding=0)
        else:
            self.stacked_plot_item.setXRange(t_start, t_end, padding=0)
    
    def on_window_size_spinbox_changed(self, value):
        """Handle window size spinbox changes"""
        # Update slider to match
        slider_value = int(value * 10)
        self.horizontal_zoom_slider.blockSignals(True)
        self.horizontal_zoom_slider.setValue(slider_value)
        self.horizontal_zoom_slider.blockSignals(False)

        # Update window size
        self.window_size_sec = value

        # In streaming mode, resize the buffer to match the new window length
        if self.mode == 'stream' and self.streaming_active:
            self.resize_stream_buffer(value)
            return

        # File mode processing below
        if not self.file_loaded:
            return

        # Stop autoplay if active
        if self.autoplay_active and self.autoplay_timer.isActive():
            self.autoplay_timer.stop()
            self.play_pause_btn.setText("▶ Play")

        # Update tracking variable
        self.last_horizontal_zoom_value = slider_value

        # Get current view center BEFORE changing window size
        if self.display_mode == 'overlay':
            view_range = self.plot_widget.viewRange()[0]
        elif self.stacked_plot_item is not None:
            view_range = self.stacked_plot_item.viewRange()[0]
        else:
            return

        t_center = (view_range[0] + view_range[1]) / 2.0

        # Recalculate windows with new size
        self.recalculate_windows()

        # Update navigation controls and re-calculate windows
        self.sync_navigation_controls()

        # Calculate new view range centered on the same time point
        t_start = t_center - (value / 2.0)
        t_end = t_center + (value / 2.0)

        # Clamp to valid time range
        t_start = max(0, t_start)
        t_end = min(self.full_time_axis[-1], t_end)

        # Adjust center if we hit a boundary
        if t_start == 0:
            t_end = min(value, self.full_time_axis[-1])
        elif t_end == self.full_time_axis[-1]:
            t_start = max(0, self.full_time_axis[-1] - value)

        # Set the new view range (this will trigger range_changed signal which updates window controls)
        if self.display_mode == 'overlay':
            self.plot_widget.setXRange(t_start, t_end, padding=0)
        else:
            self.stacked_plot_item.setXRange(t_start, t_end, padding=0)
    
    def on_sampling_rate_changed(self, value):
        """Handle sampling rate spinbox changes"""
        if not self.file_loaded:
            return
        
        self.sampling_rate = value
        
        # Recalculate everything including IQR bounds (sampling rate affects data interpretation)
        self.prepare_data() # This also calls recalculate_windows()
        
        # Update UI
        self.sync_navigation_controls()
        
        # Refresh plot
        self.update_plot()
    
    def toggle_channel(self, channel_idx, state):
        """Toggle channel visibility"""
        if not self.file_loaded and not self.streaming_active:
            return
        if state == Qt.Checked:
            self.active_channels.add(channel_idx)
        else:
            self.active_channels.discard(channel_idx)

        # Update "Select All" checkbox state
        self.select_all_checkbox.blockSignals(True)
        if len(self.active_channels) == self.num_channels:
            self.select_all_checkbox.setChecked(True)
        elif len(self.active_channels) == 0:
            self.select_all_checkbox.setChecked(False)
        else:
            self.select_all_checkbox.setCheckState(Qt.PartiallyChecked)
        self.select_all_checkbox.blockSignals(False)

        self.update_selected_channels_label()

        # Update plots for file mode only - streaming mode updates via timer
        if self.file_loaded and not self.streaming_active:
            self.update_plot()
            self.calculate_fft_for_file()
            self.calculate_band_power_for_file()

    def update_plot(self, rebuild=True):
        """Redraw plot with active channels and IQR-based bounds
        
        Args:
            rebuild: If False, only update data without rebuilding plot structure (faster)
        """
        if not self.file_loaded:
            return
        if self.display_mode == 'overlay':
            self.update_plot_overlay(set_y_range=rebuild)
        else:
            self.update_plot_stacked(rebuild_layout=rebuild)
    
    def get_visible_data_range(self, view_range):
        """Get the sample indices for the visible time range, with some padding"""
        t_min, t_max = view_range
        # Add padding (show 20% extra on each side for smooth scrolling)
        padding = (t_max - t_min) * 0.2
        t_min_padded = max(0, t_min - padding)
        t_max_padded = min(self.full_time_axis[-1], t_max + padding)
        
        # Convert time to sample indices
        start_idx = max(0, int(t_min_padded * self.sampling_rate))
        end_idx = min(len(self.full_time_axis), int(t_max_padded * self.sampling_rate))
        
        return start_idx, end_idx
    
    def update_plot_overlay(self, set_y_range=True):
        """Update plot in overlay mode - plots only visible data
        
        Args:
            set_y_range: If True, update Y-axis bounds. Set to False when just changing windows.
        """
        # Clear existing plot items
        self.overlay_plot_items.clear()
        self.plot_widget.clear()
        
        if not self.active_channels:
            return
        
        # Get current window bounds
        start_sample, end_sample = self.get_window_bounds(self.current_window)
        t_start = self.full_time_axis[start_sample]
        t_end = self.full_time_axis[min(end_sample - 1, len(self.full_time_axis) - 1)]
        
        # Get visible data range with padding
        view_range = [t_start, t_end]
        start_idx, end_idx = self.get_visible_data_range(view_range)
        
        # Plot only visible data for each active channel
        for ch_idx in sorted(self.active_channels):
            data = self.raw_data[ch_idx, start_idx:end_idx]
            time_slice = self.full_time_axis[start_idx:end_idx]
            color = pg.mkColor(self.channel_colors[ch_idx % len(self.channel_colors)])

            plot_item = self.plot_widget.plot(
                time_slice,
                data,
                pen=pg.mkPen(color=color, width=2),
                name=self.get_channel_name(ch_idx)
            )
            self.overlay_plot_items[ch_idx] = plot_item
        
        # Only set Y range when channels change, not on every window update
        if set_y_range:
            y_min, y_max = self.get_bounds_for_channels(self.active_channels)
            self.plot_widget.setYRange(y_min, y_max, padding=0)
        
        # Set X-axis range to show current window
        self.plot_widget.setXRange(t_start, t_end, padding=0)
        
        self.plot_widget.addLegend()
    
    def on_overlay_range_changed(self):
        """Called when user zooms or pans in overlay mode - updates visible data"""
        if self.mode == 'stream' or self._skip_range_update:
            return

        if not self.file_loaded or not self.active_channels:
            return

        # Update window controls to match current view
        self.update_window_controls_from_view()

        # Get current view range
        view_range = self.plot_widget.viewRange()[0]  # [x_min, x_max]
        start_idx, end_idx = self.get_visible_data_range(view_range)

        # Update data for each active channel
        for ch_idx in sorted(self.active_channels):
            if ch_idx in self.overlay_plot_items:
                data = self.raw_data[ch_idx, start_idx:end_idx]
                time_slice = self.full_time_axis[start_idx:end_idx]
                self.overlay_plot_items[ch_idx].setData(time_slice, data)
    
    def update_plot_stacked(self, rebuild_layout=True):
        """Update plots in stacked mode - mne-lsl style with vertical offsets.

        Each channel is offset by -k * d, where d is auto-calculated spacing.
        d = total_y_span / num_active, with d/2 padding from top and bottom edges.

        Args:
            rebuild_layout: If True, rebuild plot items (when channels change).
                          If False, just update data (faster for navigation/zoom).
        """
        if not self.file_loaded or self.stacked_plot_item is None:
            return

        active_list = sorted(self.active_channels)
        num_active = len(active_list)

        if num_active == 0:
            self.stacked_plot_item.clear()
            self.stacked_plot_items.clear()
            return

        # Auto-calculate channel spacing
        d = self.compute_channel_spacing(num_active)
        self.yRange = d  # Keep yRange in sync for other code paths

        # Get window time bounds
        start_sample, end_sample = self.get_window_bounds(self.current_window)
        t_start = self.full_time_axis[start_sample]
        t_end = self.full_time_axis[min(end_sample - 1, len(self.full_time_axis) - 1)]

        # Get visible data range
        view_range = [t_start, t_end]
        start_idx, end_idx = self.get_visible_data_range(view_range)

        # Create time array for X-axis
        time_slice = self.full_time_axis[start_idx:end_idx]

        # Calculate vertical offsets: 0, -d, -2d, ...
        offsets = np.arange(0, -num_active * d, -d)

        # Vectorized: extract all channels and compute means in one numpy call
        active_indices = np.array(active_list)
        all_data = self.raw_data[active_indices, start_idx:end_idx]
        all_means = np.mean(all_data, axis=1)

        # Check if we need to rebuild (channels changed)
        if rebuild_layout or self.active_channels != self.last_stacked_channels:
            # Clear existing plot items
            self.stacked_plot_item.clear()
            self.stacked_plot_items.clear()

            # Create plot items for each channel
            for k, ch_idx in enumerate(active_list):
                offset_data = (all_data[k] - all_means[k]) * self.channel_amplitude_scale + offsets[k]

                # Get color for this channel
                color = pg.mkColor(self.channel_colors[ch_idx % len(self.channel_colors)])

                plot_item = self.stacked_plot_item.plot(
                    time_slice,
                    offset_data,
                    pen=pg.mkPen(color=color, width=1.5),
                )
                # Enable auto-downsampling for performance with many channels
                plot_item.setDownsampling(auto=True, method='peak')
                plot_item.setClipToView(True)
                self.stacked_plot_items[ch_idx] = plot_item

            # Set Y-axis ticks to show channel labels
            yticks = [
                (-k * d, self.get_channel_name(ch_idx))
                for k, ch_idx in enumerate(active_list)
            ]
            self.stacked_plot_item.getAxis('left').setTicks([yticks, []])

            # Remember which channels are displayed
            self.last_stacked_channels = self.active_channels.copy()

        else:
            # Just update data for existing plot items (fast path)
            for k, ch_idx in enumerate(active_list):
                if ch_idx in self.stacked_plot_items:
                    offset_data = (all_data[k] - all_means[k]) * self.channel_amplitude_scale + offsets[k]
                    self.stacked_plot_items[ch_idx].setData(time_slice, offset_data)

            # Update Y-axis ticks (in case spacing changed)
            yticks = [
                (-k * d, self.get_channel_name(ch_idx))
                for k, ch_idx in enumerate(active_list)
            ]
            self.stacked_plot_item.getAxis('left').setTicks([yticks, []])

        # Update axis label with current scale
        self.stacked_plot_item.setLabel('left', f'Scale: {d:.1f}')

        # Set Y range: d/2 above topmost channel, d/2 below bottommost
        y_top = d
        y_bottom = -num_active * d
        self.stacked_plot_item.setYRange(y_bottom, y_top, padding=0)

        # Set X range to current window
        self.stacked_plot_item.setXRange(t_start, t_end, padding=0)
    
    def on_stacked_range_changed(self):
        """Called when user pans in stacked mode - updates visible data"""
        if self.mode == 'stream' or self._skip_range_update:
            return

        if not self.file_loaded or not self.active_channels or self.stacked_plot_item is None:
            return

        # Update window controls to match current view
        self.update_window_controls_from_view()

        # Get the X range from plot
        view_range = self.stacked_plot_item.viewRange()[0]  # [x_min, x_max]
        start_idx, end_idx = self.get_visible_data_range(view_range)

        # Get time slice
        time_slice = self.full_time_axis[start_idx:end_idx]

        # Calculate offsets with auto-spacing
        active_list = sorted(self.active_channels)
        num_active = len(active_list)
        d = self.compute_channel_spacing(num_active)
        offsets = np.arange(0, -num_active * d, -d)

        # Vectorized: extract all channels and compute means in one numpy call
        active_indices = np.array(active_list)
        all_data = self.raw_data[active_indices, start_idx:end_idx]
        all_means = np.mean(all_data, axis=1)

        # Update data for all active channels
        for k, ch_idx in enumerate(active_list):
            if ch_idx in self.stacked_plot_items:
                offset_data = (all_data[k] - all_means[k]) * self.channel_amplitude_scale + offsets[k]
                self.stacked_plot_items[ch_idx].setData(time_slice, offset_data)

    def navigate_to_window(self, value):
        """Navigate to a specific window and update the view

        Args:
            value: Window index to navigate to
        """
        # Skip in streaming mode (no window-based navigation)
        if self.mode == 'stream':
            return

        # Stop autoplay if user manually navigates
        if self.autoplay_active and self.autoplay_timer.isActive():
            self.autoplay_timer.stop()
            self.play_pause_btn.setText("▶ Play")
        
        self.current_window = value
        
        self.slider.setValue(value)
        self.spinbox.setValue(value)
        
        self.info_label.setText(f"Window {value} / {self.num_windows-1}")
        
        # Snap to window boundaries
        start_sample, end_sample = self.get_window_bounds(value)
        t_start = self.full_time_axis[start_sample]
        t_end = self.full_time_axis[min(end_sample - 1, len(self.full_time_axis) - 1)]
        
        # Set X-axis range to new window
        if self.display_mode == 'overlay':
            self.plot_widget.setXRange(t_start, t_end, padding=0)
        else:
            if self.stacked_plot_item is not None:
                self.stacked_plot_item.setXRange(t_start, t_end, padding=0)
        
        # Update plot data and apply zoom
        self.update_plot()

        # Update FFT and band power for file mode
        if self.file_loaded and not self.streaming_active:
            self.calculate_fft_for_file()
            self.calculate_band_power_for_file()
    
    def on_slider_changed(self, value):
        """Handle slider movement"""
        self.navigate_to_window(value)
    
    def on_spinbox_changed(self, value):
        """Handle spinbox input"""
        self.navigate_to_window(value)
    
    def prev_window(self):
        """Go to previous window and snap to its boundaries"""
        if self.current_window > 0:
            self.spinbox.setValue(self.current_window - 1)
    
    def next_window(self):
        """Go to next window and snap to its boundaries"""
        if self.current_window < self.num_windows - 1:
            self.spinbox.setValue(self.current_window + 1)
    
    def auto_fit_plot(self):
        """Re-fit the plot to current window's IQR bounds and snap to window boundaries"""
        if not self.file_loaded or not self.active_channels:
            return
        
        # Reset vertical zoom slider to 100% (normal zoom)
        self.vertical_zoom_slider.setValue(100)
        
        # Get window time bounds for current window
        start_sample, end_sample = self.get_window_bounds(self.current_window)
        t_start = self.full_time_axis[start_sample]
        t_end = self.full_time_axis[min(end_sample - 1, len(self.full_time_axis) - 1)]
        
        # Get pre-calculated bounds based on current mode from channel with greatest range
        y_min, y_max = self.get_bounds_for_channels(self.active_channels)
        
        if self.display_mode == 'overlay':
            # Reset to show current window time range and bounds
            self.plot_widget.setYRange(y_min, y_max, padding=0)
            self.plot_widget.setXRange(t_start, t_end, padding=0)
        else:
            # Stacked mode: reset total_y_span and amplitude scale
            if self.stacked_plot_item is not None:
                self.total_y_span = 1600.0  # Reset to default
                self.channel_amplitude_scale = 1.0
                self.stacked_plot_item.setXRange(t_start, t_end, padding=0)
                num_active = len(self.active_channels)
                d = self.compute_channel_spacing(num_active)
                y_top = d
                y_bottom = -num_active * d
                self.stacked_plot_item.setYRange(y_bottom, y_top, padding=0)

    def on_spacebar(self):
        """Handle spacebar: dispatches to file play/pause or streaming start/stop"""
        if self.mode == 'stream':
            self.toggle_streaming_visualization()
        else:
            self.toggle_play_pause()

    def toggle_play_pause(self):
        """Toggle play/pause - single button controls autoplay (file mode only)"""
        if not self.file_loaded:
            return

        if self.autoplay_timer.isActive():
            # Currently playing - pause it
            self.autoplay_timer.stop()
            self.play_pause_btn.setText("▶ Play")
        else:
            # Start or resume playing from current view position
            self.autoplay_active = True
            view_range = self.stacked_plot_item.viewRange()
            self.autoplay_current_time = (view_range[0][0] + view_range[0][1]) / 2
            self._last_frame_time = time.perf_counter()
            self.autoplay_timer.start(33)
            self.play_pause_btn.setText("⏸ Pause")
    
    def on_playback_speed_changed(self, value):
        """Update playback speed"""
        self.autoplay_speed = value
    
    def autoplay_step(self):
        """Advance autoplay by one frame using real elapsed time"""
        if not self.file_loaded or not self.autoplay_active or not self.active_channels:
            return

        # Use real elapsed time so playback stays at true speed regardless of frame rate
        now = time.perf_counter()
        elapsed = now - self._last_frame_time
        self._last_frame_time = now

        # Cap elapsed to avoid huge jumps (e.g. after window unfocus)
        elapsed = min(elapsed, 0.1)

        # Advance by real elapsed time × speed multiplier
        time_increment = elapsed * self.autoplay_speed
        self.autoplay_current_time += time_increment

        # Get total duration
        total_duration = self.full_time_axis[-1]

        # Check if we've reached the end
        if self.autoplay_current_time >= total_duration - self.window_size_sec / 2:
            self.autoplay_current_time = self.window_size_sec / 2

        # Calculate the view range centered on current time
        half_window = self.window_size_sec / 2
        t_start = self.autoplay_current_time - half_window
        t_end = self.autoplay_current_time + half_window

        # Clamp to valid range
        if t_start < self.full_time_axis[0]:
            t_start = self.full_time_axis[0]
            t_end = t_start + self.window_size_sec
        if t_end > self.full_time_axis[-1]:
            t_end = self.full_time_axis[-1]
            t_start = t_end - self.window_size_sec

        # Block range-changed signals to prevent double data updates
        self._skip_range_update = True

        # Get visible data range
        view_range = [t_start, t_end]
        start_idx, end_idx = self.get_visible_data_range(view_range)
        time_slice = self.full_time_axis[start_idx:end_idx]

        # Suppress intermediate repaints while updating all curves
        self.plot_widget.setUpdatesEnabled(False)

        if self.display_mode == 'overlay':
            self.plot_widget.setXRange(t_start, t_end, padding=0)
            for ch_idx in sorted(self.active_channels):
                if ch_idx in self.overlay_plot_items:
                    data = self.raw_data[ch_idx, start_idx:end_idx]
                    self.overlay_plot_items[ch_idx].setData(time_slice, data)
        else:
            if self.stacked_plot_item is not None:
                self.stacked_plot_item.setXRange(t_start, t_end, padding=0)

            active_list = sorted(self.active_channels)
            num_active = len(active_list)
            d = self.compute_channel_spacing(num_active)
            offsets = np.arange(0, -num_active * d, -d)

            # Vectorized: extract all channels and compute means in one numpy call
            active_indices = np.array(active_list)
            all_data = self.raw_data[active_indices, start_idx:end_idx]
            all_means = np.mean(all_data, axis=1)

            for k, ch_idx in enumerate(active_list):
                if ch_idx in self.stacked_plot_items:
                    offset_data = (all_data[k] - all_means[k]) * self.channel_amplitude_scale + offsets[k]
                    self.stacked_plot_items[ch_idx].setData(time_slice, offset_data)

        # Re-enable repaints and trigger a single redraw
        self.plot_widget.setUpdatesEnabled(True)
        self.plot_widget.update()

        self._skip_range_update = False

        # Update window controls to reflect current position
        self.update_window_controls_from_view()

        # Throttle FFT/band power updates (~3 Hz)
        self.fft_update_counter += 1
        if self.fft_update_counter >= 5:
            self.fft_update_counter = 0
            self.calculate_fft_for_file()
            self.calculate_band_power_for_file()

    def open_settings_dialog(self):
        """Open dialog to configure application settings"""
        dialog = SettingsDialog(self.display_mode, self.theme, self)
        if dialog.exec_() == QDialog.Accepted:
            new_mode = dialog.get_mode()
            new_theme = dialog.get_theme()

            # Handle theme change
            if new_theme != self.theme:
                self.theme = new_theme
                self.apply_theme()

            # Handle display mode change
            if new_mode != self.display_mode:
                self.display_mode = new_mode

                # Rebuild plot layout
                if self.display_mode == 'overlay':
                    self.setup_overlay_mode()
                else:
                    self.setup_stacked_mode()

                # Refresh plot (will automatically apply zoom)
                self.update_plot()

                print(f"Display mode changed to: {self.display_mode}")
    
    def on_mode_changed(self):
        """Handle mode switching between file and streaming"""
        if self.file_mode_radio.isChecked():
            self.mode = 'file'
            self.file_group.setEnabled(True)
            self.stream_group.setEnabled(False)
            self.signal_group.setEnabled(False)
            # Show bottom navigation toolbar for file mode
            self.nav_widget.setVisible(True)
            # Restore play/pause for file mode
            self.play_pause_btn.setEnabled(self.file_loaded)
            self.play_pause_btn.setText("▶ Play")
            # Enable speed control for file mode
            self.playback_speed_spinbox.setEnabled(True)
            # Enable horizontal scrolling for file mode
            if hasattr(self, 'plot_widgets') and self.plot_widgets:
                for pw in self.plot_widgets:
                    if hasattr(pw, 'setMouseEnabled'):
                        pw.setMouseEnabled(x=True, y=True)
        else:
            self.mode = 'stream'
            self.file_group.setEnabled(False)
            self.stream_group.setEnabled(True)
            self.signal_group.setEnabled(True)
            # Hide bottom navigation toolbar for streaming mode (window navigation not applicable)
            self.nav_widget.setVisible(False)
            # Switch play/pause button to streaming control
            self.play_pause_btn.setEnabled(True)
            self.play_pause_btn.setText("Start Streaming")
            # Disable speed control in streaming mode (real-time only)
            self.playback_speed_spinbox.setEnabled(False)
            # Disable horizontal scrolling for streaming mode
            if hasattr(self, 'plot_widgets') and self.plot_widgets:
                for pw in self.plot_widgets:
                    if hasattr(pw, 'setMouseEnabled'):
                        pw.setMouseEnabled(x=False, y=False)

    def auto_connect_headset(self):
        """Auto-detect and connect to EEG headset"""
        if not BRAINFLOW_AVAILABLE:
            QMessageBox.critical(self, "Error", "BrainFlow is not installed.\nInstall with: pip install brainflow")
            return

        if not SERIAL_AVAILABLE:
            QMessageBox.critical(self, "Error", "pyserial is not installed.\nInstall with: pip install pyserial")
            return

        self.connect_btn.setEnabled(False)
        self.connect_btn.setText("Connecting...")

        try:
            # Find serial port
            serial_port = find_headset_port()

            if serial_port is None:
                response = QMessageBox.question(
                    self,
                    "No Headset Found",
                    "No headset detected. Use synthetic board for testing?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if response == QMessageBox.Yes:
                    self.board_id = BoardIds.SYNTHETIC_BOARD.value
                    params = BrainFlowInputParams()
                else:
                    self.connect_btn.setEnabled(True)
                    self.connect_btn.setText("Auto-Connect Headset")
                    return
            else:
                self.board_id = BoardIds.CYTON_BOARD.value
                params = BrainFlowInputParams()
                params.serial_port = serial_port

            # Initialize board
            BoardShim.enable_dev_board_logger()
            self.board = BoardShim(self.board_id, params)
            self.board.prepare_session()
            self.board.start_stream()

            # Get channel info
            self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
            self.num_channels = len(self.eeg_channels)
            self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)

            # Initialize buffer for streaming using current window length
            buffer_size = int(self.sampling_rate * self.window_size_sec)
            self.stream_buffer = np.zeros((self.num_channels, buffer_size))
            self.stream_time_axis = np.arange(buffer_size) / self.sampling_rate

            # Pre-calculate filter coefficients now that sampling rate is known
            self.calculate_filter_coefficients()

            # Update UI - sync sampling rate spinbox to detected hardware rate
            self.sampling_rate_spinbox.blockSignals(True)
            self.sampling_rate_spinbox.setValue(self.sampling_rate)
            self.sampling_rate_spinbox.blockSignals(False)
            # Lock sampling rate during streaming (hardware determines it)
            self.sampling_rate_spinbox.setEnabled(False)

            self.connection_status.setText(f"Connected: {self.num_channels} channels @ {self.sampling_rate} Hz")
            self.connection_status.setStyleSheet("color: green;")
            self.connect_btn.setText("Disconnect")
            self.connect_btn.clicked.disconnect()
            self.connect_btn.clicked.connect(self.disconnect_headset)
            self.connect_btn.setEnabled(True)

            # Enable stream controls
            self.start_eeg_stream_btn.setEnabled(True)
            self.start_marker_stream_btn.setEnabled(True)
            self.start_viz_btn.setEnabled(True)
            self.motor_imagery_btn.setEnabled(True)

            # Setup channel checkboxes for streaming
            self.setup_streaming_channels()

            QMessageBox.information(self, "Success", f"Connected to headset!\n{self.num_channels} channels @ {self.sampling_rate} Hz")

        except Exception as e:
            QMessageBox.critical(self, "Connection Error", f"Failed to connect to headset:\n{str(e)}")
            self.connect_btn.setEnabled(True)
            self.connect_btn.setText("Auto-Connect Headset")

    def disconnect_headset(self):
        """Disconnect from headset"""
        try:
            if self.streaming_active:
                self.toggle_streaming_visualization()

            if self.board is not None:
                self.board.stop_stream()
                self.board.release_session()
                self.board = None

            if self.eeg_outlet is not None:
                del self.eeg_outlet
                self.eeg_outlet = None

            if self.marker_outlet is not None:
                del self.marker_outlet
                self.marker_outlet = None

            self.connection_status.setText("Not connected")
            self.connection_status.setStyleSheet("color: red;")
            self.connect_btn.setText("Auto-Connect Headset")
            self.connect_btn.clicked.disconnect()
            self.connect_btn.clicked.connect(self.auto_connect_headset)

            self.start_eeg_stream_btn.setEnabled(False)
            self.start_marker_stream_btn.setEnabled(False)
            self.start_viz_btn.setEnabled(False)
            self.motor_imagery_btn.setEnabled(False)

            self.eeg_stream_status.setText("EEG Stream: Inactive")
            self.eeg_stream_status.setStyleSheet("color: gray;")
            self.marker_stream_status.setText("Marker Stream: Inactive")
            self.marker_stream_status.setStyleSheet("color: gray;")

            # Re-enable sampling rate editing
            self.sampling_rate_spinbox.setEnabled(True)

        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error during disconnect:\n{str(e)}")

    def setup_streaming_channels(self):
        """Setup channel checkboxes for streaming mode"""
        self.channel_dropdown_menu.clear()
        self.channel_checkboxes = []
        self.active_channels.clear()

        for i in range(self.num_channels):
            cb = QCheckBox(self.get_channel_name(i))
            cb.stateChanged.connect(lambda state, ch=i: self.toggle_channel(ch, state))
            self.channel_checkboxes.append(cb)

            # Add checkbox to menu
            action = QWidgetAction(self.channel_dropdown_menu)
            action.setDefaultWidget(cb)
            self.channel_dropdown_menu.addAction(action)

        # Enable channel group
        self.channel_group.setEnabled(True)

        # Select all channels by default
        self.select_all_checkbox.setChecked(True)

    def start_eeg_stream(self):
        """Start LSL EEG stream"""
        if not LSL_AVAILABLE:
            QMessageBox.critical(self, "Error", "pylsl is not installed.\nInstall with: pip install pylsl")
            return

        if self.eeg_outlet is not None:
            QMessageBox.information(self, "Info", "EEG stream already active")
            return

        try:
            # Get patient ID
            self.patient_id = self.patient_id_input.text().strip()
            if not self.patient_id:
                QMessageBox.warning(self, "Warning", "Please enter a Patient ID")
                return

            self.trial_number = self.trial_spinbox.value()

            # Create stream name with auto-naming
            stream_name = f"EEG_{self.patient_id}_Trial{self.trial_number:04d}"

            # Create LSL stream info
            info = StreamInfo(
                name=stream_name,
                type='EEG',
                channel_count=self.num_channels,
                nominal_srate=self.sampling_rate,
                channel_format='float32',
                source_id=f'eeg_headset_{self.patient_id}'
            )

            # Create outlet
            self.eeg_outlet = StreamOutlet(info)

            self.eeg_stream_status.setText(f"EEG Stream: Active - {stream_name}")
            self.eeg_stream_status.setStyleSheet("color: green; font-weight: bold;")
            self.start_eeg_stream_btn.setText("Stop EEG Stream")
            self.start_eeg_stream_btn.clicked.disconnect()
            self.start_eeg_stream_btn.clicked.connect(self.stop_eeg_stream)

            print(f"LSL EEG Stream started: {stream_name}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start EEG stream:\n{str(e)}")

    def stop_eeg_stream(self):
        """Stop LSL EEG stream"""
        if self.eeg_outlet is not None:
            del self.eeg_outlet
            self.eeg_outlet = None

        self.eeg_stream_status.setText("EEG Stream: Inactive")
        self.eeg_stream_status.setStyleSheet("color: gray;")
        self.start_eeg_stream_btn.setText("Start EEG Stream")
        self.start_eeg_stream_btn.clicked.disconnect()
        self.start_eeg_stream_btn.clicked.connect(self.start_eeg_stream)

    def start_marker_stream(self):
        """Start LSL Marker stream"""
        if not LSL_AVAILABLE:
            QMessageBox.critical(self, "Error", "pylsl is not installed.\nInstall with: pip install pylsl")
            return

        if self.marker_outlet is not None:
            QMessageBox.information(self, "Info", "Marker stream already active")
            return

        try:
            # Get patient ID
            self.patient_id = self.patient_id_input.text().strip()
            if not self.patient_id:
                QMessageBox.warning(self, "Warning", "Please enter a Patient ID")
                return

            self.trial_number = self.trial_spinbox.value()

            # Create stream name with auto-naming
            stream_name = f"Markers_{self.patient_id}_Trial{self.trial_number:04d}"

            # Create LSL stream info for markers
            info = StreamInfo(
                name=stream_name,
                type='Markers',
                channel_count=1,
                nominal_srate=0,  # Irregular rate for markers
                channel_format='string',
                source_id=f'markers_{self.patient_id}'
            )

            # Create outlet
            self.marker_outlet = StreamOutlet(info)

            self.marker_stream_status.setText(f"Marker Stream: Active - {stream_name}")
            self.marker_stream_status.setStyleSheet("color: green; font-weight: bold;")
            self.start_marker_stream_btn.setText("Stop Marker Stream")
            self.start_marker_stream_btn.clicked.disconnect()
            self.start_marker_stream_btn.clicked.connect(self.stop_marker_stream)

            print(f"LSL Marker Stream started: {stream_name}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start marker stream:\n{str(e)}")

    def stop_marker_stream(self):
        """Stop LSL Marker stream"""
        if self.marker_outlet is not None:
            del self.marker_outlet
            self.marker_outlet = None

        self.marker_stream_status.setText("Marker Stream: Inactive")
        self.marker_stream_status.setStyleSheet("color: gray;")
        self.start_marker_stream_btn.setText("Start Marker Stream")
        self.start_marker_stream_btn.clicked.disconnect()
        self.start_marker_stream_btn.clicked.connect(self.start_marker_stream)

    def toggle_streaming_visualization(self):
        """Start or stop real-time visualization"""
        if not self.streaming_active:
            # Start streaming visualization
            self.streaming_active = True
            self.stream_first_update = True  # Flag for initial auto-range
            self.stream_start_time = local_clock() if LSL_AVAILABLE else 0
            self.stream_timer.start(16)  # Update every ~16ms (~60 Hz) for smooth visualization
            self.start_viz_btn.setText("Stop Visualization")
            self.start_viz_btn.setStyleSheet("background-color: #f44336; color: white;")
            self.play_pause_btn.setText("Stop Streaming")

            # Clear all plots and reset references BEFORE starting new visualization
            # This ensures old data is cleared only when new data is about to replace it
            if hasattr(self, 'plot_widget') and self.plot_widget is not None:
                self.plot_widget.clear()
            if hasattr(self, 'fft_plot_widget') and self.fft_plot_widget is not None:
                self.fft_plot_widget.clear()
            if hasattr(self, 'band_power_plot_widget') and self.band_power_plot_widget is not None:
                self.band_power_plot_widget.clear()
            self.streaming_plot_items.clear()
            self.fft_plot_items.clear()
            self.band_power_bar_item = None
            self.smoothed_fft.clear()
            self.smoothed_band_power.clear()
            self.band_power_y_max = 0.0
            # Reset filter states for fresh start
            self.bandpass_zi.clear()
            self.notch_zi.clear()
            # Fill buffer with NaN so unfilled regions aren't drawn
            # (avoids zero-to-data boundary that causes filter transient spikes)
            if self.stream_buffer is not None:
                self.stream_buffer[:] = np.nan
            # Drain any stale accumulated data from the board
            if self.board is not None:
                self.board.get_board_data()
            # Discard the first 0.5s of filtered data (filter startup transient)
            self.stream_warmup_remaining = int(self.sampling_rate * 0.5)

            # Setup visualization
            self.file_loaded = True
            self.current_filename = "Live Stream"
            self.setWindowTitle("EEG Viewer - Live Streaming")

            # Setup plot based on current display mode
            if self.display_mode == 'overlay':
                self.setup_overlay_mode()
            else:
                self.setup_stacked_mode()

            # Disable horizontal scrolling for streaming mode
            if hasattr(self, 'plot_widgets') and self.plot_widgets:
                for pw in self.plot_widgets:
                    if hasattr(pw, 'setMouseEnabled'):
                        pw.setMouseEnabled(x=False, y=False)

            # Enable controls
            self.vertical_zoom_slider.setEnabled(True)
            self.vertical_zoom_spinbox.setEnabled(True)
            self.bounds_mode_combo.setEnabled(True)
            self.horizontal_zoom_slider.setEnabled(True)
            self.window_size_spinbox.setEnabled(True)

            # Disable mode switching while streaming is active
            self.file_mode_radio.setEnabled(False)
            self.stream_mode_radio.setEnabled(False)

        else:
            # Stop streaming visualization
            # Keep graphs visible - they will be cleared when new visualization starts
            self.streaming_active = False
            self.stream_timer.stop()
            self.start_viz_btn.setText("Start Visualization")
            self.start_viz_btn.setStyleSheet("")
            self.play_pause_btn.setText("Start Streaming")
            self.file_loaded = False

            # Re-enable mode switching
            self.file_mode_radio.setEnabled(True)
            self.stream_mode_radio.setEnabled(True)

    def resize_stream_buffer(self, new_window_sec):
        """Resize the streaming buffer when the user changes window length"""
        new_buf_size = int(self.sampling_rate * new_window_sec)
        if new_buf_size < 10:
            new_buf_size = 10
        old_buf = self.stream_buffer
        old_size = old_buf.shape[1]

        # Use NaN for unfilled regions so they aren't drawn (connect='finite')
        new_buf = np.full((self.num_channels, new_buf_size), np.nan)
        # Copy as much old data as fits, aligned to the right
        copy_len = min(old_size, new_buf_size)
        new_buf[:, -copy_len:] = old_buf[:, -copy_len:]

        self.stream_buffer = new_buf
        self.stream_time_axis = np.arange(new_buf_size) / self.sampling_rate
        self.raw_data = self.stream_buffer
        self.num_samples = new_buf_size

        # Reset smoothing buffers since FFT size changes with buffer size
        self.smoothed_fft.clear()
        if hasattr(self, 'fft_plot_items'):
            self.fft_plot_items.clear()
        if hasattr(self, 'fft_plot_widget') and self.fft_plot_widget is not None:
            self.fft_plot_widget.clear()

        # Force plot items to rebuild with new time axis
        self.streaming_plot_items.clear()
        if self.display_mode == 'stacked' and hasattr(self, 'stacked_plot_item') and self.stacked_plot_item is not None:
            self.stacked_plot_item.clear()
        elif hasattr(self, 'plot_widget') and self.plot_widget is not None:
            self.plot_widget.clear()

    def update_stream_data(self):
        """Update visualization with new streaming data - rolling buffer (newest on right)"""
        if self.board is None or not self.streaming_active:
            return

        try:
            # Get new data from board
            data = self.board.get_board_data()

            if data.shape[1] == 0:
                return  # No new data

            # Extract EEG channels
            eeg_data = data[self.eeg_channels, :]

            # Send to LSL if outlet is active - use push_chunk for efficiency
            if self.eeg_outlet is not None and LSL_AVAILABLE:
                chunk = eeg_data.T.tolist()
                self.eeg_outlet.push_chunk(chunk)

            # Apply signal processing (bandpass, notch with stateful filters)
            processed_data = self.process_signal(eeg_data)

            # Discard initial samples while filters settle (avoids startup transient spike)
            if self.stream_warmup_remaining > 0:
                n = processed_data.shape[1]
                self.stream_warmup_remaining -= n
                if self.stream_warmup_remaining >= 0:
                    return  # Still warming up, discard entirely
                # Warmup just finished - keep only the post-warmup tail
                keep = -self.stream_warmup_remaining
                processed_data = processed_data[:, -keep:]
                self.stream_warmup_remaining = 0

            # Rolling buffer: shift left, append new data on right
            new_samples = processed_data.shape[1]
            if new_samples > 0:
                buf_len = self.stream_buffer.shape[1]
                if new_samples >= buf_len:
                    self.stream_buffer = processed_data[:, -buf_len:]
                else:
                    self.stream_buffer[:, :-new_samples] = self.stream_buffer[:, new_samples:]
                    self.stream_buffer[:, -new_samples:] = processed_data

                self.raw_data = self.stream_buffer
                self.num_samples = buf_len

                # Update plots (every frame for smooth time-domain display)
                self.update_streaming_plots()

                # Throttle FFT and band power updates (expensive, ~20Hz is enough)
                self.fft_update_counter += 1
                if self.fft_update_counter >= self.fft_update_interval:
                    self.fft_update_counter = 0
                    self.update_fft()
                    self.update_band_power()

        except Exception as e:
            print(f"Error updating stream: {str(e)}")

    def calculate_filter_coefficients(self):
        """Pre-calculate filter coefficients for efficient signal processing"""
        if not SCIPY_AVAILABLE:
            self.filter_coeffs_valid = False
            return

        try:
            # Bandpass filter coefficients (5-35 Hz)
            self.bandpass_b, self.bandpass_a = butter(
                2, [self.lowcut, self.highcut], btype='band', fs=self.sampling_rate
            )
            # Notch filter coefficients (60 Hz)
            self.notch_b, self.notch_a = iirnotch(self.notch_freq, 30, fs=self.sampling_rate)
            # Compute initial filter state templates for stateful filtering
            self.bandpass_zi_template = lfilter_zi(self.bandpass_b, self.bandpass_a)
            self.notch_zi_template = lfilter_zi(self.notch_b, self.notch_a)
            # Reset per-channel states so they get re-initialized on next data
            self.bandpass_zi.clear()
            self.notch_zi.clear()
            self.filter_coeffs_valid = True
        except Exception as e:
            print(f"Error calculating filter coefficients: {e}")
            self.filter_coeffs_valid = False

    def process_signal(self, data):
        """Apply signal processing to raw data using stateful filters for seamless streaming"""
        processed = np.zeros_like(data)

        for i in range(data.shape[0]):
            channel_data = data[i, :].copy()

            # Use pre-calculated filter coefficients with persistent state
            if SCIPY_AVAILABLE and self.filter_coeffs_valid:
                try:
                    # Initialize filter states for this channel if needed
                    if i not in self.bandpass_zi:
                        self.bandpass_zi[i] = self.bandpass_zi_template * channel_data[0]
                    if i not in self.notch_zi:
                        self.notch_zi[i] = self.notch_zi_template * channel_data[0]

                    # Bandpass filter with persistent state
                    channel_data, self.bandpass_zi[i] = lfilter(
                        self.bandpass_b, self.bandpass_a, channel_data, zi=self.bandpass_zi[i]
                    )
                    # Notch filter with persistent state
                    channel_data, self.notch_zi[i] = lfilter(
                        self.notch_b, self.notch_a, channel_data, zi=self.notch_zi[i]
                    )
                except:
                    pass  # Skip if not enough data

            # Apply magnitude scaling
            channel_data = channel_data * (self.magnitude_scale / 100.0)

            processed[i, :] = channel_data

        return processed

    def smooth_display_data(self, data):
        """Apply OpenBCI-style moving average smoothing to data for display.
        Uses centered window with edge extension so values don't spike at boundaries."""
        if not self.smoothing_enabled:
            return data
        win = max(1, int(self.smooth_seconds * self.sampling_rate))
        if win <= 1 or not SCIPY_AVAILABLE:
            return data
        nan_mask = np.isnan(data)
        if np.all(nan_mask):
            return data
        clean = np.where(nan_mask, 0.0, data)
        # uniform_filter1d with mode='nearest' extends edge values
        # instead of dividing by a shrinking window (no edge spikes)
        out = uniform_filter1d(clean, size=win, mode='nearest')
        out[nan_mask] = np.nan
        return out

    def update_streaming_plots(self):
        """Update plot widgets with streaming data - applies OpenBCI-style smoothing at display time"""
        if not self.active_channels:
            if hasattr(self, 'plot_widget') and self.plot_widget is not None:
                if self.display_mode == 'stacked' and self.stacked_plot_item is not None:
                    self.stacked_plot_item.clear()
                else:
                    self.plot_widget.clear()
                self.streaming_plot_items.clear()
            return

        active_list = sorted(self.active_channels)
        num_active = len(active_list)

        # Use stacked mode or overlay mode
        if self.display_mode == 'stacked' and self.stacked_plot_item is not None:
            # Auto-calculate channel spacing
            d = self.compute_channel_spacing(num_active)
            self.yRange = d  # Keep in sync
            offsets = np.arange(0, -num_active * d, -d)

            # Check if active channels match current plot items
            current_plotted_channels = set(self.streaming_plot_items.keys())
            if current_plotted_channels != self.active_channels:
                self.stacked_plot_item.clear()
                self.streaming_plot_items.clear()

                # Create plot items for each channel
                for k, ch_idx in enumerate(active_list):
                    if ch_idx < self.num_channels:
                        data = self.smooth_display_data(self.stream_buffer[ch_idx, :])
                        offset_data = (data - np.nanmean(data)) * self.channel_amplitude_scale + offsets[k]
                        color = pg.mkColor(self.channel_colors[ch_idx % len(self.channel_colors)])

                        plot_item = self.stacked_plot_item.plot(
                            self.stream_time_axis,
                            offset_data,
                            pen=pg.mkPen(color=color, width=1.5),
                            connect='finite',
                        )
                        self.streaming_plot_items[ch_idx] = plot_item

                # Set Y-axis ticks with channel labels
                yticks = [(-k * d, self.get_channel_name(ch_idx)) for k, ch_idx in enumerate(active_list)]
                self.stacked_plot_item.getAxis('left').setTicks([yticks, []])

                # Set Y range: one full gap above topmost, one below bottommost
                y_top = d
                y_bottom = -num_active * d
                self.stacked_plot_item.setYRange(y_bottom, y_top, padding=0)

            else:
                # Just update data (fast path)
                for k, ch_idx in enumerate(active_list):
                    if ch_idx in self.streaming_plot_items and ch_idx < self.num_channels:
                        data = self.smooth_display_data(self.stream_buffer[ch_idx, :])
                        offset_data = (data - np.nanmean(data)) * self.channel_amplitude_scale + offsets[k]
                        self.streaming_plot_items[ch_idx].setData(self.stream_time_axis, offset_data)

            # Update scale label and fix X range to full buffer window
            self.stacked_plot_item.setLabel('left', f'Scale: {d:.1f}')
            self.stacked_plot_item.setXRange(0, self.stream_time_axis[-1], padding=0)

        elif hasattr(self, 'plot_widget') and self.plot_widget is not None:
            # Overlay mode (original behavior)
            current_plotted_channels = set(self.streaming_plot_items.keys())

            if current_plotted_channels != self.active_channels:
                self.plot_widget.clear()
                self.streaming_plot_items.clear()

            for ch_idx in active_list:
                if ch_idx < self.num_channels:
                    data = self.smooth_display_data(self.stream_buffer[ch_idx, :])
                    color = pg.mkColor(self.channel_colors[ch_idx % len(self.channel_colors)])

                    if ch_idx in self.streaming_plot_items:
                        self.streaming_plot_items[ch_idx].setData(self.stream_time_axis, data)
                    else:
                        plot_item = self.plot_widget.plot(
                            self.stream_time_axis,
                            data,
                            pen=pg.mkPen(color=color, width=2),
                            name=self.get_channel_name(ch_idx),
                            connect='finite',
                        )
                        self.streaming_plot_items[ch_idx] = plot_item

            if self.stream_first_update:
                self.stream_first_update = False
                self.plot_widget.enableAutoRange()

    def update_fft(self):
        """Calculate and display FFT for active channels with smoothing - reuses plot items"""
        if not self.active_channels or self.stream_buffer is None:
            # Clear the plot when no channels are selected
            if hasattr(self, 'fft_plot_widget') and self.fft_plot_widget is not None:
                self.fft_plot_widget.clear()
                self.fft_plot_items.clear()
            return

        # Remove plot items for channels that are no longer active
        channels_to_remove = [ch for ch in self.fft_plot_items if ch not in self.active_channels]
        for ch_idx in channels_to_remove:
            self.fft_plot_widget.removeItem(self.fft_plot_items[ch_idx])
            del self.fft_plot_items[ch_idx]

        for ch_idx in sorted(self.active_channels):
            if ch_idx < self.num_channels:
                data = np.nan_to_num(self.stream_buffer[ch_idx, :], nan=0.0)

                # Apply Hanning window to eliminate spectral leakage
                window = np.hanning(len(data))
                windowed_data = data * window

                # Compute FFT on windowed data
                fft_vals = np.fft.rfft(windowed_data)
                fft_freq = np.fft.rfftfreq(len(data), 1.0 / self.sampling_rate)

                # Convert to power (dB), normalize for window energy loss
                fft_power = 20 * np.log10(np.abs(fft_vals) * 2.0 / np.sum(window) + 1e-10)

                # Temporal EMA smoothing
                mask = fft_freq <= 50
                new_fft = fft_power[mask]
                if ch_idx in self.smoothed_fft:
                    self.smoothed_fft[ch_idx] = (
                        self.fft_smoothing_alpha * new_fft +
                        (1 - self.fft_smoothing_alpha) * self.smoothed_fft[ch_idx]
                    )
                else:
                    self.smoothed_fft[ch_idx] = new_fft.copy()

                # Frequency-domain smoothing for a clean curve shape
                display_fft = uniform_filter1d(self.smoothed_fft[ch_idx], size=5)

                # Update or create plot item
                color = pg.mkColor(self.channel_colors[ch_idx % len(self.channel_colors)])

                if ch_idx in self.fft_plot_items:
                    self.fft_plot_items[ch_idx].setData(fft_freq[mask], display_fft)
                else:
                    plot_item = self.fft_plot_widget.plot(
                        fft_freq[mask],
                        display_fft,
                        pen=pg.mkPen(color=color, width=2),
                        name=self.get_channel_name(ch_idx)
                    )
                    self.fft_plot_items[ch_idx] = plot_item

                # Store for reference
                self.fft_data[ch_idx] = (fft_freq[mask], self.smoothed_fft[ch_idx])

    def update_band_power(self):
        """Calculate and display band power for active channels - reuses bar graph item"""
        if not self.active_channels or self.stream_buffer is None:
            # Clear the plot when no channels are selected
            if hasattr(self, 'band_power_plot_widget') and self.band_power_plot_widget is not None:
                self.band_power_plot_widget.clear()
                self.band_power_bar_item = None
            return

        if not SCIPY_AVAILABLE:
            return

        # Define frequency bands
        bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Mu': (8, 12),
            'Beta': (13, 30),
            'Gamma': (30, 50)
        }

        # Average across all active channels
        band_powers = {band: 0.0 for band in bands.keys()}

        for ch_idx in sorted(self.active_channels):
            if ch_idx < self.num_channels:
                data = np.nan_to_num(self.stream_buffer[ch_idx, :], nan=0.0)

                # Compute PSD using Welch's method
                freqs, psd = sp_signal.welch(data, fs=self.sampling_rate, nperseg=min(256, len(data)))

                # Calculate power in each band
                for band_name, (low, high) in bands.items():
                    band_mask = (freqs >= low) & (freqs < high)
                    if np.any(band_mask):
                        band_powers[band_name] += np.mean(psd[band_mask])

        # Average across channels
        num_active = len(self.active_channels)
        if num_active > 0:
            for band in band_powers:
                band_powers[band] /= num_active

        # Apply exponential moving average smoothing
        if self.smoothed_band_power:
            for band_name in band_powers.keys():
                if band_name in self.smoothed_band_power:
                    self.smoothed_band_power[band_name] = (
                        self.band_power_smoothing_alpha * band_powers[band_name] +
                        (1 - self.band_power_smoothing_alpha) * self.smoothed_band_power[band_name]
                    )
                else:
                    self.smoothed_band_power[band_name] = band_powers[band_name]
        else:
            # First time, initialize with current values
            self.smoothed_band_power = band_powers.copy()

        # Plot smoothed band powers as bar chart
        band_names = list(bands.keys())
        band_labels = [f"{name}\n({low}-{high} Hz)" for name, (low, high) in bands.items()]
        band_values = [self.smoothed_band_power[name] for name in band_names]
        x_pos = np.arange(len(band_names))
        bar_color = (200, 80, 80) if self.theme == 'dark' else (60, 100, 200)

        # Stable Y-axis: track running max, slowly decay so axis doesn't jump
        current_max = max(band_values) if band_values else 1.0
        if current_max > self.band_power_y_max:
            self.band_power_y_max = current_max
        else:
            # Slowly decay toward current max (prevents axis from being stuck too high)
            self.band_power_y_max = 0.995 * self.band_power_y_max + 0.005 * current_max

        # Reuse or create bar graph item
        if self.band_power_bar_item is None:
            # First time: create bar graph and add to widget
            self.band_power_bar_item = pg.BarGraphItem(x=x_pos, height=band_values, width=0.6, brush=bar_color)
            self.band_power_plot_widget.addItem(self.band_power_bar_item)

            # Set x-axis labels (only needs to be done once)
            ax = self.band_power_plot_widget.getAxis('bottom')
            ax.setTicks([[(i, band_labels[i]) for i in range(len(band_labels))]])
        else:
            # Update existing bar graph item
            self.band_power_bar_item.setOpts(height=band_values, brush=bar_color)

        # Lock Y-axis to stable max with 10% headroom
        self.band_power_plot_widget.setYRange(0, self.band_power_y_max * 1.1, padding=0)

    def motor_imagery_task_placeholder(self):
        """Placeholder for motor imagery task - to be integrated later"""
        QMessageBox.information(
            self,
            "Motor Imagery Task",
            "Motor imagery task integration point.\n\nThis will be connected to your motor imagery task implementation later."
        )

        # Send a marker if marker stream is active
        if self.marker_outlet is not None and LSL_AVAILABLE:
            self.marker_outlet.push_sample(['MOTOR_IMAGERY_START'])
            print("Marker sent: MOTOR_IMAGERY_START")

    def on_magnitude_changed(self, text):
        """Handle magnitude scale change"""
        try:
            self.magnitude_scale = int(text)
        except:
            self.magnitude_scale = 100

    def on_smooth_slider_changed(self, value):
        """Handle OpenBCI-style smooth slider change (value is in tens of ms)"""
        ms = value * 10
        self.smooth_seconds = ms / 1000.0
        self.smooth_label.setText(f"{ms} ms")

    def on_smooth_checkbox_changed(self, state):
        """Enable or disable smoothing"""
        self.smoothing_enabled = (state == Qt.Checked)
        self.smooth_slider.setEnabled(self.smoothing_enabled)

    def toggle_all_channels(self, state):
        """Toggle all channel checkboxes"""
        is_checked = (state == Qt.Checked)
        for cb in self.channel_checkboxes:
            cb.blockSignals(True)
            cb.setChecked(is_checked)
            cb.blockSignals(False)

        # Manually update active channels
        if is_checked:
            self.active_channels = set(range(self.num_channels))
        else:
            self.active_channels.clear()

        self.update_selected_channels_label()

        # Update plots for file mode only - streaming mode updates via timer
        if self.file_loaded and not self.streaming_active:
            self.update_plot()
            self.calculate_fft_for_file()
            self.calculate_band_power_for_file()

    def update_selected_channels_label(self):
        """Update the label showing selected channels"""
        if not self.active_channels:
            self.selected_channels_label.setText("No channels selected")
        elif len(self.active_channels) == self.num_channels:
            self.selected_channels_label.setText(f"All {self.num_channels} channels selected")
        else:
            selected_list = sorted(list(self.active_channels))
            if len(selected_list) <= 5:
                channels_str = ", ".join([self.get_channel_name(i) for i in selected_list])
                self.selected_channels_label.setText(f"Selected: {channels_str}")
            else:
                self.selected_channels_label.setText(f"{len(selected_list)} channels selected")

    def apply_theme(self):
        """Apply the selected theme to the application"""
        app = QApplication.instance()

        if self.theme == 'dark':
            # Sleek dark theme
            dark_palette = QPalette()
            dark_palette.setColor(QPalette.Window, QColor(30, 30, 30))
            dark_palette.setColor(QPalette.WindowText, QColor(230, 230, 230))
            dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
            dark_palette.setColor(QPalette.AlternateBase, QColor(35, 35, 35))
            dark_palette.setColor(QPalette.ToolTipBase, QColor(50, 50, 50))
            dark_palette.setColor(QPalette.ToolTipText, QColor(230, 230, 230))
            dark_palette.setColor(QPalette.Text, QColor(230, 230, 230))
            dark_palette.setColor(QPalette.Button, QColor(45, 45, 45))
            dark_palette.setColor(QPalette.ButtonText, QColor(230, 230, 230))
            dark_palette.setColor(QPalette.BrightText, QColor(255, 100, 100))
            dark_palette.setColor(QPalette.Link, QColor(80, 160, 255))
            dark_palette.setColor(QPalette.Highlight, QColor(70, 130, 220))
            dark_palette.setColor(QPalette.HighlightedText, Qt.white)
            dark_palette.setColor(QPalette.Disabled, QPalette.Text, QColor(100, 100, 100))
            dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(100, 100, 100))
            app.setPalette(dark_palette)

            # Sleek dark plots
            pg.setConfigOption('background', (20, 20, 20))
            pg.setConfigOption('foreground', (230, 230, 230))

            # Sleek stylesheet
            app.setStyleSheet("""
                QGroupBox { border: 1px solid #3a3a3a; border-radius: 5px; margin-top: 10px; padding-top: 10px; font-weight: bold; }
                QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 2px 5px; color: #e6e6e6; }
                QPushButton { background-color: #3d3d3d; border: 1px solid #555; border-radius: 4px; padding: 5px 10px; color: #e6e6e6; }
                QPushButton:hover { background-color: #4a4a4a; border: 1px solid #6a6a6a; }
                QPushButton:pressed { background-color: #2a2a2a; }
                QPushButton:disabled { background-color: #2a2a2a; color: #646464; }
                QComboBox { background-color: #3d3d3d; border: 1px solid #555; border-radius: 4px; padding: 3px 10px; color: #e6e6e6; }
                QComboBox:hover { border: 1px solid #6a6a6a; }
                QSpinBox, QDoubleSpinBox, QLineEdit { background-color: #3d3d3d; border: 1px solid #555; border-radius: 4px; padding: 3px; color: #e6e6e6; }
                QSlider::groove:horizontal { background: #3a3a3a; height: 6px; border-radius: 3px; }
                QSlider::handle:horizontal { background: #4686d6; width: 14px; margin: -4px 0; border-radius: 7px; }
                QSlider::groove:vertical { background: #3a3a3a; width: 6px; border-radius: 3px; }
                QSlider::handle:vertical { background: #4686d6; height: 14px; margin: 0 -4px; border-radius: 7px; }
                QTabWidget::pane { border: 1px solid #3a3a3a; background: #1e1e1e; }
                QTabBar::tab { background: #2d2d2d; border: 1px solid #3a3a3a; padding: 8px 16px; color: #e6e6e6; }
                QTabBar::tab:selected { background: #3d3d3d; border-bottom-color: #3d3d3d; }
                QTabBar::tab:hover { background: #3a3a3a; }
                QMessageBox { background-color: #2a2a2a; color: #e6e6e6; }
                QMessageBox QLabel { color: #e6e6e6; }
                QDialog { background-color: #2a2a2a; color: #e6e6e6; }
            """)

        else:
            # Light theme
            app.setPalette(QApplication.style().standardPalette())
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')

            app.setStyleSheet("""
                QGroupBox { border: 1px solid #c0c0c0; border-radius: 5px; margin-top: 10px; padding-top: 10px; font-weight: bold; }
                QPushButton { border: 1px solid #b0b0b0; border-radius: 4px; padding: 5px 10px; background-color: #f0f0f0; color: #1a1a1a; }
                QPushButton:hover { background-color: #e0e0e0; border: 1px solid #909090; }
                QPushButton:pressed { background-color: #d0d0d0; }
                QTabBar::tab { padding: 8px 16px; }
                QMessageBox { background-color: #f5f5f5; color: #1a1a1a; }
                QMessageBox QLabel { color: #1a1a1a; }
                QDialog { background-color: #f5f5f5; color: #1a1a1a; }
            """)

        # Regenerate channel colors for the new theme
        self.generate_channel_colors()

        # Update backgrounds and foregrounds on existing plot widgets
        bg = (20, 20, 20) if self.theme == 'dark' else 'w'
        fg = (230, 230, 230) if self.theme == 'dark' else 'k'
        if hasattr(self, 'plot_widget') and self.plot_widget is not None:
            self.plot_widget.setBackground(bg)
        for pw in [getattr(self, 'fft_plot_widget', None),
                    getattr(self, 'band_power_plot_widget', None)]:
            if pw is not None:
                pw.setBackground(bg)
                for axis_name in ('left', 'bottom'):
                    ax = pw.getAxis(axis_name)
                    ax.setPen(pg.mkPen(color=fg))
                    ax.setTextPen(pg.mkPen(color=fg))

        # Reset band power bar item so it recreates with new colors/labels
        self.band_power_bar_item = None
        if hasattr(self, 'band_power_plot_widget') and self.band_power_plot_widget is not None:
            self.band_power_plot_widget.clear()

        # Clear streaming FFT items so they recreate with new colors
        if hasattr(self, 'fft_plot_items'):
            self.fft_plot_items.clear()
        if hasattr(self, 'fft_plot_widget') and self.fft_plot_widget is not None:
            self.fft_plot_widget.clear()

        # Rebuild plots so new colors and background take effect
        if self.file_loaded or self.streaming_active:
            if self.display_mode == 'stacked':
                self.setup_stacked_mode()
            else:
                self.setup_overlay_mode()
            self.update_plot()

    def calculate_fft_for_file(self):
        """Calculate and display FFT for file mode with smoothing - reuses plot items"""
        if not self.active_channels or self.raw_data is None:
            # Clear the plot when no channels are selected
            if hasattr(self, 'fft_plot_widget') and self.fft_plot_widget is not None:
                self.fft_plot_widget.clear()
                self.fft_plot_items.clear()
            return

        if not SCIPY_AVAILABLE:
            return

        # Remove plot items for channels that are no longer active
        channels_to_remove = [ch for ch in self.fft_plot_items if ch not in self.active_channels]
        for ch_idx in channels_to_remove:
            self.fft_plot_widget.removeItem(self.fft_plot_items[ch_idx])
            del self.fft_plot_items[ch_idx]

        # Get current window data
        start_sample, end_sample = self.get_window_bounds(self.current_window)

        for ch_idx in sorted(self.active_channels):
            if ch_idx < self.num_channels:
                data = self.raw_data[ch_idx, start_sample:end_sample]

                # Apply Hanning window to eliminate spectral leakage
                window = np.hanning(len(data))
                windowed_data = data * window

                # Compute FFT on windowed data
                fft_vals = np.fft.rfft(windowed_data)
                fft_freq = np.fft.rfftfreq(len(data), 1.0 / self.sampling_rate)

                # Convert to power (dB), normalize for window energy loss
                fft_power = 20 * np.log10(np.abs(fft_vals) * 2.0 / np.sum(window) + 1e-10)

                # Temporal EMA smoothing
                mask = fft_freq <= 50
                new_fft_data = fft_power[mask]

                # Check if smoothed buffer exists and has correct shape
                if ch_idx in self.smoothed_fft:
                    if self.smoothed_fft[ch_idx].shape != new_fft_data.shape:
                        del self.smoothed_fft[ch_idx]

                if ch_idx in self.smoothed_fft:
                    self.smoothed_fft[ch_idx] = (
                        self.fft_smoothing_alpha * new_fft_data +
                        (1 - self.fft_smoothing_alpha) * self.smoothed_fft[ch_idx]
                    )
                else:
                    self.smoothed_fft[ch_idx] = new_fft_data.copy()

                # Frequency-domain smoothing for a clean curve shape
                display_fft = uniform_filter1d(self.smoothed_fft[ch_idx], size=5)

                color = pg.mkColor(self.channel_colors[ch_idx % len(self.channel_colors)])

                # Reuse existing plot item or create new one
                if ch_idx in self.fft_plot_items:
                    self.fft_plot_items[ch_idx].setData(fft_freq[mask], display_fft)
                else:
                    plot_item = self.fft_plot_widget.plot(
                        fft_freq[mask],
                        display_fft,
                        pen=pg.mkPen(color=color, width=2),
                        name=self.get_channel_name(ch_idx)
                    )
                    self.fft_plot_items[ch_idx] = plot_item

    def calculate_band_power_for_file(self):
        """Calculate and display band power for file mode with smoothing - reuses bar graph item"""
        if not self.active_channels or self.raw_data is None:
            # Clear the plot when no channels are selected
            if hasattr(self, 'band_power_plot_widget') and self.band_power_plot_widget is not None:
                self.band_power_plot_widget.clear()
                self.band_power_bar_item = None
            return

        if not SCIPY_AVAILABLE:
            return

        # Define frequency bands
        bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Mu': (8, 12),
            'Beta': (13, 30),
            'Gamma': (30, 50)
        }

        # Get current window data
        start_sample, end_sample = self.get_window_bounds(self.current_window)

        # Average across all active channels
        band_powers = {band: 0.0 for band in bands.keys()}

        for ch_idx in sorted(self.active_channels):
            if ch_idx < self.num_channels:
                data = self.raw_data[ch_idx, start_sample:end_sample]

                # Compute PSD using Welch's method
                freqs, psd = sp_signal.welch(data, fs=self.sampling_rate, nperseg=min(256, len(data)))

                # Calculate power in each band
                for band_name, (low, high) in bands.items():
                    band_mask = (freqs >= low) & (freqs < high)
                    if np.any(band_mask):
                        band_powers[band_name] += np.mean(psd[band_mask])

        # Average across channels
        num_active = len(self.active_channels)
        if num_active > 0:
            for band in band_powers:
                band_powers[band] /= num_active

        # Apply exponential moving average smoothing
        if self.smoothed_band_power:
            for band_name in band_powers.keys():
                if band_name in self.smoothed_band_power:
                    self.smoothed_band_power[band_name] = (
                        self.band_power_smoothing_alpha * band_powers[band_name] +
                        (1 - self.band_power_smoothing_alpha) * self.smoothed_band_power[band_name]
                    )
                else:
                    self.smoothed_band_power[band_name] = band_powers[band_name]
        else:
            # First time, initialize with current values
            self.smoothed_band_power = band_powers.copy()

        # Plot smoothed band powers
        band_names = list(bands.keys())
        band_labels = [f"{name}\n({low}-{high} Hz)" for name, (low, high) in bands.items()]
        band_values = [self.smoothed_band_power[name] for name in band_names]
        x_pos = np.arange(len(band_names))
        bar_color = (200, 80, 80) if self.theme == 'dark' else (60, 100, 200)

        # Stable Y-axis: track running max, slowly decay
        current_max = max(band_values) if band_values else 1.0
        if current_max > self.band_power_y_max:
            self.band_power_y_max = current_max
        else:
            self.band_power_y_max = 0.995 * self.band_power_y_max + 0.005 * current_max

        # Reuse or create bar graph item (avoids flicker from clear+recreate)
        if self.band_power_bar_item is None:
            self.band_power_bar_item = pg.BarGraphItem(x=x_pos, height=band_values, width=0.6, brush=bar_color)
            self.band_power_plot_widget.addItem(self.band_power_bar_item)

            # Set x-axis labels (only needs to be done once)
            ax = self.band_power_plot_widget.getAxis('bottom')
            ax.setTicks([[(i, band_labels[i]) for i in range(len(band_labels))]])
        else:
            self.band_power_bar_item.setOpts(height=band_values, brush=bar_color)

        # Lock Y-axis to stable max with 10% headroom
        self.band_power_plot_widget.setYRange(0, self.band_power_y_max * 1.1, padding=0)


def main():
    # Initialize app with no file
    # Check if QApplication instance already exists
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    viewer = SegmentViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
