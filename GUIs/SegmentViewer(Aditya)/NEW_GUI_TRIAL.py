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
    QFileDialog, QMessageBox, QComboBox
)
from PyQt5.QtCore import Qt, QTimer

# Import your data loading functions
from conversions import get_gdf_array, get_pkl_array

class WindowSettingsDialog(QDialog):
    """Dialog for configuring window display settings"""
    def __init__(self, current_mode, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Window Settings")
        
        layout = QVBoxLayout()
        
        # Display mode selection
        mode_label = QLabel("Display Mode:")
        layout.addWidget(mode_label)
        
        self.mode_group = QButtonGroup()
        
        self.overlay_radio = QRadioButton("Overlay - All channels in same plot")
        self.stacked_radio = QRadioButton("Stacked - Separate subplot per channel")
        
        self.mode_group.addButton(self.overlay_radio)
        self.mode_group.addButton(self.stacked_radio)
        
        if current_mode == 'overlay':
            self.overlay_radio.setChecked(True)
        else:
            self.stacked_radio.setChecked(True)
        
        layout.addWidget(self.overlay_radio)
        layout.addWidget(self.stacked_radio)
        
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
    
    def get_mode(self):
        """Return selected display mode"""
        return 'overlay' if self.overlay_radio.isChecked() else 'stacked'


class SegmentViewer(QMainWindow):
    def __init__(self, window_size_sec=2.0, sampling_rate=125):
        super().__init__()
        self.raw_data = None  # No data loaded initially
        self.window_size_sec = window_size_sec
        self.sampling_rate = sampling_rate
        self.display_mode = 'overlay'  # 'overlay' or 'stacked'
        self.file_loaded = False
        self.current_filename = ""
        
        # Initialize with no segmentation
        self.segmented = None
        self.num_windows = 0
        self.num_channels = 0
        self.num_samples = 0
        self.current_window = 0
        self.active_channels = set()
        self.last_stacked_channels = set()  # Track which channels are in stacked layout
        
        # Track last processed horizontal zoom value to avoid expensive updates
        self.last_horizontal_zoom_value = None
        
        # Vertical bounds mode: 'iqr' or 'minmax'
        self.vertical_bounds_mode = 'iqr'
        
        # Vertical zoom factor for stacked mode (amplitude scaling)
        self.stacked_vertical_zoom = 1.0
        
        # Autoplay state
        self.autoplay_active = False
        self.autoplay_speed = 1.0  # 1x = real-time
        self.autoplay_timer = QTimer()
        self.autoplay_timer.timeout.connect(self.autoplay_step)
        self.autoplay_current_time = 0.0  # Current time position in seconds
        
        self.setWindowTitle("Segment Viewer - No File Loaded")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.main_layout = QVBoxLayout(main_widget)
        
        # File loading section
        file_layout = QHBoxLayout()
        load_file_btn = QPushButton("📁 Load File")
        load_file_btn.clicked.connect(self.load_file)
        file_layout.addWidget(load_file_btn)
        
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("color: gray; font-style: italic;")
        file_layout.addWidget(self.file_label)
        file_layout.addStretch()
        
        self.main_layout.addLayout(file_layout)
        
        # Channel checkboxes (initially disabled but visible)
        self.channel_group = QGroupBox("Channels")
        channel_layout = QHBoxLayout()
        self.channel_checkboxes = []
        self.channel_group.setLayout(channel_layout)
        self.channel_group.setEnabled(False)  # Disabled until file loaded
        self.main_layout.addWidget(self.channel_group)
        
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
        
        # Autoplay button
        self.autoplay_btn = QPushButton("▶ Autoplay")
        self.autoplay_btn.clicked.connect(self.toggle_autoplay)
        self.autoplay_btn.setEnabled(False)  # Disabled until file loaded
        self.autoplay_btn.setMaximumWidth(120)
        horizontal_zoom_layout.addWidget(self.autoplay_btn)
        
        horizontal_zoom_layout.addStretch()  # Push everything to the left
        
        plot_column_layout.addWidget(horizontal_zoom_widget)
        
        plot_and_zoom_layout.addWidget(plot_column_widget, stretch=1)
        
        self.main_layout.addWidget(plot_and_zoom_widget, stretch=1)
        
        # Autoplay playback controls (hidden by default)
        self.playback_controls_widget = QWidget()
        playback_controls_layout = QHBoxLayout(self.playback_controls_widget)
        playback_controls_layout.setContentsMargins(0, 5, 0, 5)
        
        playback_controls_layout.addStretch()
        
        # Play/Pause button
        self.play_pause_btn = QPushButton("⏸ Pause")
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.play_pause_btn.setMaximumWidth(100)
        playback_controls_layout.addWidget(self.play_pause_btn)
        
        # Playback speed label and spinbox
        playback_controls_layout.addWidget(QLabel("  Speed:"))
        self.playback_speed_spinbox = QDoubleSpinBox()
        self.playback_speed_spinbox.setRange(0.1, 10.0)
        self.playback_speed_spinbox.setSingleStep(0.1)
        self.playback_speed_spinbox.setValue(1.0)
        self.playback_speed_spinbox.setDecimals(1)
        self.playback_speed_spinbox.setSuffix("x")
        self.playback_speed_spinbox.setMaximumWidth(80)
        self.playback_speed_spinbox.valueChanged.connect(self.on_playback_speed_changed)
        playback_controls_layout.addWidget(self.playback_speed_spinbox)
        
        playback_controls_layout.addStretch()
        
        self.playback_controls_widget.setVisible(False)  # Hidden until autoplay is activated
        self.main_layout.addWidget(self.playback_controls_widget)
        
        # Create initial "no file" message
        self.no_file_label = QLabel("No file loaded\n\nClick 'Load File' to begin")
        self.no_file_label.setAlignment(Qt.AlignCenter)
        self.no_file_label.setStyleSheet("font-size: 24px; color: gray;")
        self.plot_layout.addWidget(self.no_file_label)
        
        # Window navigation (initially disabled but visible)
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
        
        # Auto-fit button
        self.autofit_btn = QPushButton("📈 Auto-Fit")
        self.autofit_btn.clicked.connect(self.auto_fit_plot)
        nav_layout.addWidget(self.autofit_btn)
        
        # Window settings button
        self.window_settings_btn = QPushButton("👁️ Window Settings")
        self.window_settings_btn.clicked.connect(self.open_window_settings_dialog)
        nav_layout.addWidget(self.window_settings_btn)
        
        self.nav_widget.setEnabled(False)  # Disabled until file loaded
        self.main_layout.addWidget(self.nav_widget)
    
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
                self.raw_data = get_gdf_array(file_path)
            elif file_path.endswith('.pkl'):
                self.raw_data = get_pkl_array(file_path)
            else:
                QMessageBox.warning(self, "Error", "Unsupported file type. Please select a .gdf or .pkl file.")
                return
            
            # Update UI with loaded file
            self.current_filename = file_path.split('/')[-1]
            self.file_label.setText(f"Loaded: {self.current_filename}")
            self.file_label.setStyleSheet("color: green; font-weight: bold;")
            self.file_loaded = True
            
            # Prepare data and calculate window parameters
            self.prepare_data()
            self.current_window = 0
            
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
            
            # Setup channel checkboxes
            channel_layout = self.channel_group.layout()
            # Clear existing checkboxes
            while channel_layout.count():
                child = channel_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            
            self.channel_checkboxes = []
            self.active_channels.clear()  # Reset active channels
            for i in range(self.num_channels):
                cb = QCheckBox(f"Ch{i+1}")
                cb.stateChanged.connect(lambda state, ch=i: self.toggle_channel(ch, state))
                channel_layout.addWidget(cb)
                self.channel_checkboxes.append(cb)
            
            # Enable channel group
            self.channel_group.setEnabled(True)
            
            # Enable zoom sliders and spinboxes
            self.vertical_zoom_slider.setEnabled(True)
            self.vertical_zoom_spinbox.setEnabled(True)
            self.bounds_mode_combo.setEnabled(True)
            self.horizontal_zoom_slider.setEnabled(True)
            self.window_size_spinbox.setEnabled(True)
            self.sampling_rate_spinbox.setEnabled(True)
            self.autoplay_btn.setEnabled(True)  # Enable autoplay button
            
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
        """Setup single plot widget with vertically offset channels and multiple Y-axes"""
        # Clear existing plots
        while self.plot_layout.count():
            child = self.plot_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Create a graphics layout widget to hold everything
        self.plot_widget = pg.GraphicsLayoutWidget()
        
        # Create a single plot item - will be recreated when channels are selected
        self.stacked_plot_item = None
        
        self.plot_layout.addWidget(self.plot_widget)
        self.plot_widgets = []  # Will be set when plot is created
        
        # Dictionary to store plot items for each channel
        self.stacked_plot_items = {}
        
        # Dictionary to store Y-axis items for each channel
        self.stacked_y_axes = {}
    
    def prepare_data(self):
        """Prepare data and calculate window parameters and vertical bounds"""
        self.recalculate_windows()
        
        # Pre-calculate both IQR and min-max bounds for entire dataset (per channel)
        # This is expensive so only do it once when file is loaded
        self.channel_iqr_bounds = {}
        self.channel_minmax_bounds = {}
        
        for ch_idx in range(self.num_channels):
            # Get all data for this channel
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
    
    def recalculate_windows(self):
        """Recalculate window parameters (fast, no IQR calculation)"""
        # Calculate window parameters
        self.window_size_samples = int(self.window_size_sec * self.sampling_rate)
        self.stride = int(self.window_size_samples * 0.8)  # 20% overlap
        
        # Calculate number of windows
        total_samples = self.raw_data.shape[1]
        self.num_windows = (total_samples - self.window_size_samples) // self.stride + 1
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
        if not self.file_loaded:
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
        """Get start and end sample indices for a given window"""
        start_sample = window_idx * self.stride
        end_sample = start_sample + self.window_size_samples
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
        - Horizontal scroll (left/right): Pan time axis
        - Vertical scroll: Disabled (to prevent accidental zooming)
        
        Note: pyqtgraph uses QGraphicsSceneWheelEvent, not QWheelEvent, which has
        different methods (delta() and orientation() instead of angleDelta()).
        """
        from PyQt5.QtCore import Qt
        
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
        if not self.file_loaded or not self.active_channels:
            return
        
        # Get the zoom factor (1.0 = 100%)
        zoom_factor = value / 100.0
        
        if self.display_mode == 'overlay':
            # Get current bounds based on mode and scale them
            y_min, y_max = self.get_bounds_for_channels(self.active_channels)
            y_center = (y_min + y_max) / 2.0
            y_range = (y_max - y_min) / zoom_factor
            
            new_y_min = y_center - y_range / 2.0
            new_y_max = y_center + y_range / 2.0
            self.plot_widget.setYRange(new_y_min, new_y_max, padding=0)
        else:
            # Stacked mode: just update the zoom factor (will be applied in next data update)
            self.stacked_vertical_zoom = zoom_factor
            
            # Manually update the existing plot items with new zoom
            if self.active_channels and self.stacked_plot_items and self.stacked_plot_item is not None:
                # Get current view range
                view_range = self.stacked_plot_item.viewRange()[0]
                start_idx, end_idx = self.get_visible_data_range(view_range)
                
                # Update each channel with new zoom
                active_list = sorted(self.active_channels)
                channel_height = 1.0
                
                for idx, ch_idx in enumerate(active_list):
                    if ch_idx in self.stacked_plot_items:
                        data = self.raw_data[ch_idx, start_idx:end_idx]
                        time_slice = self.full_time_axis[start_idx:end_idx]
                        
                        ch_min, ch_max = self.get_channel_bounds(ch_idx)
                        data_range = ch_max - ch_min
                        if data_range > 0:
                            # Center the data around 0, then scale, then offset
                            normalized_data = ((data - ch_min) / data_range - 0.5) * channel_height * 0.8 * zoom_factor
                        else:
                            normalized_data = np.zeros_like(data)
                        
                        vertical_offset = idx * channel_height + channel_height * 0.5
                        offset_data = normalized_data + vertical_offset
                        
                        plot_item = self.stacked_plot_items[ch_idx]
                        plot_item.setData(time_slice, offset_data)
                
                # Update tick labels (not positions) to reflect new zoom
                left_axis = self.stacked_plot_item.getAxis('left')
                
                # Recalculate scale factor
                all_values = []
                for ch_idx in active_list:
                    ch_min, ch_max = self.get_channel_bounds(ch_idx)
                    all_values.extend([abs(ch_min), abs(ch_max)])
                
                max_abs_val = max(all_values) if all_values else 1.0
                
                if max_abs_val == 0:
                    scale_factor = 1.0
                else:
                    scale_exponent = int(np.floor(np.log10(max_abs_val)))
                    if max_abs_val < 0.01 or max_abs_val > 10000:
                        scale_factor = 10 ** scale_exponent
                    else:
                        scale_factor = 1.0
                
                tick_positions = []
                for idx, ch_idx in enumerate(active_list):
                    ch_min, ch_max = self.get_channel_bounds(ch_idx)
                    vertical_offset = idx * channel_height + channel_height * 0.5
                    
                    # Calculate what amplitude the FIXED positions represent
                    # Positions stay at +/- 0.4, but values change with zoom
                    axis_half_range = (ch_max - ch_min) * 0.4 / zoom_factor
                    axis_half_range_scaled = axis_half_range / scale_factor
                    
                    # Format labels
                    if axis_half_range_scaled < 0.01 and axis_half_range_scaled > 0:
                        upper_label = f"{axis_half_range_scaled:.2e}"
                        lower_label = f"{-axis_half_range_scaled:.2e}"
                    elif axis_half_range_scaled < 1:
                        upper_label = f"{axis_half_range_scaled:.3f}"
                        lower_label = f"{-axis_half_range_scaled:.3f}"
                    elif axis_half_range_scaled < 100:
                        upper_label = f"{axis_half_range_scaled:.2f}"
                        lower_label = f"{-axis_half_range_scaled:.2f}"
                    else:
                        upper_label = f"{axis_half_range_scaled:.1f}"
                        lower_label = f"{-axis_half_range_scaled:.1f}"
                    
                    # Tick positions are FIXED, only labels change
                    tick_positions.extend([
                        (vertical_offset + 0.4, upper_label),
                        (vertical_offset, "0"),
                        (vertical_offset - 0.4, lower_label)
                    ])
                
                left_axis.setTicks([tick_positions])
    
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
                # Update spinbox but skip expensive recalculation
                new_window_size = value / 10.0
                self.window_size_spinbox.blockSignals(True)
                self.window_size_spinbox.setValue(new_window_size)
                self.window_size_spinbox.blockSignals(False)
                return
        
        # Store this value as the last processed one
        self.last_horizontal_zoom_value = value
        
        # Convert slider value to seconds (slider is scaled by 10 for finer control)
        new_window_size = value / 10.0
        
        # Update the spinbox
        self.window_size_spinbox.blockSignals(True)
        self.window_size_spinbox.setValue(new_window_size)
        self.window_size_spinbox.blockSignals(False)
        
        # Get current view center BEFORE changing window size
        if self.display_mode == 'overlay':
            view_range = self.plot_widget.viewRange()[0]
        else:
            view_range = self.plot_widgets[0].viewRange()[0]
        
        t_center = (view_range[0] + view_range[1]) / 2.0
        
        # Update window size (this affects window calculations but not the view directly)
        self.window_size_sec = new_window_size
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
            self.plot_widgets[0].setXRange(t_start, t_end, padding=0)
    
    def on_window_size_spinbox_changed(self, value):
        """Handle window size spinbox changes"""
        if not self.file_loaded:
            return
        
        # Stop autoplay if active
        if self.autoplay_active and self.autoplay_timer.isActive():
            self.autoplay_timer.stop()
            self.play_pause_btn.setText("▶ Play")
        
        # Update slider to match
        slider_value = int(value * 10)
        self.horizontal_zoom_slider.blockSignals(True)
        self.horizontal_zoom_slider.setValue(slider_value)
        self.horizontal_zoom_slider.blockSignals(False)
        
        # Update tracking variable
        self.last_horizontal_zoom_value = slider_value
        
        # Get current view center BEFORE changing window size
        if self.display_mode == 'overlay':
            view_range = self.plot_widget.viewRange()[0]
        else:
            view_range = self.plot_widgets[0].viewRange()[0]
        
        t_center = (view_range[0] + view_range[1]) / 2.0
        
        # Update window size (this affects window calculations but not the view directly)
        self.window_size_sec = value
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
            self.plot_widgets[0].setXRange(t_start, t_end, padding=0)
    
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
        if not self.file_loaded:
            return
        if state == Qt.Checked:
            self.active_channels.add(channel_idx)
        else:
            self.active_channels.discard(channel_idx)
        self.update_plot()
    
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
        
        # Color palette for different channels
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w', 'orange']
        
        # Plot only visible data for each active channel
        for ch_idx in sorted(self.active_channels):
            data = self.raw_data[ch_idx, start_idx:end_idx]
            time_slice = self.full_time_axis[start_idx:end_idx]
            color = colors[ch_idx % len(colors)]
            
            plot_item = self.plot_widget.plot(
                time_slice,
                data,
                pen=pg.mkPen(color=color, width=2),
                name=f'Ch{ch_idx+1}'
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
        """Update plots in stacked mode - all channels on one plot with vertical offsets
        
        Args:
            rebuild_layout: If True, rebuild plot items (when channels change). 
                          If False, just update data (when navigating windows).
        """
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w', 'orange']
        
        # Get window time bounds
        start_sample, end_sample = self.get_window_bounds(self.current_window)
        t_start = self.full_time_axis[start_sample]
        t_end = self.full_time_axis[min(end_sample - 1, len(self.full_time_axis) - 1)]
        
        # Get visible data range with padding
        view_range = [t_start, t_end]
        start_idx, end_idx = self.get_visible_data_range(view_range)
        
        # Calculate vertical spacing between channels
        active_list = sorted(self.active_channels)
        num_active = len(active_list)
        
        if num_active == 0:
            return
        
        # Calculate the range needed for each channel and total offset spacing
        # We'll normalize each channel to a standard height and offset them
        channel_height = 1.0  # Normalized height for each channel
        total_height = num_active * channel_height
        
        # Check if we need to rebuild (channels changed)
        if rebuild_layout or self.active_channels != self.last_stacked_channels:
            # Clear the entire graphics layout
            self.plot_widget.clear()
            self.stacked_plot_items.clear()
            self.stacked_y_axes.clear()
            
            # All axes and the plot will share row 0, with axes stacked in column 0
            # Create Y-axes for each channel first
            for idx, ch_idx in enumerate(active_list):
                # Get bounds for this channel
                ch_min, ch_max = self.get_channel_bounds(ch_idx)
                color = colors[ch_idx % len(colors)]
                
                # Create axis item
                axis = pg.AxisItem(orientation='left')
                axis.setPen(color)
                axis.setTextPen(color)
                axis.setLabel(f'Ch{ch_idx+1}', color=color)
                
                # Calculate vertical position for this channel
                vertical_offset = idx * channel_height + channel_height * 0.5
                span = channel_height * 0.8 * self.stacked_vertical_zoom
                
                # Add axis to column 0 at its position
                # Note: We can't vertically position axes independently in pyqtgraph layout
                # So we'll use a different approach - just show one axis with custom ticks
                
                # Store axis info for later
                self.stacked_y_axes[ch_idx] = {
                    'axis': axis,
                    'ch_min': ch_min,
                    'ch_max': ch_max,
                    'vertical_offset': vertical_offset,
                    'span': span,
                    'idx': idx,
                    'color': color
                }
            
            # Create the main plot item - it will span all rows
            self.stacked_plot_item = self.plot_widget.addPlot(row=0, col=0, rowspan=num_active)
            self.stacked_plot_item.setLabel('bottom', 'Time (seconds)')
            self.stacked_plot_item.showGrid(x=True, y=False, alpha=0.3)
            
            # Keep the default left axis but customize it
            left_axis = self.stacked_plot_item.getAxis('left')
            left_axis.setLabel('')  # Remove label
            
            # Determine the overall scale factor for displaying values
            # Find the max absolute value across all channels to determine scale
            all_values = []
            for ch_idx in active_list:
                ch_min, ch_max = self.get_channel_bounds(ch_idx)
                all_values.extend([abs(ch_min), abs(ch_max)])
            
            max_abs_val = max(all_values) if all_values else 1.0
            
            # Determine appropriate scale factor
            if max_abs_val == 0:
                scale_exponent = 0
                scale_factor = 1.0
            else:
                scale_exponent = int(np.floor(np.log10(max_abs_val)))
                # Use scale for very small (< 0.01) or very large (> 10000) numbers
                if max_abs_val < 0.01 or max_abs_val > 10000:
                    scale_factor = 10 ** scale_exponent
                else:
                    scale_factor = 1.0
                    scale_exponent = 0
            
            # Add scale label if needed
            if scale_exponent != 0:
                # Format the exponent nicely
                if scale_exponent > 0:
                    scale_text = f'×10^{scale_exponent}'
                else:
                    scale_text = f'×10^{scale_exponent}'
                
                # Add the scale indicator to the axis label
                left_axis.setLabel(scale_text)
                # Set label color (the label is a QGraphicsTextItem)
                left_axis.label.setDefaultTextColor(pg.mkColor('w'))
            
            # Create tick positions with labels showing bounds
            tick_positions = []
            for idx, ch_idx in enumerate(active_list):
                info = self.stacked_y_axes[ch_idx]
                ch_min = info['ch_min']
                ch_max = info['ch_max']
                vertical_offset = idx * channel_height + channel_height * 0.5
                
                # Calculate the axis range (what the segment represents)
                # Positions are FIXED at +/- 0.4 from center (based on 100% zoom)
                # But the VALUES they represent change with zoom
                axis_half_range = (ch_max - ch_min) * 0.4 / self.stacked_vertical_zoom
                
                # Apply scale factor
                axis_half_range_scaled = axis_half_range / scale_factor
                
                # Format labels based on magnitude
                if axis_half_range_scaled < 0.01 and axis_half_range_scaled > 0:
                    upper_label = f"{axis_half_range_scaled:.2e}"
                    lower_label = f"{-axis_half_range_scaled:.2e}"
                elif axis_half_range_scaled < 1:
                    upper_label = f"{axis_half_range_scaled:.3f}"
                    lower_label = f"{-axis_half_range_scaled:.3f}"
                elif axis_half_range_scaled < 100:
                    upper_label = f"{axis_half_range_scaled:.2f}"
                    lower_label = f"{-axis_half_range_scaled:.2f}"
                else:
                    upper_label = f"{axis_half_range_scaled:.1f}"
                    lower_label = f"{-axis_half_range_scaled:.1f}"
                
                # Tick positions are FIXED at +/- 0.4 (as if zoom were 100%)
                tick_positions.extend([
                    (vertical_offset + 0.4, upper_label),  # Upper bound - FIXED position
                    (vertical_offset, "0"),                 # Center (0) - FIXED position
                    (vertical_offset - 0.4, lower_label)   # Lower bound - FIXED position
                ])
            
            # Set manual ticks
            left_axis.setTicks([tick_positions])
            
            # Color the axis labels based on which channel region they're in
            left_axis.setTextPen('w')  # Default white
            
            # Setup mouse events
            self.stacked_plot_item.setMouseEnabled(x=True, y=True)
            view_box = self.stacked_plot_item.getViewBox()
            view_box.setMouseMode(pg.ViewBox.RectMode)
            
            def make_wheel_handler(vb):
                return lambda event: self.custom_wheel_event(vb, event)
            view_box.wheelEvent = make_wheel_handler(view_box)
            
            # Connect range change signal
            self.stacked_plot_item.sigRangeChanged.connect(self.on_stacked_range_changed)
            
            # Update plot_widgets for compatibility
            self.plot_widgets = [self.stacked_plot_item]
            
            # Add channel labels as TextItems
            for idx, ch_idx in enumerate(active_list):
                info = self.stacked_y_axes[ch_idx]
                color = info['color']
                vertical_offset = info['vertical_offset']
                
                # Add channel label
                label = pg.TextItem(f'Ch{ch_idx+1}', color=color, anchor=(1, 0.5))
                label.setPos(t_start, vertical_offset)
                self.stacked_plot_item.addItem(label)
            
            # Plot each active channel with vertical offset
            for idx, ch_idx in enumerate(active_list):
                # Get data and normalize it
                data = self.raw_data[ch_idx, start_idx:end_idx]
                time_slice = self.full_time_axis[start_idx:end_idx]
                
                # Normalize channel data to fit within one channel_height unit
                ch_min, ch_max = self.get_channel_bounds(ch_idx)
                data_range = ch_max - ch_min
                if data_range > 0:
                    # Center the data around 0, then scale, then offset
                    normalized_data = ((data - ch_min) / data_range - 0.5) * channel_height * 0.8 * self.stacked_vertical_zoom
                else:
                    normalized_data = np.zeros_like(data)
                
                # Apply vertical offset (to center of channel's space)
                vertical_offset = idx * channel_height + channel_height * 0.5
                offset_data = normalized_data + vertical_offset
                
                # Plot with color
                color = colors[ch_idx % len(colors)]
                plot_item = self.stacked_plot_item.plot(
                    time_slice,
                    offset_data,
                    pen=pg.mkPen(color=color, width=1.5),
                    name=f'Ch{ch_idx+1}'
                )
                
                # Store plot item for dynamic updates
                self.stacked_plot_items[ch_idx] = plot_item
            
            # Set Y range to show all channels with some padding
            self.stacked_plot_item.setYRange(-0.2, total_height + 0.2, padding=0)
            
            # Set X range to current window
            self.stacked_plot_item.setXRange(t_start, t_end, padding=0)
            
            # Add legend
            self.stacked_plot_item.addLegend()
            
            # Remember which channels are displayed
            self.last_stacked_channels = self.active_channels.copy()
        else:
            # Just update data for existing plot items
            for idx, ch_idx in enumerate(active_list):
                if ch_idx in self.stacked_plot_items:
                    # Get data and normalize it
                    data = self.raw_data[ch_idx, start_idx:end_idx]
                    time_slice = self.full_time_axis[start_idx:end_idx]
                    
                    # Normalize channel data
                    ch_min, ch_max = self.get_channel_bounds(ch_idx)
                    data_range = ch_max - ch_min
                    if data_range > 0:
                        # Center the data around 0, then scale, then offset
                        normalized_data = ((data - ch_min) / data_range - 0.5) * channel_height * 0.8 * self.stacked_vertical_zoom
                    else:
                        normalized_data = np.zeros_like(data)
                    
                    # Apply vertical offset (to center of channel's space)
                    vertical_offset = idx * channel_height + channel_height * 0.5
                    offset_data = normalized_data + vertical_offset
                    
                    # Update plot item
                    plot_item = self.stacked_plot_items[ch_idx]
                    plot_item.setData(time_slice, offset_data)
            
            # Update X range
            self.stacked_plot_item.setXRange(t_start, t_end, padding=0)
    
    def on_stacked_range_changed(self):
        """Called when user zooms or pans in stacked mode - updates visible data for all channels"""
        if not self.file_loaded or not self.active_channels or self.stacked_plot_item is None:
            return
        
        # Update window controls to match current view
        self.update_window_controls_from_view()
        
        # Get the X range from plot
        view_range = self.stacked_plot_item.viewRange()[0]  # [x_min, x_max]
        start_idx, end_idx = self.get_visible_data_range(view_range)
        
        # Calculate vertical spacing
        active_list = sorted(self.active_channels)
        channel_height = 1.0
        
        # Update data for all active channels
        for idx, ch_idx in enumerate(active_list):
            if ch_idx in self.stacked_plot_items:
                # Get data and normalize it
                data = self.raw_data[ch_idx, start_idx:end_idx]
                time_slice = self.full_time_axis[start_idx:end_idx]
                
                # Normalize channel data
                ch_min, ch_max = self.get_channel_bounds(ch_idx)
                data_range = ch_max - ch_min
                if data_range > 0:
                    # Center the data around 0, then scale, then offset
                    normalized_data = ((data - ch_min) / data_range - 0.5) * channel_height * 0.8 * self.stacked_vertical_zoom
                else:
                    normalized_data = np.zeros_like(data)
                
                # Apply vertical offset (to center of channel's space)
                vertical_offset = idx * channel_height + channel_height * 0.5
                offset_data = normalized_data + vertical_offset
                
                # Update plot item
                plot_item = self.stacked_plot_items[ch_idx]
                plot_item.setData(time_slice, offset_data)
    
    def navigate_to_window(self, value):
        """Navigate to a specific window and update the view
        
        Args:
            value: Window index to navigate to
        """
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
            # Stacked mode: Set X range and reset Y range to show all channels
            if self.stacked_plot_item is not None:
                self.stacked_plot_item.setXRange(t_start, t_end, padding=0)
                
                # Calculate total height needed for all active channels
                num_active = len(self.active_channels)
                channel_height = 1.0
                total_height = num_active * channel_height
                # Add padding and account for centering (channels go from 0.5 to total_height - 0.5)
                self.stacked_plot_item.setYRange(-0.2, total_height + 0.2, padding=0)
    
    def toggle_autoplay(self):
        """Toggle autoplay mode on/off"""
        if not self.file_loaded:
            return
        
        self.autoplay_active = not self.autoplay_active
        
        if self.autoplay_active:
            # Entering autoplay mode
            self.autoplay_btn.setText("⏹ Exit Autoplay")
            self.playback_controls_widget.setVisible(True)
            self.play_pause_btn.setText("⏸ Pause")  # Set to Pause since we're starting playback
            
            # Initialize autoplay position from current view
            if self.display_mode == 'overlay':
                view_range = self.plot_widget.viewRange()
            else:
                view_range = self.plot_widgets[0].viewRange()
            
            self.autoplay_current_time = view_range[0][0]  # Start of current view
            
            # Start the timer - update every 16ms (~60 FPS for smooth animation)
            self.autoplay_timer.start(16)
        else:
            # Exiting autoplay mode
            self.autoplay_btn.setText("▶ Autoplay")
            self.playback_controls_widget.setVisible(False)
            self.autoplay_timer.stop()
    
    def toggle_play_pause(self):
        """Toggle play/pause during autoplay"""
        if not self.autoplay_active:
            return
        
        if self.autoplay_timer.isActive():
            # Currently playing - pause it
            self.autoplay_timer.stop()
            self.play_pause_btn.setText("▶ Play")
        else:
            # Currently paused - resume it
            self.autoplay_timer.start(16)
            self.play_pause_btn.setText("⏸ Pause")
    
    def on_playback_speed_changed(self, value):
        """Update playback speed"""
        self.autoplay_speed = value
    
    def autoplay_step(self):
        """Advance autoplay by one frame"""
        if not self.file_loaded or not self.autoplay_active or not self.active_channels:
            return
        
        # Calculate time increment based on speed (16ms interval, speed multiplier)
        # Real-time means 0.016 seconds per frame at 1x speed
        time_increment = 0.016 * self.autoplay_speed
        
        # Advance current time
        self.autoplay_current_time += time_increment
        
        # Get total duration
        total_duration = self.full_time_axis[-1]
        
        # Check if we've reached the end
        if self.autoplay_current_time >= total_duration - self.window_size_sec / 2:
            # Loop back to beginning
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
        
        # Update the view range and data (smoothly scroll the graph)
        if self.display_mode == 'overlay':
            # Set X range
            self.plot_widget.setXRange(t_start, t_end, padding=0)
            
            # Update visible data for dynamic loading
            view_range = [t_start, t_end]
            start_idx, end_idx = self.get_visible_data_range(view_range)
            
            # Update data for each active channel
            for ch_idx in sorted(self.active_channels):
                if ch_idx in self.overlay_plot_items:
                    data = self.raw_data[ch_idx, start_idx:end_idx]
                    time_slice = self.full_time_axis[start_idx:end_idx]
                    self.overlay_plot_items[ch_idx].setData(time_slice, data)
        else:
            # Stacked mode: update plot with offsets
            if self.stacked_plot_item is not None:
                self.stacked_plot_item.setXRange(t_start, t_end, padding=0)
            
            # Update visible data for dynamic loading
            view_range = [t_start, t_end]
            start_idx, end_idx = self.get_visible_data_range(view_range)
            
            # Calculate vertical spacing
            active_list = sorted(self.active_channels)
            channel_height = 1.0
            
            # Update data for all active channels with offsets
            for idx, ch_idx in enumerate(active_list):
                if ch_idx in self.stacked_plot_items:
                    # Get data and normalize it
                    data = self.raw_data[ch_idx, start_idx:end_idx]
                    time_slice = self.full_time_axis[start_idx:end_idx]
                    
                    # Normalize channel data
                    ch_min, ch_max = self.get_channel_bounds(ch_idx)
                    data_range = ch_max - ch_min
                    if data_range > 0:
                        # Center the data around 0, then scale, then offset
                        normalized_data = ((data - ch_min) / data_range - 0.5) * channel_height * 0.8 * self.stacked_vertical_zoom
                    else:
                        normalized_data = np.zeros_like(data)
                    
                    # Apply vertical offset (to center of channel's space)
                    vertical_offset = idx * channel_height + channel_height * 0.5
                    offset_data = normalized_data + vertical_offset
                    
                    # Update plot item
                    plot_item = self.stacked_plot_items[ch_idx]
                    plot_item.setData(time_slice, offset_data)
        
        # Update window controls to reflect current position
        self.update_window_controls_from_view()
    
    def open_window_settings_dialog(self):
        """Open dialog to configure window display settings"""
        dialog = WindowSettingsDialog(self.display_mode, self)
        if dialog.exec_() == QDialog.Accepted:
            new_mode = dialog.get_mode()
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
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        from PyQt5.QtCore import Qt
        
        # Spacebar toggles play/pause when in autoplay mode
        if event.key() == Qt.Key_Space:
            if self.autoplay_active:
                self.toggle_play_pause()
                event.accept()
                return
        
        # Let the parent class handle other keys
        super().keyPressEvent(event)


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
