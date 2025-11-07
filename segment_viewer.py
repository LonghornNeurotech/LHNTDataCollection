#!/usr/bin/env python3
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
    QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt

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
        
        # Vertical zoom slider (Y-axis zoom) on the left
        self.vertical_zoom_slider = QSlider(Qt.Vertical)
        self.vertical_zoom_slider.setMinimum(10)  # 10% zoom (zoomed out)
        self.vertical_zoom_slider.setMaximum(500)  # 500% zoom (zoomed in)
        self.vertical_zoom_slider.setValue(100)  # 100% = normal
        self.vertical_zoom_slider.setTickPosition(QSlider.TicksRight)
        self.vertical_zoom_slider.setTickInterval(50)
        self.vertical_zoom_slider.valueChanged.connect(self.on_vertical_zoom_changed)
        self.vertical_zoom_slider.setEnabled(False)  # Disabled until file loaded
        plot_and_zoom_layout.addWidget(self.vertical_zoom_slider)
        
        # Right side: plot container + horizontal zoom slider
        plot_column_widget = QWidget()
        plot_column_layout = QVBoxLayout(plot_column_widget)
        plot_column_layout.setContentsMargins(0, 0, 0, 0)
        
        # Plot container (will be replaced when switching modes)
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
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
        
        plot_column_layout.addWidget(horizontal_zoom_widget)
        
        plot_and_zoom_layout.addWidget(plot_column_widget, stretch=1)
        
        self.main_layout.addWidget(plot_and_zoom_widget, stretch=1)
        
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
            self.horizontal_zoom_slider.setEnabled(True)
            self.window_size_spinbox.setEnabled(True)
            self.sampling_rate_spinbox.setEnabled(True)
            
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
        """Setup multiple plot widgets for stacked display with linked X-axes"""
        # Clear existing plots
        while self.plot_layout.count():
            child = self.plot_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Create separate plot widget for each channel
        self.plot_widgets = []
        self.stacked_plot_items = {}  # Dictionary to store plot items for each channel
        
        # Keep track of first plot to link X-axes
        first_plot = None
        
        for i in range(self.num_channels):
            plot = pg.PlotWidget()
            plot.setLabel('left', f'Ch{i+1}')
            
            # Only show X-axis on the last channel
            if i == self.num_channels - 1:
                plot.setLabel('bottom', 'Time (seconds)')
            else:
                # Hide X-axis for non-bottom plots to save space
                plot.getAxis('bottom').setStyle(showValues=False)
                plot.getAxis('bottom').setHeight(0)  # Remove space allocated for axis
            
            plot.showGrid(x=True, y=True, alpha=0.3)
            
            # Link X-axis to first plot (so all channels share same time axis)
            if i == 0:
                first_plot = plot
            else:
                # Link this plot's X-axis to the first plot
                plot.setXLink(first_plot)
            
            # Disable default mouse wheel behavior and add custom zoom
            plot.setMouseEnabled(x=True, y=True)
            view_box = plot.getViewBox()
            view_box.setMouseMode(pg.ViewBox.RectMode)
            
            # Install custom wheel event using a closure that properly captures view_box
            def make_wheel_handler(vb):
                return lambda event: self.custom_wheel_event(vb, event)
            view_box.wheelEvent = make_wheel_handler(view_box)
            
            # Connect range change signal for dynamic updates
            # Only connect for first plot since X-axes are linked
            if i == 0:
                def make_range_handler(ch):
                    return lambda: self.on_stacked_range_changed(ch)
                plot.sigRangeChanged.connect(make_range_handler(i))
            
            # Don't add to layout yet - will be added dynamically in update_plot_stacked
            self.plot_widgets.append(plot)
    
    def prepare_data(self):
        """Prepare data and calculate window parameters and IQR bounds"""
        self.recalculate_windows()
        
        # Pre-calculate IQR bounds for entire dataset (per channel)
        # This is expensive so only do it once when file is loaded
        self.channel_iqr_bounds = {}
        for ch_idx in range(self.num_channels):
            # Get all data for this channel
            channel_data = self.raw_data[ch_idx, :]
            q1 = np.quantile(channel_data, 0.25)
            q3 = np.quantile(channel_data, 0.75)
            iqr = q3 - q1
            y_min = q1 - iqr * 5.0
            y_max = q3 + iqr * 5.0
            self.channel_iqr_bounds[ch_idx] = (y_min, y_max)
    
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
            # In stacked mode, all X-axes are linked, so use first plot
            view_range = self.plot_widgets[0].viewRange()[0]
        
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
    
    def on_vertical_zoom_changed(self, value):
        """Handle vertical zoom slider changes (Y-axis zoom)
        
        Args:
            value: Zoom percentage (10-500, where 100 = normal)
        """
        if not self.file_loaded or not self.active_channels:
            return
        
        # Get the zoom factor (1.0 = 100%)
        zoom_factor = value / 100.0
        
        if self.display_mode == 'overlay':
            # Get current IQR bounds and scale them
            y_min, y_max = self.get_iqr_bounds_for_channels(self.active_channels)
            y_center = (y_min + y_max) / 2.0
            y_range = (y_max - y_min) / zoom_factor
            
            new_y_min = y_center - y_range / 2.0
            new_y_max = y_center + y_range / 2.0
            self.plot_widget.setYRange(new_y_min, new_y_max, padding=0)
        else:
            # Stacked mode: scale each channel's Y range independently
            for ch_idx in self.active_channels:
                plot = self.plot_widgets[ch_idx]
                ch_y_min, ch_y_max = self.channel_iqr_bounds[ch_idx]
                y_center = (ch_y_min + ch_y_max) / 2.0
                y_range = (ch_y_max - ch_y_min) / zoom_factor
                
                new_y_min = y_center - y_range / 2.0
                new_y_max = y_center + y_range / 2.0
                plot.setYRange(new_y_min, new_y_max, padding=0)
    
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
        
        # Apply current vertical zoom value after any plot update
        if rebuild:
            current_zoom = self.vertical_zoom_slider.value()
            if current_zoom != 100:  # Only apply if zoom is not at default
                self.on_vertical_zoom_changed(current_zoom)
    
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
            y_min, y_max = self.get_iqr_bounds_for_channels(self.active_channels)
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
        """Update plots in stacked mode - plots only visible data
        
        Args:
            rebuild_layout: If True, rebuild layout (when channels change). 
                          If False, just update X-axis bounds (when navigating windows).
        """
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w', 'orange']
        
        # Get window time bounds
        start_sample, end_sample = self.get_window_bounds(self.current_window)
        t_start = self.full_time_axis[start_sample]
        t_end = self.full_time_axis[min(end_sample - 1, len(self.full_time_axis) - 1)]
        
        # Get visible data range with padding
        view_range = [t_start, t_end]
        start_idx, end_idx = self.get_visible_data_range(view_range)
        
        # Check if we need to rebuild layout (channels changed)
        if rebuild_layout or self.active_channels != self.last_stacked_channels:
            # Clear the layout
            while self.plot_layout.count():
                child = self.plot_layout.takeAt(0)
                if child.widget():
                    # Remove from layout but don't delete the widget
                    child.widget().setParent(None)
            
            # Clear stacked plot items
            self.stacked_plot_items.clear()
            
            # Add only active channel plots to layout
            active_list = sorted(self.active_channels)
            for idx, i in enumerate(active_list):
                plot = self.plot_widgets[i]
                plot.clear()
                
                # Update X-axis visibility - only the last visible plot should show it
                if idx == len(active_list) - 1:
                    # This is the bottom plot - show X-axis
                    plot.setLabel('bottom', 'Time (seconds)')
                    plot.getAxis('bottom').setStyle(showValues=True)
                    plot.getAxis('bottom').setHeight(None)  # Use default height
                else:
                    # Not the bottom plot - hide X-axis to save space
                    plot.setLabel('bottom', '')
                    plot.getAxis('bottom').setStyle(showValues=False)
                    plot.getAxis('bottom').setHeight(0)
                
                # Plot only visible data for this channel
                data = self.raw_data[i, start_idx:end_idx]
                time_slice = self.full_time_axis[start_idx:end_idx]
                color = colors[i % len(colors)]
                plot_item = plot.plot(time_slice, data, pen=pg.mkPen(color=color, width=2))
                
                # Store plot item for dynamic updates
                self.stacked_plot_items[i] = plot_item
                
                # Use individual channel's IQR bounds for separate Y-axis ranges
                ch_y_min, ch_y_max = self.channel_iqr_bounds[i]
                plot.setYRange(ch_y_min, ch_y_max, padding=0)
                
                # Set X-axis range to current window (only needed on first since they're linked)
                if idx == 0:
                    plot.setXRange(t_start, t_end, padding=0)
                
                # Add to layout with equal stretch
                self.plot_layout.addWidget(plot, stretch=1)
                plot.show()
            
            # Remember which channels are in the layout
            self.last_stacked_channels = self.active_channels.copy()
        else:
            # Update X-axis bounds (only on first plot since they're linked)
            self.plot_widgets[list(self.active_channels)[0]].setXRange(t_start, t_end, padding=0)
            
            # Update visible data for all channels
            for i in self.active_channels:
                if i in self.stacked_plot_items:
                    data = self.raw_data[i, start_idx:end_idx]
                    time_slice = self.full_time_axis[start_idx:end_idx]
                    self.stacked_plot_items[i].setData(time_slice, data)
    
    def on_stacked_range_changed(self, channel_idx):
        """Called when user zooms or pans in stacked mode - updates visible data for all channels
        
        Since X-axes are linked, we only need to connect this to the first plot.
        When it fires, we update all active channels.
        """
        if not self.file_loaded or not self.active_channels:
            return
        
        # Update window controls to match current view
        self.update_window_controls_from_view()
        
        # Get the X range from first plot (all are linked)
        plot = self.plot_widgets[0]
        view_range = plot.viewRange()[0]  # [x_min, x_max]
        start_idx, end_idx = self.get_visible_data_range(view_range)
        
        # Update data for all active channels
        for ch_idx in self.active_channels:
            if ch_idx in self.stacked_plot_items:
                data = self.raw_data[ch_idx, start_idx:end_idx]
                time_slice = self.full_time_axis[start_idx:end_idx]
                self.stacked_plot_items[ch_idx].setData(time_slice, data)
    
    def navigate_to_window(self, value):
        """Navigate to a specific window and update the view
        
        Args:
            value: Window index to navigate to
        """
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
            self.plot_widgets[0].setXRange(t_start, t_end, padding=0)
        
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
        
        # Get pre-calculated IQR bounds from channel with greatest range
        y_min, y_max = self.get_iqr_bounds_for_channels(self.active_channels)
        
        if self.display_mode == 'overlay':
            # Reset to show current window time range and IQR bounds
            self.plot_widget.setYRange(y_min, y_max, padding=0)
            self.plot_widget.setXRange(t_start, t_end, padding=0)
        else:
            # Stacked mode: X-axes are linked, set each channel's own Y bounds
            self.plot_widgets[0].setXRange(t_start, t_end, padding=0)
            for ch_idx in self.active_channels:
                plot = self.plot_widgets[ch_idx]
                # Get individual channel's IQR bounds
                ch_y_min, ch_y_max = self.channel_iqr_bounds[ch_idx]
                plot.setYRange(ch_y_min, ch_y_max, padding=0)
    
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


def main():
    # Initialize app with no file
    app = QApplication(sys.argv)
    viewer = SegmentViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
