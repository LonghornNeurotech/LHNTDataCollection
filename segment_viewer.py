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


class SamplingDialog(QDialog):
    """Dialog for configuring sampling parameters"""
    def __init__(self, current_window_size, current_sampling_rate, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sampling Settings")
        
        layout = QFormLayout()
        
        # Window size in seconds
        self.window_size_spin = QDoubleSpinBox()
        self.window_size_spin.setRange(0.1, 60.0)
        self.window_size_spin.setSingleStep(0.5)
        self.window_size_spin.setValue(current_window_size)
        self.window_size_spin.setDecimals(1)
        self.window_size_spin.setSuffix(" sec")
        layout.addRow("Window Size:", self.window_size_spin)
        
        # Sampling rate
        self.sampling_rate_spin = QSpinBox()
        self.sampling_rate_spin.setRange(1, 10000)
        self.sampling_rate_spin.setSingleStep(1)
        self.sampling_rate_spin.setValue(current_sampling_rate)
        self.sampling_rate_spin.setSuffix(" Hz")
        layout.addRow("Sampling Rate:", self.sampling_rate_spin)
        
        # Info label
        self.info_label = QLabel()
        self.update_info()
        layout.addRow(self.info_label)
        
        # Update info when values change
        self.window_size_spin.valueChanged.connect(self.update_info)
        self.sampling_rate_spin.valueChanged.connect(self.update_info)
        
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        
        self.setLayout(layout)
    
    def update_info(self):
        """Update the info label with calculated values"""
        window_size = self.window_size_spin.value()
        sampling_rate = self.sampling_rate_spin.value()
        samples_per_window = int(window_size * sampling_rate)
        self.info_label.setText(f"Samples per window: {samples_per_window}")
    
    def get_values(self):
        """Return window size and sampling rate"""
        return self.window_size_spin.value(), self.sampling_rate_spin.value()


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
        
        # Plot container (will be replaced when switching modes)
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.main_layout.addWidget(self.plot_container, stretch=1)  # Give plot area most of the space
        
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
        
        # Sampling settings button
        self.sampling_btn = QPushButton("⚙ Sampling Settings")
        self.sampling_btn.clicked.connect(self.open_sampling_dialog)
        nav_layout.addWidget(self.sampling_btn)
        
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
        
        # Enable auto-downsampling for performance
        self.plot_widget.setDownsampling(auto=True, mode='peak')
        
        # Connect range change signal for dynamic updates
        self.plot_widget.sigRangeChanged.connect(self.on_overlay_range_changed)
        
        self.plot_layout.addWidget(self.plot_widget)
        self.plot_widgets = [self.plot_widget]  # Store in list for consistency
        
        # Dictionary to store plot items for each channel
        self.overlay_plot_items = {}
    
    def setup_stacked_mode(self):
        """Setup multiple plot widgets for stacked display"""
        # Clear existing plots
        while self.plot_layout.count():
            child = self.plot_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Create separate plot widget for each channel
        self.plot_widgets = []
        self.stacked_plot_items = {}  # Dictionary to store plot items for each channel
        
        for i in range(self.num_channels):
            plot = pg.PlotWidget()
            plot.setLabel('left', f'Ch{i+1}')
            if i == self.num_channels - 1:
                plot.setLabel('bottom', 'Time (seconds)')
            plot.showGrid(x=True, y=True, alpha=0.3)
            
            # Enable auto-downsampling for performance
            plot.setDownsampling(auto=True, mode='peak')
            
            # Connect range change signal for dynamic updates
            plot.sigRangeChanged.connect(lambda _, ch=i: self.on_stacked_range_changed(ch))
            
            # Don't add to layout yet - will be added dynamically in update_plot_stacked
            self.plot_widgets.append(plot)
    
    def prepare_data(self):
        """Prepare data and calculate window parameters"""
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
        
        # Pre-calculate IQR bounds for entire dataset (per channel)
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
    
    def toggle_channel(self, channel_idx, state):
        """Toggle channel visibility"""
        if not self.file_loaded:
            return
        if state == Qt.Checked:
            self.active_channels.add(channel_idx)
        else:
            self.active_channels.discard(channel_idx)
        self.update_plot()
    
    def update_plot(self):
        """Redraw plot with active channels and IQR-based bounds"""
        if not self.file_loaded:
            return
        if self.display_mode == 'overlay':
            self.update_plot_overlay()
        else:
            self.update_plot_stacked()
    
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
        
        # Get current view range
        view_range = self.plot_widget.viewRange()[0]  # [x_min, x_max]
        start_idx, end_idx = self.get_visible_data_range(view_range)
        
        # Color palette
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w', 'orange']
        
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
            
            # Get the IQR bounds from channel with greatest range
            y_min, y_max = self.get_iqr_bounds_for_channels(self.active_channels)
            
            # Add only active channel plots to layout
            active_list = sorted(self.active_channels)
            for idx, i in enumerate(active_list):
                plot = self.plot_widgets[i]
                plot.clear()
                
                # Update the bottom label - only the last plot should have it
                if idx == len(active_list) - 1:
                    plot.setLabel('bottom', 'Time (seconds)')
                else:
                    plot.setLabel('bottom', '')
                
                # Plot only visible data for this channel
                data = self.raw_data[i, start_idx:end_idx]
                time_slice = self.full_time_axis[start_idx:end_idx]
                color = colors[i % len(colors)]
                plot_item = plot.plot(time_slice, data, pen=pg.mkPen(color=color, width=2))
                
                # Store plot item for dynamic updates
                self.stacked_plot_items[i] = plot_item
                
                # Use the same IQR bounds for all channels (from channel with greatest range)
                plot.setYRange(y_min, y_max, padding=0)
                
                # Set X-axis range to current window
                plot.setXRange(t_start, t_end, padding=0)
                
                # Add to layout with equal stretch
                self.plot_layout.addWidget(plot, stretch=1)
                plot.show()
            
            # Remember which channels are in the layout
            self.last_stacked_channels = self.active_channels.copy()
        else:
            # Update visible data and X-axis bounds
            for i in self.active_channels:
                plot = self.plot_widgets[i]
                # Update X-axis range to current window
                plot.setXRange(t_start, t_end, padding=0)
                
                # Update visible data
                if i in self.stacked_plot_items:
                    data = self.raw_data[i, start_idx:end_idx]
                    time_slice = self.full_time_axis[start_idx:end_idx]
                    self.stacked_plot_items[i].setData(time_slice, data)
    
    def on_stacked_range_changed(self, channel_idx):
        """Called when user zooms or pans in stacked mode - updates visible data for one channel"""
        if not self.file_loaded or channel_idx not in self.active_channels:
            return
        
        plot = self.plot_widgets[channel_idx]
        view_range = plot.viewRange()[0]  # [x_min, x_max]
        start_idx, end_idx = self.get_visible_data_range(view_range)
        
        # Update data for this channel
        if channel_idx in self.stacked_plot_items:
            data = self.raw_data[channel_idx, start_idx:end_idx]
            time_slice = self.full_time_axis[start_idx:end_idx]
            self.stacked_plot_items[channel_idx].setData(time_slice, data)
    
    def on_slider_changed(self, value):
        """Handle slider movement"""
        self.current_window = value
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(value)
        self.spinbox.blockSignals(False)
        self.info_label.setText(f"Window {value} / {self.num_windows-1}")
        # Don't reset Y-axis when navigating windows
        if self.display_mode == 'overlay':
            self.update_plot_overlay(set_y_range=False)
        else:
            self.update_plot_stacked(rebuild_layout=False)
    
    def on_spinbox_changed(self, value):
        """Handle spinbox input"""
        self.current_window = value
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(False)
        self.info_label.setText(f"Window {value} / {self.num_windows-1}")
        # Don't reset Y-axis when navigating windows
        if self.display_mode == 'overlay':
            self.update_plot_overlay(set_y_range=False)
        else:
            self.update_plot_stacked(rebuild_layout=False)
    
    def prev_window(self):
        """Go to previous window"""
        if self.current_window > 0:
            self.spinbox.setValue(self.current_window - 1)
    
    def next_window(self):
        """Go to next window"""
        if self.current_window < self.num_windows - 1:
            self.spinbox.setValue(self.current_window + 1)
    
    def auto_fit_plot(self):
        """Re-fit the plot to IQR bounds (useful after manual zoom/pan)"""
        if not self.file_loaded or not self.active_channels:
            return
        
        # Get window time bounds
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
            # Stacked mode: fit each visible plot with same bounds
            for ch_idx in self.active_channels:
                plot = self.plot_widgets[ch_idx]
                plot.setYRange(y_min, y_max, padding=0)
                plot.setXRange(t_start, t_end, padding=0)
    
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
                
                # Refresh plot
                self.update_plot()
                print(f"Display mode changed to: {self.display_mode}")
    
    def open_sampling_dialog(self):
        """Open dialog to configure sampling parameters"""
        dialog = SamplingDialog(self.window_size_sec, self.sampling_rate, self)
        if dialog.exec_() == QDialog.Accepted:
            new_window_size, new_sampling_rate = dialog.get_values()
            self.window_size_sec = new_window_size
            self.sampling_rate = new_sampling_rate
            
            # Recalculate window parameters
            print(f"Recalculating with window={new_window_size}s, rate={new_sampling_rate}Hz...")
            self.prepare_data()
            
            # Update UI
            self.slider.setMaximum(self.num_windows - 1)
            self.spinbox.setMaximum(self.num_windows - 1)
            self.current_window = min(self.current_window, self.num_windows - 1)
            self.slider.setValue(self.current_window)
            self.spinbox.setValue(self.current_window)
            self.info_label.setText(f"Window {self.current_window} / {self.num_windows-1}")
            self.setWindowTitle(f"Segment Viewer - {self.current_filename} - {self.num_windows} windows, {self.num_channels} channels")
            
            # Refresh plot (X-axis will be updated automatically)
            self.update_plot()
            
            print(f"New window parameters: {self.num_windows} windows, {self.num_samples_per_window} samples per window")


def main():
    # Initialize app with no file
    app = QApplication(sys.argv)
    viewer = SegmentViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
