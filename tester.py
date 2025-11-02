import sys
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import traceback

# =============================================================================
# IMPORT YOUR FILTERING FUNCTIONS HERE
# =============================================================================
# Students: Import your filtering functions from your modules here
# 
# Example:
# from advanced_filtering import spectrum_interpolation_notch, zero_phase_bandpass
# from csp_module import CSPFilter
#
# Uncomment the line below and import your actual functions:
from advanced_filtering import spectrum_interpolation_notch, zero_phase_bandpass

class FilteringTester:
    def __init__(self):
        # Set up BrainFlow synthetic board
        BoardShim.enable_dev_board_logger()
        params = BrainFlowInputParams()
        self.board_id = BoardIds.SYNTHETIC_BOARD.value
        self.board = BoardShim(self.board_id, params)
        
        # Get board info
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        print(f"Sampling rate: {self.sampling_rate} Hz")
        print(f"EEG channels: {self.eeg_channels}")
        
        # Filter parameters (matching your project specs)
        self.lowcut = 5.0
        self.highcut = 35.0
        self.notch_freq = 60.0
        
        # Data buffer settings
        self.buffer_size = int(5 * self.sampling_rate)  # 5 seconds of data
        self.display_size = int(2 * self.sampling_rate)  # Display last 2 seconds
        
        # Initialize data buffers for 8 channels (for visualization)
        self.n_display_channels = min(8, len(self.eeg_channels))
        self.raw_buffer = np.zeros((self.n_display_channels, self.buffer_size))
        self.spectrum_notch_buffer = np.zeros((self.n_display_channels, self.buffer_size))
        self.zero_phase_buffer = np.zeros((self.n_display_channels, self.buffer_size))
        self.full_pipeline_buffer = np.zeros((self.n_display_channels, self.buffer_size))
        
        # Channel visibility tracking (all channels visible by default)
        self.channel_visibility = [True] * self.n_display_channels
        
        # Setup GUI
        self.setup_gui()
        
        # Start board
        self.board.prepare_session()
        self.board.start_stream()
        print("BrainFlow streaming started...")

    def setup_gui(self):
        """Set up the PyQtGraph visualization window"""
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication(sys.argv)
        
        # Main window
        self.win = pg.GraphicsLayoutWidget(show=True, title="Real-time Filtering Tester")
        self.win.resize(1400, 900)
        self.win.setWindowTitle("EEG Filtering Test Framework v7 - Longhorn Neurotech")
        
        # Create plots for each filtering method
        self.plots = {}
        self.curves = {}
        
        # Colors for 8 different channels
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
                      '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']
        
        # Add settings button in the top right
        self.setup_settings_button()
        
        # Row 1: Raw data and spectrum interpolation notch
        self.plots['raw'] = self.win.addPlot(title="Raw Synthetic EEG Data", row=0, col=0)
        self.plots['raw'].setLabel('left', 'Amplitude (μV)')
        self.plots['raw'].setLabel('bottom', 'Samples')
        
        self.plots['spectrum_notch'] = self.win.addPlot(title="Your Spectrum Interpolation Notch Filter", row=0, col=1)
        self.plots['spectrum_notch'].setLabel('left', 'Amplitude (μV)')
        self.plots['spectrum_notch'].setLabel('bottom', 'Samples')
        
        # Row 2: Zero-phase bandpass and full pipeline
        self.plots['zero_phase'] = self.win.addPlot(title="Your Zero-Phase Bandpass Filter", row=1, col=0)
        self.plots['zero_phase'].setLabel('left', 'Amplitude (μV)')
        self.plots['zero_phase'].setLabel('bottom', 'Samples')
        
        self.plots['full_pipeline'] = self.win.addPlot(title="Your Complete Filtering Pipeline", row=1, col=1)
        self.plots['full_pipeline'].setLabel('left', 'Amplitude (μV)')
        self.plots['full_pipeline'].setLabel('bottom', 'Samples')
        
        # Initialize curves for each plot
        for plot_name, plot in self.plots.items():
            self.curves[plot_name] = []
            for i in range(self.n_display_channels):
                curve = plot.plot(pen=pg.mkPen(color=self.colors[i], width=2), 
                                name=f'Ch{self.eeg_channels[i]}')
                self.curves[plot_name].append(curve)
            
            # Add legend
            plot.addLegend(offset=(-10, 10))
            
            # Set Y-axis range
            plot.setYRange(-150, 150)
            
        # Add instruction text
        self.instruction_text = pg.TextItem(
            "Import your filtering functions at the top of this file to see results!\n8 EEG channels displayed • Use Settings button to toggle channel visibility", 
            color='yellow', anchor=(0, 0))
        self.plots['raw'].addItem(self.instruction_text)
        self.instruction_text.setPos(10, 120)
        
        # Add status text
        self.status_text = pg.TextItem("Status: Running...", color='white', anchor=(0, 1))
        self.plots['raw'].addItem(self.status_text)
        self.status_text.setPos(10, -120)
        
        # Timer for updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(50)  # Update every 50ms

    def setup_settings_button(self):
        """Create and position the settings button"""
        # Create a simple button widget
        self.settings_button = QtWidgets.QPushButton("⚙ Settings")
        self.settings_button.setFixedSize(100, 30)
        self.settings_button.setStyleSheet("""
            QPushButton {
                background-color: #4ECDC4;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #45B7D1;
            }
            QPushButton:pressed {
                background-color: #3A9BC1;
            }
        """)
        self.settings_button.clicked.connect(self.show_settings_dialog)
        
        # Add button as a separate widget (avoid grid layout conflicts)
        self.settings_proxy = QtWidgets.QGraphicsProxyWidget()
        self.settings_proxy.setWidget(self.settings_button)
        
        # Add to scene instead of layout to avoid conflicts
        self.win.scene().addItem(self.settings_proxy)
        self.settings_proxy.setPos(1290, 10)

    def show_settings_dialog(self):
        """Show the channel visibility settings dialog"""
        dialog = ChannelSettingsDialog(self.n_display_channels, self.eeg_channels, 
                                     self.channel_visibility, self.colors, self.win)
        dialog.channel_changed.connect(self.update_channel_visibility)
        dialog.show()

    def update_channel_visibility(self, channel_idx, visible):
        """Update visibility of a specific channel"""
        self.channel_visibility[channel_idx] = visible
        
        # Update all plot curves for this channel
        for plot_name in self.curves:
            curve = self.curves[plot_name][channel_idx]
            if visible:
                curve.show()
            else:
                curve.hide()

    # =========================================================================
    # REPLACE THESE PLACEHOLDER FUNCTIONS WITH YOUR IMPLEMENTATIONS
    # =========================================================================
    
    def apply_spectrum_interpolation_notch(self, signal):
        return spectrum_interpolation_notch(signal, self.sampling_rate, self.notch_freq)

    def apply_zero_phase_bandpass(self, signal):
        return zero_phase_bandpass(signal, self.lowcut, self.highcut, self.sampling_rate)

    def apply_full_filtering_pipeline(self, signal):
        filtered = self.apply_zero_phase_bandpass(signal)
        filtered = self.apply_spectrum_interpolation_notch(filtered)
        return filtered

    # =========================================================================
    # TESTER FRAMEWORK CODE (DO NOT MODIFY)
    # =========================================================================

    def update_plots(self):
        """Update all plots with new data"""
        try:
            # Get new data from board
            data = self.board.get_board_data()
            
            if data.shape[1] == 0:
                return
            
            # Extract EEG data for display channels
            eeg_data = data[self.eeg_channels[:self.n_display_channels], :]
            
            if eeg_data.shape[1] == 0:
                return
            
            # Update buffers
            new_samples = eeg_data.shape[1]
            
            # Shift old data and add new data
            self.raw_buffer = np.roll(self.raw_buffer, -new_samples, axis=1)
            self.raw_buffer[:, -new_samples:] = eeg_data
            
            # Apply filtering to each channel
            for ch_idx in range(self.n_display_channels):
                channel_data = self.raw_buffer[ch_idx, :].copy()
                
                # Apply your filtering functions
                spectrum_filtered = self.apply_spectrum_interpolation_notch(channel_data)
                self.spectrum_notch_buffer[ch_idx, :] = spectrum_filtered
                
                zero_phase_filtered = self.apply_zero_phase_bandpass(channel_data)
                self.zero_phase_buffer[ch_idx, :] = zero_phase_filtered
                
                full_pipeline_filtered = self.apply_full_filtering_pipeline(channel_data)
                self.full_pipeline_buffer[ch_idx, :] = full_pipeline_filtered
            
            # Update plot data (show last display_size samples)
            x_data = np.arange(self.display_size)
            
            # Update curves
            for ch_idx in range(self.n_display_channels):
                # Only update visible channels
                if self.channel_visibility[ch_idx]:
                    # Raw data
                    self.curves['raw'][ch_idx].setData(x_data, 
                        self.raw_buffer[ch_idx, -self.display_size:])
                    
                    # Spectrum interpolation notch
                    self.curves['spectrum_notch'][ch_idx].setData(x_data, 
                        self.spectrum_notch_buffer[ch_idx, -self.display_size:])
                    
                    # Zero-phase bandpass
                    self.curves['zero_phase'][ch_idx].setData(x_data, 
                        self.zero_phase_buffer[ch_idx, -self.display_size:])
                    
                    # Full pipeline
                    self.curves['full_pipeline'][ch_idx].setData(x_data, 
                        self.full_pipeline_buffer[ch_idx, -self.display_size:])
            
            # Update status
            current_time = time.strftime("%H:%M:%S")
            self.status_text.setText(f"Status: Running | Time: {current_time} | Samples: {data.shape[1]}")
            
        except Exception as e:
            print(f"Error in update_plots: {e}")
            traceback.print_exc()

    def compute_signal_stats(self):
        """Compute and display signal statistics for verification"""
        try:
            print(f"\n" + "="*50)
            print("SIGNAL STATISTICS (Last 2 seconds)")
            print("="*50)
            
            for ch_idx in range(self.n_display_channels):
                if not self.channel_visibility[ch_idx]:
                    continue
                    
                ch_name = f"Channel {self.eeg_channels[ch_idx]}"
                
                raw_data = self.raw_buffer[ch_idx, -self.display_size:]
                spectrum_data = self.spectrum_notch_buffer[ch_idx, -self.display_size:]
                zero_phase_data = self.zero_phase_buffer[ch_idx, -self.display_size:]
                full_data = self.full_pipeline_buffer[ch_idx, -self.display_size:]
                
                print(f"\n{ch_name}:")
                print(f"  Raw:              μ={np.mean(raw_data):6.2f}, σ={np.std(raw_data):6.2f}")
                print(f"  Spectrum Notch:   μ={np.mean(spectrum_data):6.2f}, σ={np.std(spectrum_data):6.2f}")
                print(f"  Zero-phase BP:    μ={np.mean(zero_phase_data):6.2f}, σ={np.std(zero_phase_data):6.2f}")
                print(f"  Full Pipeline:    μ={np.mean(full_data):6.2f}, σ={np.std(full_data):6.2f}")
            
        except Exception as e:
            print(f"Error computing statistics: {e}")

    def run(self):
        """Start the application"""
        try:
            print("\n" + "="*70)
            print("EEG FILTERING TEST FRAMEWORK v7 - LONGHORN NEUROTECH")
            print("="*70)
            print("INSTRUCTIONS:")
            print("1. Import your filtering functions at the top of this file")
            print("2. Replace the placeholder functions with your implementations")
            print("3. Run this script to see real-time results")
            print("4. Use the Settings button to control channel visibility")
            print("5. Compare raw data vs your filtered data")
            print(f"\nSYSTEM INFO:")
            print(f"Channels displayed: {self.n_display_channels}")
            print(f"Sampling rate: {self.sampling_rate} Hz")
            print(f"Filter specs: {self.lowcut}-{self.highcut} Hz bandpass, {self.notch_freq} Hz notch")
            print(f"Data source: BrainFlow Synthetic Board")
            print("\nPress Ctrl+C to stop...")
            print("="*70)
            
            # Start timer for periodic statistics
            stats_timer = QtCore.QTimer()
            stats_timer.timeout.connect(self.compute_signal_stats)
            stats_timer.start(15000)  # Every 15 seconds
            
            # Run the application
            self.app.exec_()
            
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'board'):
                self.board.stop_stream()
                self.board.release_session()
                print("BrainFlow streaming stopped.")
        except Exception as e:
            print(f"Error during cleanup: {e}")


class ChannelSettingsDialog(QtWidgets.QDialog):
    """Dialog for controlling channel visibility settings"""
    
    channel_changed = QtCore.pyqtSignal(int, bool)  # channel_idx, visible
    
    def __init__(self, n_channels, eeg_channels, visibility, colors, parent=None):
        super().__init__(parent)
        self.n_channels = n_channels
        self.eeg_channels = eeg_channels
        self.visibility = visibility.copy()
        self.colors = colors
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the dialog UI"""
        self.setWindowTitle("Channel Visibility Settings")
        self.setFixedSize(300, 350)
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: white;
            }
            QCheckBox {
                color: white;
                font-size: 12px;
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #555;
                background-color: #333;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #4ECDC4;
                background-color: #4ECDC4;
            }
            QPushButton {
                background-color: #4ECDC4;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45B7D1;
            }
        """)
        
        layout = QtWidgets.QVBoxLayout()
        
        # Title
        title = QtWidgets.QLabel("Select EEG Channels to Display:")
        title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Channel checkboxes
        self.checkboxes = []
        for i in range(self.n_channels):
            checkbox = QtWidgets.QCheckBox(f"Channel {self.eeg_channels[i]}")
            checkbox.setChecked(self.visibility[i])
            
            checkbox.stateChanged.connect(lambda state, idx=i: self.on_checkbox_changed(idx, state))
            self.checkboxes.append(checkbox)
            layout.addWidget(checkbox)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        # Select All button
        select_all_btn = QtWidgets.QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all)
        button_layout.addWidget(select_all_btn)
        
        # Clear All button
        clear_all_btn = QtWidgets.QPushButton("Clear All")
        clear_all_btn.clicked.connect(self.clear_all)
        button_layout.addWidget(clear_all_btn)
        
        layout.addLayout(button_layout)
        
        # Close button
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
    
    def on_checkbox_changed(self, channel_idx, state):
        """Handle checkbox state change"""
        visible = state == QtCore.Qt.Checked
        self.visibility[channel_idx] = visible
        self.channel_changed.emit(channel_idx, visible)
    
    def select_all(self):
        """Select all channels"""
        for i, checkbox in enumerate(self.checkboxes):
            checkbox.setChecked(True)
    
    def clear_all(self):
        """Clear all channels"""
        for i, checkbox in enumerate(self.checkboxes):
            checkbox.setChecked(False)


def main():
    """Main function to run the filtering tester"""
    print("Starting EEG Filtering Test Framework v7...")
    print("Make sure you have imported your filtering functions!")
    
    try:
        tester = FilteringTester()
        tester.run()
    except Exception as e:
        print(f"Failed to start filtering tester: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()