import sys
import numpy as np
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg
import mne

class RealtimeEEGViewer(QtWidgets.QMainWindow):
    def __init__(self, edf_path):
        super().__init__()
        
        # Load EDF file
        print("Loading EDF file...")
        self.raw = mne.io.read_raw_edf(edf_path, preload=True)
        self.data, self.times = self.raw[:, :]  # Get all data
        self.sfreq = self.raw.info['sfreq']
        self.n_channels = len(self.raw.ch_names)
        
        print(f"Loaded {self.n_channels} channels at {self.sfreq} Hz")
        print(f"Duration: {len(self.times) / self.sfreq:.2f} seconds")
        
        # Settings for real-time simulation
        self.window_size = 5.0  # seconds to display
        self.update_interval = 50  # milliseconds
        self.samples_per_update = int(self.sfreq * self.update_interval / 1000)
        self.current_sample = 0
        
        # Store data for display
        self.display_samples = int(self.window_size * self.sfreq)
        self.display_data = np.zeros((self.n_channels, self.display_samples))
        
        self.initUI()
        
        # Timer for real-time updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(self.update_interval)
        
    def initUI(self):
        self.setWindowTitle('Real-time EEG Viewer')
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout()
        
        # Control panel
        control_layout = QtWidgets.QHBoxLayout()
        
        self.play_button = QtWidgets.QPushButton('Pause')
        self.play_button.clicked.connect(self.toggle_playback)
        control_layout.addWidget(self.play_button)
        
        self.reset_button = QtWidgets.QPushButton('Reset')
        self.reset_button.clicked.connect(self.reset_playback)
        control_layout.addWidget(self.reset_button)
        
        self.speed_label = QtWidgets.QLabel('Speed: 1.0x')
        control_layout.addWidget(self.speed_label)
        
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(50)
        self.speed_slider.setValue(10)
        self.speed_slider.valueChanged.connect(self.change_speed)
        control_layout.addWidget(self.speed_slider)
        
        self.time_label = QtWidgets.QLabel('Time: 0.0 s')
        control_layout.addWidget(self.time_label)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # Plot widget
        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget)
        
        central_widget.setLayout(layout)
        
        # Create plots for each channel
        self.plots = []
        self.curves = []
        
        # Display up to 16 channels (or all if fewer)
        channels_to_display = min(16, self.n_channels)
        
        for i in range(channels_to_display):
            if i > 0:
                plot = self.plot_widget.addPlot(row=i, col=0)
                plot.setXLink(self.plots[0])  # Link x-axes
            else:
                plot = self.plot_widget.addPlot(row=i, col=0)
                plot.setLabel('bottom', 'Time', units='s')
            
            plot.setLabel('left', self.raw.ch_names[i])
            plot.showGrid(x=True, y=True, alpha=0.3)
            
            curve = plot.plot(pen=pg.mkPen(color=(100, 200, 255), width=1))
            
            self.plots.append(plot)
            self.curves.append(curve)
        
        self.time_axis = np.linspace(0, self.window_size, self.display_samples)
        
    def update_plot(self):
        # Check if we've reached the end
        if self.current_sample >= self.data.shape[1]:
            self.current_sample = 0  # Loop back to start
        
        # Get new samples
        end_sample = min(self.current_sample + self.samples_per_update, 
                        self.data.shape[1])
        new_samples = end_sample - self.current_sample
        
        if new_samples > 0:
            # Shift display data left
            self.display_data = np.roll(self.display_data, -new_samples, axis=1)
            
            # Add new data
            self.display_data[:, -new_samples:] = self.data[:, 
                                                            self.current_sample:end_sample]
            
            # Update plots
            for i, curve in enumerate(self.curves):
                curve.setData(self.time_axis, self.display_data[i])
            
            self.current_sample = end_sample
            
            # Update time label
            current_time = self.current_sample / self.sfreq
            self.time_label.setText(f'Time: {current_time:.2f} s / {self.times[-1]:.2f} s')
    
    def toggle_playback(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText('Play')
        else:
            self.timer.start(self.update_interval)
            self.play_button.setText('Pause')
    
    def reset_playback(self):
        self.current_sample = 0
        self.display_data = np.zeros((self.n_channels, self.display_samples))
        if not self.timer.isActive():
            self.timer.start(self.update_interval)
            self.play_button.setText('Pause')
    
    def change_speed(self, value):
        speed = value / 10.0
        self.speed_label.setText(f'Speed: {speed:.1f}x')
        self.samples_per_update = int(self.sfreq * self.update_interval / 1000 * speed)

def main():
    # Path to your EDF file
    edf_path = r"C:\\Users\\d0nmega\\Downloads\\sub-001_ses-01_task-szMonitoring_run-01_eeg.edf"
    
    app = QtWidgets.QApplication(sys.argv)
    viewer = RealtimeEEGViewer(edf_path)
    viewer.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()