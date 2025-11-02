import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pickle
from scipy.signal import butter, filtfilt, welch
from scipy.fft import rfft, rfftfreq, irfft
from scipy.linalg import eig
import os

class FilteringFunctions:
    """Advanced filtering implementations"""
    
    @staticmethod
    def spectrum_interpolation_notch(signal, fs, notch_freq=60.0, notch_width=2):
        """Remove line noise using spectrum interpolation"""
        fft_signal = rfft(signal)
        freqs = rfftfreq(len(signal), 1/fs)
        noise_bin = np.argmin(np.abs(freqs - notch_freq))
        
        if noise_bin > notch_width and noise_bin < len(fft_signal) - notch_width:
            fft_signal[noise_bin] = (fft_signal[noise_bin - notch_width] + 
                                     fft_signal[noise_bin + notch_width]) / 2
        
        return irfft(fft_signal, len(signal))
    
    @staticmethod
    def zero_phase_bandpass(signal, lowcut, highcut, fs, order=4):
        """Apply zero-phase bandpass filter using filtfilt"""
        nyquist = fs / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)

class CSPFilter:
    """Common Spatial Patterns implementation"""
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.filters = None
        self.eigenvalues = None
        
    def fit(self, X_left, X_right):
        """Train CSP filters"""
        cov_left = []
        cov_right = []
        
        for trial in X_left:
            C = trial @ trial.T
            C = C / np.trace(C)
            cov_left.append(C)
            
        for trial in X_right:
            C = trial @ trial.T
            C = C / np.trace(C)
            cov_right.append(C)
        
        C_left = np.mean(cov_left, axis=0)
        C_right = np.mean(cov_right, axis=0)
        C_composite = C_left + C_right
        
        eigenvalues, eigenvectors = eig(C_left, C_composite)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx].real
        eigenvectors = eigenvectors[:, idx].real
        
        m = self.n_components
        self.filters = np.column_stack([
            eigenvectors[:, :m],
            eigenvectors[:, -m:]
        ])
        self.eigenvalues = eigenvalues
        return self
    
    def transform(self, X):
        """Apply CSP transformation"""
        if self.filters is None:
            raise ValueError("CSP not fitted yet!")
            
        if X.ndim == 2:
            Z = self.filters.T @ X
            variances = np.var(Z, axis=1)
            total_var = np.sum(variances)
            return np.log(variances / total_var) if total_var > 0 else np.zeros(len(variances))
        else:
            features = []
            for trial in X:
                Z = self.filters.T @ trial
                variances = np.var(Z, axis=1)
                total_var = np.sum(variances)
                feat = np.log(variances / total_var) if total_var > 0 else np.zeros(len(variances))
                features.append(feat)
            return np.array(features)

class FilteringPage(QWidget):
    """Page for visualizing all filtering steps"""
    
    def __init__(self):
        super().__init__()
        self.fs = 125
        self.current_data = None
        self.csp_filter = CSPFilter(n_components=2)
        self.csp_trained = False
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Control panel
        control_panel = QHBoxLayout()
        
        self.spin_window = QSpinBox()
        self.spin_window.setMinimum(0)
        self.spin_window.setMaximum(0)
        self.spin_window.valueChanged.connect(self.update_display)
        control_panel.addWidget(QLabel("Window:"))
        control_panel.addWidget(self.spin_window)
        
        self.label_info = QLabel("No data loaded")
        control_panel.addWidget(self.label_info)
        control_panel.addStretch()
        
        self.check_show_all = QCheckBox("Show All Channels")
        self.check_show_all.setChecked(False)
        self.check_show_all.stateChanged.connect(self.update_display)
        control_panel.addWidget(self.check_show_all)
        
        layout.addLayout(control_panel)
        
        # Create figure for visualizations
        self.figure = Figure(figsize=(16, 10))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
    
    def load_data(self, data_dict):
        """Load data from main window"""
        self.current_data = data_dict
        self.spin_window.setMaximum(len(data_dict['windows']) - 1)
        self.spin_window.setValue(0)
        self.update_display()
    
    def update_display(self):
        """Update all filter visualizations"""
        if self.current_data is None:
            return
        
        self.figure.clear()
        
        # Get current window
        window_idx = self.spin_window.value()
        window = self.current_data['windows'][window_idx]
        
        if window.ndim == 3:
            window = window[0]
        
        n_channels, n_samples = window.shape
        time = np.arange(n_samples) / self.fs
        
        # Determine how many channels to show
        show_all = self.check_show_all.isChecked()
        channels_to_show = n_channels if show_all else min(8, n_channels)
        
        # Create subplots
        ax1 = self.figure.add_subplot(2, 4, 1)  # Raw
        ax2 = self.figure.add_subplot(2, 4, 2)  # After Bandpass
        ax3 = self.figure.add_subplot(2, 4, 3)  # After Notch
        ax4 = self.figure.add_subplot(2, 4, 4)  # After CSP (if trained)
        ax5 = self.figure.add_subplot(2, 4, 5)  # PSD comparison
        ax6 = self.figure.add_subplot(2, 4, 6)  # Phase response
        ax7 = self.figure.add_subplot(2, 4, 7)  # CSP Topoplot 1
        ax8 = self.figure.add_subplot(2, 4, 8)  # CSP Topoplot 2
        
        # Update label
        label_text = f"Window {window_idx+1}/{len(self.current_data['windows'])}"
        if 'labels' in self.current_data and self.current_data['labels'] is not None:
            label = self.current_data['labels'][window_idx]
            label_text += f" | Label: {'Left' if label == 0 else 'Right'}"
        self.label_info.setText(label_text)
        
        # Apply filters step by step
        filtered_bandpass = np.zeros_like(window)
        filtered_notch = np.zeros_like(window)
        
        for ch in range(n_channels):
            # Bandpass filter
            filtered_bandpass[ch] = FilteringFunctions.zero_phase_bandpass(
                window[ch], 5.0, 35.0, self.fs, order=4
            )
            # Notch filter
            filtered_notch[ch] = FilteringFunctions.spectrum_interpolation_notch(
                filtered_bandpass[ch], self.fs, notch_freq=60.0
            )
        
        # Calculate spacing for channel display
        spacing = np.std(window) * 3
        
        # 1. Raw Signal
        ax1.clear()
        for i in range(channels_to_show):
            offset = i * spacing
            ax1.plot(time, window[i] - offset, 'b-', alpha=0.7, linewidth=0.8)
            ax1.text(-0.05, -offset, f'Ch{i+1}', fontsize=8, ha='right', va='center')
        ax1.set_title('Raw EEG Signal')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Channels')
        ax1.grid(True, alpha=0.3)
        ax1.set_yticks([])
        
        # 2. After Bandpass
        ax2.clear()
        for i in range(channels_to_show):
            offset = i * spacing
            ax2.plot(time, filtered_bandpass[i] - offset, 'g-', alpha=0.7, linewidth=0.8)
            ax2.text(-0.05, -offset, f'Ch{i+1}', fontsize=8, ha='right', va='center')
        ax2.set_title('After Bandpass (5-35 Hz)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Channels')
        ax2.grid(True, alpha=0.3)
        ax2.set_yticks([])
        
        # 3. After Notch
        ax3.clear()
        for i in range(channels_to_show):
            offset = i * spacing
            ax3.plot(time, filtered_notch[i] - offset, 'r-', alpha=0.7, linewidth=0.8)
            ax3.text(-0.05, -offset, f'Ch{i+1}', fontsize=8, ha='right', va='center')
        ax3.set_title('After Spectrum Interpolation (60Hz removed)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Channels')
        ax3.grid(True, alpha=0.3)
        ax3.set_yticks([])
        
        # 4. After CSP (if trained)
        if self.csp_trained and self.csp_filter.filters is not None:
            csp_signals = self.csp_filter.filters.T @ filtered_notch
            ax4.clear()
            for i in range(csp_signals.shape[0]):
                offset = i * spacing
                ax4.plot(time, csp_signals[i] - offset, 'm-', alpha=0.7, linewidth=0.8)
                ax4.text(-0.05, -offset, f'CSP{i+1}', fontsize=8, ha='right', va='center')
            ax4.set_title('After CSP Spatial Filtering')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('CSP Components')
            ax4.grid(True, alpha=0.3)
            ax4.set_yticks([])
        else:
            ax4.text(0.5, 0.5, 'CSP Not Trained\nUse Train CSP button', 
                    ha='center', va='center', fontsize=12)
            ax4.set_xticks([])
            ax4.set_yticks([])
        
        # 5. PSD Comparison (Channel 1)
        ax5.clear()
        freqs_raw, psd_raw = welch(window[0], self.fs, nperseg=min(256, n_samples))
        freqs_filt, psd_filt = welch(filtered_notch[0], self.fs, nperseg=min(256, n_samples))
        
        ax5.semilogy(freqs_raw, psd_raw, 'b-', label='Raw', alpha=0.7)
        ax5.semilogy(freqs_filt, psd_filt, 'r-', label='Filtered', alpha=0.7)
        ax5.axvline(60, color='k', linestyle='--', alpha=0.5, label='60 Hz')
        ax5.set_title('Power Spectral Density (Ch1)')
        ax5.set_xlabel('Frequency (Hz)')
        ax5.set_ylabel('PSD')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Phase Response
        ax6.clear()
        # Show phase difference between raw and filtered
        fft_raw = rfft(window[0])
        fft_filt = rfft(filtered_notch[0])
        freqs = rfftfreq(n_samples, 1/self.fs)
        
        mask = np.abs(fft_raw) > 1e-10
        phase_diff = np.angle(fft_filt[mask] / fft_raw[mask])
        
        ax6.plot(freqs[mask], np.degrees(phase_diff), 'g-', alpha=0.7)
        ax6.set_title('Phase Preservation')
        ax6.set_xlabel('Frequency (Hz)')
        ax6.set_ylabel('Phase Shift (degrees)')
        ax6.grid(True, alpha=0.3)
        ax6.axhline(0, color='k', linestyle='--', alpha=0.3)
        
        # 7-8. CSP Topoplots
        if self.csp_trained and self.csp_filter.filters is not None:
            self._plot_topoplot(ax7, self.csp_filter.filters[:, 0], 
                              'CSP Filter 1 (Left)', n_channels)
            self._plot_topoplot(ax8, self.csp_filter.filters[:, -1], 
                              'CSP Filter 2 (Right)', n_channels)
        else:
            for ax in [ax7, ax8]:
                ax.text(0.5, 0.5, 'Train CSP\nto see topoplots', 
                       ha='center', va='center', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def _plot_topoplot(self, ax, weights, title, n_channels):
        """Plot spatial pattern"""
        ax.clear()
        
        # Create positions based on channel count
        if n_channels == 8:
            positions = [
                (0.3, 0.8), (0.7, 0.8),
                (0.2, 0.5), (0.8, 0.5),
                (0.2, 0.3), (0.8, 0.3),
                (0.3, 0.1), (0.7, 0.1),
            ]
        elif n_channels == 16:
            positions = []
            for i in range(4):
                for j in range(4):
                    positions.append((j*0.25 + 0.125, 1 - i*0.25 - 0.125))
        else:
            angles = np.linspace(0, 2*np.pi, n_channels, endpoint=False)
            positions = [(0.5 + 0.3*np.cos(a), 0.5 + 0.3*np.sin(a)) for a in angles]
        
        vmax = np.max(np.abs(weights)) if np.any(weights) else 1
        norm_weights = weights / vmax if vmax > 0 else weights
        
        for i, (x, y) in enumerate(positions):
            color = plt.cm.RdBu_r((norm_weights[i] + 1) / 2)
            circle = plt.Circle((x, y), 0.06, color=color, ec='black', lw=1)
            ax.add_patch(circle)
            ax.text(x, y, f'{i+1}', ha='center', va='center', fontsize=8)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title)

class DataValidationPage(QWidget):
    """Page for data quality validation"""
    
    def __init__(self):
        super().__init__()
        self.fs = 125
        self.current_data = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Info panel
        info_panel = QHBoxLayout()
        self.label_status = QLabel("No data loaded")
        info_panel.addWidget(self.label_status)
        info_panel.addStretch()
        layout.addLayout(info_panel)
        
        # Create figure
        self.figure = Figure(figsize=(16, 10))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
    
    def load_data(self, data_dict):
        """Load and validate data"""
        self.current_data = data_dict
        self.validate_all_data()
    
    def validate_all_data(self):
        """Perform comprehensive data validation"""
        if self.current_data is None:
            return
        
        self.figure.clear()
        
        windows = self.current_data['windows']
        n_windows = len(windows)
        n_channels = windows[0].shape[0] if windows[0].ndim >= 2 else windows[0].shape[0]
        
        # Initialize validation metrics
        spike_percentages = np.zeros((n_windows, n_channels))
        snr_values = np.zeros((n_windows, n_channels))
        variance_values = np.zeros((n_windows, n_channels))
        bad_channels = []
        bad_windows = []
        
        # Validate each window
        for w_idx, window in enumerate(windows):
            if window.ndim == 3:
                window = window[0]
            
            for ch_idx in range(n_channels):
                signal = window[ch_idx]
                
                # 1. Check for spikes (z-score > 5)
                z_scores = np.abs((signal - np.mean(signal)) / np.std(signal))
                spike_count = np.sum(z_scores > 5)
                spike_percentages[w_idx, ch_idx] = (spike_count / len(signal)) * 100
                
                # 2. Calculate SNR
                freqs, psd = welch(signal, self.fs)
                signal_band = (freqs >= 8) & (freqs <= 30)  # Alpha/beta band
                noise_band = (freqs >= 45) & (freqs <= 55)  # Noise band
                
                signal_power = np.mean(psd[signal_band]) if np.any(signal_band) else 1
                noise_power = np.mean(psd[noise_band]) if np.any(noise_band) else 1
                snr_values[w_idx, ch_idx] = 10 * np.log10(signal_power / noise_power)
                
                # 3. Calculate variance
                variance_values[w_idx, ch_idx] = np.var(signal)
                
                # Mark bad channels (>1% spikes)
                if spike_percentages[w_idx, ch_idx] > 1.0:
                    if ch_idx not in bad_channels:
                        bad_channels.append(ch_idx)
            
            # Mark bad windows (too many bad channels)
            bad_ch_count = np.sum(spike_percentages[w_idx] > 1.0)
            if bad_ch_count > n_channels * 0.3:  # More than 30% channels bad
                bad_windows.append(w_idx)
        
        # Create visualizations
        # 1. Spike percentage heatmap
        ax1 = self.figure.add_subplot(2, 3, 1)
        im1 = ax1.imshow(spike_percentages.T, aspect='auto', cmap='hot', vmin=0, vmax=2)
        ax1.set_title('Spike Percentage (>5 SD)')
        ax1.set_xlabel('Window')
        ax1.set_ylabel('Channel')
        ax1.set_yticks(range(n_channels))
        ax1.set_yticklabels([f'Ch{i+1}' for i in range(n_channels)])
        plt.colorbar(im1, ax=ax1, label='%')
        
        # Mark bad windows
        for bad_w in bad_windows:
            ax1.axvline(bad_w, color='red', alpha=0.3, linewidth=2)
        
        # 2. SNR heatmap
        ax2 = self.figure.add_subplot(2, 3, 2)
        im2 = ax2.imshow(snr_values.T, aspect='auto', cmap='coolwarm', vmin=-10, vmax=20)
        ax2.set_title('Signal-to-Noise Ratio (dB)')
        ax2.set_xlabel('Window')
        ax2.set_ylabel('Channel')
        ax2.set_yticks(range(n_channels))
        ax2.set_yticklabels([f'Ch{i+1}' for i in range(n_channels)])
        plt.colorbar(im2, ax=ax2, label='dB')
        
        # 3. Variance distribution
        ax3 = self.figure.add_subplot(2, 3, 3)
        for ch in range(n_channels):
            if ch in bad_channels:
                ax3.plot(variance_values[:, ch], 'r-', alpha=0.5, linewidth=1)
            else:
                ax3.plot(variance_values[:, ch], 'b-', alpha=0.3, linewidth=1)
        ax3.set_title('Signal Variance Over Time')
        ax3.set_xlabel('Window')
        ax3.set_ylabel('Variance')
        ax3.grid(True, alpha=0.3)
        
        # 4. Channel quality summary
        ax4 = self.figure.add_subplot(2, 3, 4)
        channel_quality = np.mean(spike_percentages < 1.0, axis=0) * 100
        bars = ax4.bar(range(n_channels), channel_quality)
        
        for i, bar in enumerate(bars):
            if i in bad_channels:
                bar.set_color('red')
            else:
                bar.set_color('green')
        
        ax4.set_title('Channel Quality Score')
        ax4.set_xlabel('Channel')
        ax4.set_ylabel('% Good Windows')
        ax4.set_ylim([0, 105])
        ax4.set_xticks(range(n_channels))
        ax4.set_xticklabels([f'Ch{i+1}' for i in range(n_channels)])
        ax4.axhline(90, color='orange', linestyle='--', alpha=0.5)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Window quality timeline
        ax5 = self.figure.add_subplot(2, 3, 5)
        window_quality = np.mean(spike_percentages < 1.0, axis=1) * 100
        ax5.plot(window_quality, 'b-', linewidth=2)
        ax5.fill_between(range(n_windows), window_quality, alpha=0.3)
        ax5.axhline(70, color='red', linestyle='--', alpha=0.5, label='Poor')
        ax5.axhline(90, color='orange', linestyle='--', alpha=0.5, label='Good')
        ax5.set_title('Window Quality Over Time')
        ax5.set_xlabel('Window')
        ax5.set_ylabel('% Good Channels')
        ax5.set_ylim([0, 105])
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary statistics
        ax6 = self.figure.add_subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""DATA QUALITY SUMMARY
        
Total Windows: {n_windows}
Total Channels: {n_channels}

Bad Channels: {bad_channels if bad_channels else 'None'}
Bad Windows: {len(bad_windows)} ({len(bad_windows)/n_windows*100:.1f}%)

Overall Quality Score: {np.mean(window_quality):.1f}%
Avg SNR: {np.mean(snr_values):.1f} dB
Avg Spike Rate: {np.mean(spike_percentages):.3f}%

Status: {'✓ GOOD' if np.mean(window_quality) > 85 else '⚠ NEEDS IMPROVEMENT'}
"""
        
        ax6.text(0.1, 0.9, summary_text, fontsize=11, 
                verticalalignment='top', family='monospace')
        
        # Update status
        quality_score = np.mean(window_quality)
        if quality_score > 85:
            status = "Data Quality: GOOD ✓"
            self.label_status.setStyleSheet("color: green; font-weight: bold;")
        elif quality_score > 70:
            status = "Data Quality: ACCEPTABLE"
            self.label_status.setStyleSheet("color: orange; font-weight: bold;")
        else:
            status = "Data Quality: POOR - Check electrodes!"
            self.label_status.setStyleSheet("color: red; font-weight: bold;")
        
        self.label_status.setText(status)
        
        self.figure.tight_layout()
        self.canvas.draw()

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.current_data = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('EEG Signal Processing & Validation Suite')
        self.setGeometry(100, 100, 1600, 900)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create toolbar
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # Add actions to toolbar
        load_action = QAction('📁 Load PKL File', self)
        load_action.triggered.connect(self.load_pkl_file)
        toolbar.addAction(load_action)
        
        toolbar.addSeparator()
        
        train_csp_action = QAction('🧠 Train CSP', self)
        train_csp_action.triggered.connect(self.train_csp)
        toolbar.addAction(train_csp_action)
        
        toolbar.addSeparator()
        
        # Add label for file info
        self.label_file = QLabel("No file loaded")
        toolbar.addWidget(self.label_file)
        
        # Create tab widget for pages
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Create pages
        self.filtering_page = FilteringPage()
        self.validation_page = DataValidationPage()
        
        # Add pages to tabs
        self.tabs.addTab(self.filtering_page, "🔧 Filtering Pipeline")
        self.tabs.addTab(self.validation_page, "✓ Data Validation")
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def load_pkl_file(self):
        """Load PKL file with file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select PKL File", "", "Pickle Files (*.pkl);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'rb') as f:
                    loaded_data = pickle.load(f)
                
                # Parse the loaded data
                windows = []
                labels = []
                
                if isinstance(loaded_data, list):
                    for item in loaded_data:
                        if isinstance(item, dict):
                            windows.append(np.array(item.get('data', item.get('window', []))))
                            labels.append(item.get('label', 0))
                        elif isinstance(item, tuple):
                            windows.append(np.array(item[0]))
                            labels.append(item[1] if len(item) > 1 else 0)
                        else:
                            windows.append(np.array(item))
                            labels.append(0)
                elif isinstance(loaded_data, dict):
                    if 'windows' in loaded_data:
                        windows = loaded_data['windows']
                        labels = loaded_data.get('labels', [0] * len(windows))
                    elif 'data' in loaded_data:
                        windows = [loaded_data['data']]
                        labels = [loaded_data.get('label', 0)]
                else:
                    windows = [np.array(loaded_data)]
                    labels = [0]
                
                # Store processed data
                self.current_data = {
                    'windows': windows,
                    'labels': labels if labels else None,
                    'file_path': file_path
                }
                
                # Update pages
                self.filtering_page.load_data(self.current_data)
                self.validation_page.load_data(self.current_data)
                
                # Update UI
                filename = os.path.basename(file_path)
                self.label_file.setText(f"File: {filename} | Windows: {len(windows)}")
                self.status_bar.showMessage(f"Loaded {len(windows)} windows from {filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")
                self.status_bar.showMessage("Failed to load file")
    
    def train_csp(self):
        """Train CSP filter on loaded data"""
        if self.current_data is None:
            QMessageBox.warning(self, "Warning", "Please load data first!")
            return
        
        if self.current_data['labels'] is None:
            QMessageBox.warning(self, "Warning", "No labels found in data!")
            return
        
        try:
            # Split data by label
            windows = np.array(self.current_data['windows'])
            labels = np.array(self.current_data['labels'])
            
            X_left = windows[labels == 0]
            X_right = windows[labels == 1]
            
            if len(X_left) < 2 or len(X_right) < 2:
                QMessageBox.warning(self, "Warning", 
                                   f"Not enough data for CSP training!\n"
                                   f"Left: {len(X_left)}, Right: {len(X_right)}")
                return
            
            # Train CSP
            self.filtering_page.csp_filter.fit(X_left, X_right)
            self.filtering_page.csp_trained = True
            self.filtering_page.update_display()
            
            QMessageBox.information(self, "Success", 
                                   f"CSP trained successfully!\n"
                                   f"Left trials: {len(X_left)}\n"
                                   f"Right trials: {len(X_right)}")
            
            self.status_bar.showMessage("CSP filter trained successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to train CSP:\n{str(e)}")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()