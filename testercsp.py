"""
Enhanced Testing Framework with Epoch-Based Analysis
Longhorn Neurotech - Real-Time Processing Team

NEW FEATURES:
1. CSP plots more visible (reduced height, better layout)
2. Epoch-based analysis (1-second windows)
3. Statistical trends over multiple epochs
4. Long-term quality tracking
"""

import sys
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import traceback
from collections import deque

# =============================================================================
# IMPORT YOUR IMPLEMENTATIONS HERE
# =============================================================================
try:
    from advanced_filtering import (
        spectrum_interpolation_notch, 
        zero_phase_bandpass,
        complete_filtering_pipeline
    )
    FILTERING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import filtering functions: {e}")
    FILTERING_AVAILABLE = False

try:
    from csp_module import CSPFilter
    CSP_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import CSP module: {e}")
    CSP_AVAILABLE = False


class EpochAnalyzer:
    """Analyze signal quality over longer epochs (1+ seconds)"""
    
    def __init__(self, epoch_length=1.0, max_history=60):
        """
        Args:
            epoch_length: Length of each epoch in seconds
            max_history: Maximum number of epochs to store
        """
        self.epoch_length = epoch_length
        self.max_history = max_history
        
        # History storage
        self.snr_history = deque(maxlen=max_history)
        self.notch_eff_history = deque(maxlen=max_history)
        self.quality_history = deque(maxlen=max_history)
        self.timestamp_history = deque(maxlen=max_history)
        
        self.start_time = time.time()
    
    def add_epoch(self, snr, notch_eff, quality_rating):
        """Add analysis results from one epoch"""
        current_time = time.time() - self.start_time
        
        self.snr_history.append(snr)
        self.notch_eff_history.append(notch_eff)
        self.quality_history.append(quality_rating)
        self.timestamp_history.append(current_time)
    
    def get_statistics(self):
        """Get statistical summary of all epochs"""
        if len(self.snr_history) == 0:
            return None
        
        snr_array = np.array(self.snr_history)
        notch_array = np.array(self.notch_eff_history)
        
        return {
            'n_epochs': len(self.snr_history),
            'snr_mean': np.mean(snr_array),
            'snr_std': np.std(snr_array),
            'snr_min': np.min(snr_array),
            'snr_max': np.max(snr_array),
            'notch_mean': np.mean(notch_array),
            'notch_std': np.std(notch_array),
            'quality_counts': self._count_quality_ratings(),
            'duration': self.timestamp_history[-1] if len(self.timestamp_history) > 0 else 0
        }
    
    def _count_quality_ratings(self):
        """Count occurrences of each quality rating"""
        counts = {'EXCELLENT': 0, 'GOOD': 0, 'FAIR': 0, 'POOR': 0}
        for rating in self.quality_history:
            if rating in counts:
                counts[rating] += 1
        return counts
    
    def get_trend_data(self):
        """Get data for plotting trends"""
        return {
            'timestamps': list(self.timestamp_history),
            'snr': list(self.snr_history),
            'notch_eff': list(self.notch_eff_history)
        }


class SignalQualityMetrics:
    """Compute signal quality metrics with professional benchmarks"""
    
    @staticmethod
    def compute_snr_db(signal, noise):
        """Compute SNR in decibels"""
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power < 1e-10:
            return 100.0
        
        snr = signal_power / noise_power
        snr_db = 10 * np.log10(snr)
        
        return snr_db
    
    @staticmethod
    def compute_frequency_snr(signal, fs, signal_band=(8, 30), noise_band=(55, 65)):
        """Compute SNR in frequency domain"""
        from scipy.fft import rfft, rfftfreq
        
        fft_signal = np.abs(rfft(signal))
        freqs = rfftfreq(len(signal), 1/fs)
        
        signal_mask = (freqs >= signal_band[0]) & (freqs <= signal_band[1])
        signal_power = np.mean(fft_signal[signal_mask] ** 2)
        
        noise_mask = (freqs >= noise_band[0]) & (freqs <= noise_band[1])
        noise_power = np.mean(fft_signal[noise_mask] ** 2)
        
        if noise_power < 1e-10:
            return 100.0
        
        snr = signal_power / noise_power
        snr_db = 10 * np.log10(snr)
        
        return snr_db
    
    @staticmethod
    def get_quality_rating(snr_db):
        """Convert SNR to quality rating"""
        if snr_db >= 20:
            return ("EXCELLENT", "#00FF00", "Publication quality")
        elif snr_db >= 15:
            return ("GOOD", "#90EE90", "Research quality")
        elif snr_db >= 10:
            return ("FAIR", "#FFD700", "Acceptable")
        else:
            return ("POOR", "#FF4444", "Needs improvement")
    
    @staticmethod
    def compute_notch_effectiveness(raw_signal, filtered_signal, fs, notch_freq=60.0):
        """Measure notch filter effectiveness"""
        from scipy.fft import rfft, rfftfreq
        
        fft_raw = np.abs(rfft(raw_signal))
        fft_filtered = np.abs(rfft(filtered_signal))
        freqs = rfftfreq(len(raw_signal), 1/fs)
        
        idx_60hz = np.argmin(np.abs(freqs - notch_freq))
        
        power_raw = fft_raw[idx_60hz] ** 2
        power_filtered = fft_filtered[idx_60hz] ** 2
        
        if power_filtered < 1e-10:
            return 100.0, 100.0
        
        reduction_db = 10 * np.log10(power_raw / power_filtered)
        effectiveness = (1 - power_filtered / power_raw) * 100
        
        return reduction_db, effectiveness


class EnhancedFilteringTester:
    def __init__(self):
        # Set up BrainFlow synthetic board
        BoardShim.enable_dev_board_logger()
        params = BrainFlowInputParams()
        self.board_id = BoardIds.SYNTHETIC_BOARD.value
        self.board = BoardShim(self.board_id, params)
        
        # Get board info
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        
        # Filter parameters
        self.lowcut = 5.0
        self.highcut = 35.0
        self.notch_freq = 60.0
        self.filter_order = 6
        
        # Data buffer settings
        self.buffer_size = int(5 * self.sampling_rate)
        self.display_size = int(2 * self.sampling_rate)
        
        # Epoch analysis settings
        self.epoch_length = 1.0  # 1 second epochs
        self.epoch_samples = int(self.epoch_length * self.sampling_rate)
        self.epoch_analyzer = EpochAnalyzer(epoch_length=self.epoch_length, max_history=60)
        self.samples_since_epoch = 0
        
        # Initialize data buffers
        self.n_display_channels = min(8, len(self.eeg_channels))
        self.raw_buffer = np.zeros((self.n_display_channels, self.buffer_size))
        self.filtered_buffer = np.zeros((self.n_display_channels, self.buffer_size))
        
        # CSP demonstration
        self.setup_csp_demo()
        
        # Setup GUI
        self.setup_gui()
        
        # Start board
        self.board.prepare_session()
        self.board.start_stream()
        print("BrainFlow streaming started...")
    
    def setup_csp_demo(self):
        """Generate synthetic CSP demonstration data"""
        if not CSP_AVAILABLE:
            return
        
        try:
            np.random.seed(42)
            n_channels = 8
            n_samples = 500
            n_trials = 20
            
            # Generate left hand imagery
            X_left = []
            for _ in range(n_trials):
                t = np.linspace(0, 2, n_samples)
                signal = np.zeros((n_channels, n_samples))
                
                for ch in range(n_channels):
                    if ch in [4, 5, 6]:  # Right hemisphere
                        signal[ch] = 3.0 * np.sin(2 * np.pi * 10 * t)
                    else:
                        signal[ch] = 1.0 * np.sin(2 * np.pi * 10 * t)
                    signal[ch] += 0.5 * np.random.randn(n_samples)
                
                X_left.append(signal)
            
            # Generate right hand imagery
            X_right = []
            for _ in range(n_trials):
                t = np.linspace(0, 2, n_samples)
                signal = np.zeros((n_channels, n_samples))
                
                for ch in range(n_channels):
                    if ch in [1, 2, 3]:  # Left hemisphere
                        signal[ch] = 3.0 * np.sin(2 * np.pi * 10 * t)
                    else:
                        signal[ch] = 1.0 * np.sin(2 * np.pi * 10 * t)
                    signal[ch] += 0.5 * np.random.randn(n_samples)
                
                X_right.append(signal)
            
            # Train CSP
            self.csp_filter = CSPFilter(n_components=2)
            self.csp_filter.fit(X_left, X_right)
            
            features_left = self.csp_filter.transform(X_left)
            features_right = self.csp_filter.transform(X_right)
            
            self.csp_demo_data = {
                'features_left': features_left,
                'features_right': features_right,
                'filters': self.csp_filter.filters,
                'n_channels': n_channels
            }
            
            print("✓ CSP demo data generated")
            
        except Exception as e:
            print(f"Error setting up CSP demo: {e}")
            self.csp_demo_data = None
    
    def setup_gui(self):
        """Set up the visualization window with better layout"""
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication(sys.argv)
        
        # Main window
        self.main_widget = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        
        # Graphics layout for plots
        self.win = pg.GraphicsLayoutWidget()
        self.main_layout.addWidget(self.win)
        
        # Metrics panel at bottom
        self.metrics_panel = QtWidgets.QTextEdit()
        self.metrics_panel.setReadOnly(True)
        self.metrics_panel.setMaximumHeight(180)
        self.metrics_panel.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: white;
                font-family: 'Courier New', monospace;
                font-size: 9pt;
                border: 2px solid #333;
            }
        """)
        self.main_layout.addWidget(self.metrics_panel)
        
        # Show window
        self.main_widget.resize(1600, 1100)
        self.main_widget.setWindowTitle("Enhanced EEG Testing - Epoch Analysis")
        self.main_widget.show()
        
        self.plots = {}
        self.curves = {}
        
        # Colors
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
                      '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']
        
        # ROW 1: Time-domain signals (reduced height)
        self.plots['raw'] = self.win.addPlot(title="Raw EEG Signal", row=0, col=0)
        self.plots['raw'].setLabel('left', 'Amplitude (μV)')
        self.plots['raw'].setLabel('bottom', 'Samples')
        self.plots['raw'].setMaximumHeight(200)
        
        self.plots['filtered'] = self.win.addPlot(title="Filtered Signal", row=0, col=1)
        self.plots['filtered'].setLabel('left', 'Amplitude (μV)')
        self.plots['filtered'].setLabel('bottom', 'Samples')
        self.plots['filtered'].setMaximumHeight(200)
        
        # ROW 2: Frequency domain (reduced height)
        self.plots['fft_raw'] = self.win.addPlot(title="Frequency Spectrum - Raw", row=1, col=0)
        self.plots['fft_raw'].setLabel('left', 'Power')
        self.plots['fft_raw'].setLabel('bottom', 'Frequency (Hz)')
        self.plots['fft_raw'].setXRange(0, 80)
        self.plots['fft_raw'].setMaximumHeight(180)
        
        self.plots['fft_filtered'] = self.win.addPlot(title="Frequency Spectrum - Filtered", row=1, col=1)
        self.plots['fft_filtered'].setLabel('left', 'Power')
        self.plots['fft_filtered'].setLabel('bottom', 'Frequency (Hz)')
        self.plots['fft_filtered'].setXRange(0, 80)
        self.plots['fft_filtered'].setMaximumHeight(180)
        
        # ROW 3: Epoch trends (NEW)
        self.plots['snr_trend'] = self.win.addPlot(title="SNR Trend (1-sec epochs)", row=2, col=0)
        self.plots['snr_trend'].setLabel('left', 'SNR (dB)')
        self.plots['snr_trend'].setLabel('bottom', 'Time (s)')
        self.plots['snr_trend'].setMaximumHeight(150)
        self.plots['snr_trend'].addLine(y=20, pen=pg.mkPen('g', style=QtCore.Qt.DashLine))
        self.plots['snr_trend'].addLine(y=15, pen=pg.mkPen('y', style=QtCore.Qt.DashLine))
        self.plots['snr_trend'].addLine(y=10, pen=pg.mkPen('r', style=QtCore.Qt.DashLine))
        
        self.plots['notch_trend'] = self.win.addPlot(title="Notch Efficiency Trend", row=2, col=1)
        self.plots['notch_trend'].setLabel('left', 'Efficiency (%)')
        self.plots['notch_trend'].setLabel('bottom', 'Time (s)')
        self.plots['notch_trend'].setMaximumHeight(150)
        self.plots['notch_trend'].addLine(y=90, pen=pg.mkPen('g', style=QtCore.Qt.DashLine))
        
        # ROW 4: CSP visualization (reduced height for visibility)
        if CSP_AVAILABLE and self.csp_demo_data is not None:
            self.plots['csp_space'] = self.win.addPlot(title="CSP Feature Space", row=3, col=0)
            self.plots['csp_space'].setLabel('left', 'Component 2')
            self.plots['csp_space'].setLabel('bottom', 'Component 1')
            self.plots['csp_space'].setMaximumHeight(180)
            
            self.plots['csp_filters'] = self.win.addPlot(title="CSP Spatial Filters", row=3, col=1)
            self.plots['csp_filters'].setLabel('left', 'Weight')
            self.plots['csp_filters'].setLabel('bottom', 'Channel')
            self.plots['csp_filters'].setMaximumHeight(180)
            
            # Plot CSP data
            features_left = self.csp_demo_data['features_left']
            features_right = self.csp_demo_data['features_right']
            
            self.plots['csp_space'].plot(
                features_left[:, 0], features_left[:, 1],
                pen=None, symbol='o', symbolBrush='b', symbolSize=10,
                name='Left Hand'
            )
            self.plots['csp_space'].plot(
                features_right[:, 0], features_right[:, 1],
                pen=None, symbol='t', symbolBrush='r', symbolSize=10,
                name='Right Hand'
            )
            self.plots['csp_space'].addLegend()
            
            # Plot filters
            n_channels = self.csp_demo_data['n_channels']
            filters = self.csp_demo_data['filters']
            
            x = np.arange(n_channels)
            width = 0.35
            
            filter_1 = filters[:, 0]
            filter_last = filters[:, -1]
            
            bg1 = pg.BarGraphItem(x=x-width/2, height=filter_1, width=width, 
                                 brush='b', name='Filter 1')
            bg2 = pg.BarGraphItem(x=x+width/2, height=filter_last, width=width,
                                 brush='r', name='Filter 2')
            
            self.plots['csp_filters'].addItem(bg1)
            self.plots['csp_filters'].addItem(bg2)
            self.plots['csp_filters'].setXRange(-0.5, n_channels-0.5)
            self.plots['csp_filters'].addLegend()
        
        # Initialize signal curves
        for plot_name in ['raw', 'filtered']:
            self.curves[plot_name] = []
            for i in range(self.n_display_channels):
                curve = self.plots[plot_name].plot(
                    pen=pg.mkPen(color=self.colors[i], width=1.5)
                )
                self.curves[plot_name].append(curve)
        
        # Initialize FFT curves
        self.curves['fft_raw'] = self.plots['fft_raw'].plot(
            pen=pg.mkPen(color='#4ECDC4', width=2)
        )
        self.curves['fft_filtered'] = self.plots['fft_filtered'].plot(
            pen=pg.mkPen(color='#00FF00', width=2)
        )
        
        # Initialize trend curves
        self.curves['snr_trend'] = self.plots['snr_trend'].plot(
            pen=pg.mkPen(color='#00FFFF', width=2),
            symbol='o', symbolSize=5, symbolBrush='c'
        )
        self.curves['notch_trend'] = self.plots['notch_trend'].plot(
            pen=pg.mkPen(color='#FFD700', width=2),
            symbol='s', symbolSize=5, symbolBrush='y'
        )
        
        # Add markers
        self.plots['fft_raw'].addLine(x=60, pen=pg.mkPen('r', style=QtCore.Qt.DashLine, width=2))
        self.plots['fft_filtered'].addLine(x=60, pen=pg.mkPen('r', style=QtCore.Qt.DashLine, width=2))
        self.plots['fft_filtered'].addLine(x=5, pen=pg.mkPen('y', style=QtCore.Qt.DashLine))
        self.plots['fft_filtered'].addLine(x=35, pen=pg.mkPen('y', style=QtCore.Qt.DashLine))
        
        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(50)
    
    def apply_filtering(self, signal):
        """Apply enhanced filtering pipeline"""
        if not FILTERING_AVAILABLE:
            return signal
        
        try:
            from scipy.signal import butter, filtfilt, medfilt
            
            # Bandpass
            b, a = butter(self.filter_order, [self.lowcut, self.highcut], 
                         btype='band', fs=self.sampling_rate)
            filtered = filtfilt(b, a, signal)
            
            # Notch
            filtered = spectrum_interpolation_notch(filtered, self.sampling_rate, 
                                                   self.notch_freq)
            
            # Spike removal
            filtered = medfilt(filtered, kernel_size=3)
            
            return filtered
            
        except Exception as e:
            print(f"Filtering error: {e}")
            return signal
    
    def update_plots(self):
        """Update all plots with new data"""
        try:
            data = self.board.get_board_data()
            
            if data.shape[1] == 0:
                return
            
            eeg_data = data[self.eeg_channels[:self.n_display_channels], :]
            
            if eeg_data.shape[1] == 0:
                return
            
            # Update buffers
            new_samples = eeg_data.shape[1]
            self.raw_buffer = np.roll(self.raw_buffer, -new_samples, axis=1)
            self.raw_buffer[:, -new_samples:] = eeg_data
            
            # Apply filtering
            for ch_idx in range(self.n_display_channels):
                channel_data = self.raw_buffer[ch_idx, :].copy()
                filtered_data = self.apply_filtering(channel_data)
                self.filtered_buffer[ch_idx, :] = filtered_data
            
            # Update time-domain plots
            x_data = np.arange(self.display_size)
            
            for ch_idx in range(self.n_display_channels):
                self.curves['raw'][ch_idx].setData(
                    x_data,
                    self.raw_buffer[ch_idx, -self.display_size:]
                )
                self.curves['filtered'][ch_idx].setData(
                    x_data,
                    self.filtered_buffer[ch_idx, -self.display_size:]
                )
            
            # Update frequency plots
            self.update_frequency_plots()
            
            # Check if we should analyze an epoch
            self.samples_since_epoch += new_samples
            if self.samples_since_epoch >= self.epoch_samples:
                self.analyze_epoch()
                self.samples_since_epoch = 0
            
            # Update metrics display
            self.update_metrics_display()
            
        except Exception as e:
            print(f"Error in update_plots: {e}")
            traceback.print_exc()
    
    def update_frequency_plots(self):
        """Update FFT plots and compute SNR"""
        from scipy.fft import rfft, rfftfreq
        
        raw_signal = self.raw_buffer[0, -self.display_size:]
        filtered_signal = self.filtered_buffer[0, -self.display_size:]
        
        fft_raw = np.abs(rfft(raw_signal))
        fft_filtered = np.abs(rfft(filtered_signal))
        freqs = rfftfreq(len(raw_signal), 1/self.sampling_rate)
        
        self.curves['fft_raw'].setData(freqs, fft_raw)
        self.curves['fft_filtered'].setData(freqs, fft_filtered)
        
        # Compute current metrics
        snr_time = SignalQualityMetrics.compute_snr_db(
            filtered_signal,
            raw_signal - filtered_signal
        )
        
        snr_freq = SignalQualityMetrics.compute_frequency_snr(
            filtered_signal,
            self.sampling_rate,
            signal_band=(8, 30),
            noise_band=(55, 65)
        )
        
        notch_reduction, notch_eff = SignalQualityMetrics.compute_notch_effectiveness(
            raw_signal,
            filtered_signal,
            self.sampling_rate,
            self.notch_freq
        )
        
        rating, color, description = SignalQualityMetrics.get_quality_rating(snr_freq)
        
        self.current_metrics = {
            'snr_time': snr_time,
            'snr_freq': snr_freq,
            'notch_reduction': notch_reduction,
            'notch_eff': notch_eff,
            'rating': rating,
            'color': color,
            'description': description
        }
    
    def analyze_epoch(self):
        """Analyze one complete epoch and update trends"""
        if not hasattr(self, 'current_metrics'):
            return
        
        m = self.current_metrics
        
        # Add to epoch analyzer
        self.epoch_analyzer.add_epoch(
            m['snr_freq'],
            m['notch_eff'],
            m['rating']
        )
        
        # Update trend plots
        trend_data = self.epoch_analyzer.get_trend_data()
        
        if len(trend_data['timestamps']) > 0:
            self.curves['snr_trend'].setData(
                trend_data['timestamps'],
                trend_data['snr']
            )
            self.curves['notch_trend'].setData(
                trend_data['timestamps'],
                trend_data['notch_eff']
            )
    
    def update_metrics_display(self):
        """Update metrics panel with current and statistical data"""
        if not hasattr(self, 'current_metrics'):
            return
        
        m = self.current_metrics
        stats = self.epoch_analyzer.get_statistics()
        
        if stats is None:
            stats_text = "Collecting first epoch..."
        else:
            stats_text = f"""
EPOCH STATISTICS ({stats['n_epochs']} epochs, {stats['duration']:.1f}s total):
  SNR:   Mean={stats['snr_mean']:5.1f} dB  StdDev={stats['snr_std']:4.1f} dB  Range=[{stats['snr_min']:5.1f}, {stats['snr_max']:5.1f}] dB
  Notch: Mean={stats['notch_mean']:5.1f}%   StdDev={stats['notch_std']:4.1f}%
  Quality: EXCELLENT={stats['quality_counts']['EXCELLENT']:2d}  GOOD={stats['quality_counts']['GOOD']:2d}  FAIR={stats['quality_counts']['FAIR']:2d}  POOR={stats['quality_counts']['POOR']:2d}
"""
        
        metrics_text = f"""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║ CURRENT WINDOW (Real-time)                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════════════════╣
║ SNR: {m['snr_freq']:6.1f} dB  │  Notch: {m['notch_eff']:5.1f}%  │  Rating: {m['rating']:10s} - {m['description']}      ║
╠═══════════════════════════════════════════════════════════════════════════════════════════╣
║ {stats_text}║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
"""
        self.metrics_panel.setPlainText(metrics_text)
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'board'):
                self.board.stop_stream()
                self.board.release_session()
                print("BrainFlow streaming stopped.")
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def run(self):
        """Start the application"""
        try:
            print("\n" + "="*80)
            print("ENHANCED EEG TESTING - EPOCH ANALYSIS")
            print("="*80)
            print("\nFEATURES:")
            print("✓ Epoch-based analysis (1-second windows)")
            print("✓ Long-term quality tracking")
            print("✓ Statistical trends over time")
            print("✓ CSP visualization (scroll down to see)")
            print("\nPress Ctrl+C to stop...")
            print("="*80)
            
            self.app.exec_()
            
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.cleanup()


def main():
    """Main function"""
    print("Starting Enhanced EEG Testing with Epoch Analysis...")
    
    if not FILTERING_AVAILABLE:
        print("\n⚠️  WARNING: Filtering functions not available!")
    
    if not CSP_AVAILABLE:
        print("\n⚠️  INFO: CSP module not available.")
    
    try:
        tester = EnhancedFilteringTester()
        tester.run()
    except Exception as e:
        print(f"Failed to start tester: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()