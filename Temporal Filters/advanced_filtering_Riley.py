import numpy as np
from scipy.signal import butter, filtfilt, medfilt
#Riley :p
# Advanced_filtering tests out using a more advanced form of notch filter called spectrum interpolation
# In case spectrum interpolation is too computationally expensive for your model, switch back to simple notch filter
def _as_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim > 1:
        raise ValueError("Expected 1D array (single channel).")
    return x.astype(float, copy=False)

def spectrum_interpolation_notch(signal, fs, notch_freq=60.0, notch_width=2):
    """Remove 60Hz using FFT interpolation - ENHANCED"""
    # Algorithm from PDF Section 2
    fft_signal = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1/fs)
    noise_bin = np.argmin(np.abs(freqs - notch_freq))
    
    # More aggressive notch - interpolate wider range
    if noise_bin - notch_width >= 0 and noise_bin + notch_width < len(fft_signal):
        # Interpolate using points further away for smoother transition
        fft_signal[noise_bin] = (fft_signal[noise_bin - notch_width] + 
                                 fft_signal[noise_bin + notch_width]) / 2
        
        # Also attenuate adjacent bins
        for offset in range(-1, 2):
            if offset != 0:
                idx = noise_bin + offset
                if 0 <= idx < len(fft_signal):
                    fft_signal[idx] *= 0.5
    
    return np.fft.irfft(fft_signal, len(signal))

def zero_phase_bandpass(signal, lowcut, highcut, fs, order=6):
    """Apply filtfilt for zero phase lag - ENHANCED with higher order"""
    # Algorithm from PDF Section 3 - increased default order
    b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)
    return filtfilt(b, a, signal)

def remove_spikes(signal, threshold=3.0, kernel_size=5):
    """
    Remove sharp spikes using median filter and z-score thresholding.
    
    Args:
        signal: Input signal
        threshold: Z-score threshold for spike detection (default 3.0)
        kernel_size: Kernel size for median filter (default 5)
    
    Returns:
        signal with spikes removed
    """
    # Compute z-scores
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)
    
    if signal_std < 1e-10:
        return signal
    
    z_scores = np.abs((signal - signal_mean) / signal_std)
    
    # Apply median filter to entire signal
    filtered = medfilt(signal, kernel_size=kernel_size)
    
    # Replace spikes (high z-scores) with median-filtered values
    spike_mask = z_scores > threshold
    result = signal.copy()
    result[spike_mask] = filtered[spike_mask]
    
    return result

def kalman_filter_denoise(
    signal: np.ndarray,
    fs: float,
    process_var: float = 1e-4,
    meas_var: float = 1e-2,
    model: str = "position_velocity",
    dt: float | None = None,
) -> np.ndarray:
    """
    Adaptive Kalman filter denoiser for 1D signals.

    Models the signal as a latent state evolving smoothly in time with
    Gaussian process and measurement noise. Works online or offline, and
    adapts its confidence between model prediction and measurement.

    Parameters
    ----------
    signal : (n,) array
        Noisy input signal (single channel).
    fs : float
        Sampling rate (Hz).
    process_var : float
        Process (system) noise variance Q; higher values make the filter
        track faster but pass more noise.
    meas_var : float
        Measurement noise variance R; higher values make the filter smoother
        but more sluggish.
    model : {'random_walk', 'position_velocity'}
        - 'random_walk' : state = [x]
        - 'position_velocity' : state = [x, dx/dt]
    dt : float, optional
        Sampling interval (defaults to 1/fs).

    Returns
    -------
    filtered : (n,) array
        Kalman-smoothed signal estimate.
    """
    x = _as_1d(signal)
    n = x.size
    if dt is None:
        dt = 1.0 / fs

    if model == "random_walk":
        # State: [x]
        A = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[process_var]])
        R = np.array([[meas_var]])
        x_est = np.array([[x[0]]])
        P = np.array([[1.0]])
    elif model == "position_velocity":
        # State: [x, v]
        A = np.array([[1.0, dt],
                      [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = np.array([[process_var, 0.0],
                      [0.0, process_var]])
        R = np.array([[meas_var]])
        x_est = np.array([[x[0]], [0.0]])
        P = np.eye(2)
    else:
        raise ValueError("model must be 'random_walk' or 'position_velocity'")

    filtered = np.zeros_like(x)
    for k, z in enumerate(x):
        # Predict
        x_pred = A @ x_est
        P_pred = A @ P @ A.T + Q

        # Update
        y = np.array([[z]]) - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_est = x_pred + K @ y
        P = (np.eye(A.shape[0]) - K @ H) @ P_pred

        filtered[k] = x_est[0, 0]

    return filtered

def complete_filtering_pipeline(signal, fs, lowcut=5.0, highcut=35.0, 
                                notch_freq=60.0, order=6, 
                                remove_spikes_flag=True):
    """
    Combine all filters with spike removal - ENHANCED
    
    Args:
        signal: Input EEG signal
        fs: Sampling rate
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        notch_freq: Notch filter frequency (Hz)
        order: Butterworth filter order
        remove_spikes_flag: Whether to apply spike removal
    
    Returns:
        Fully filtered signal
    """
    # Step 1: Bandpass filter (zero-phase)
    filtered = zero_phase_bandpass(signal, lowcut, highcut, fs, order)
    
    # Step 2: Notch filter
    filtered = spectrum_interpolation_notch(filtered, fs, notch_freq)
    
    # Step 3: Spike removal (optional but recommended)
    if remove_spikes_flag:
        filtered = remove_spikes(filtered, threshold=3.0, kernel_size=5)
    
    return filtered