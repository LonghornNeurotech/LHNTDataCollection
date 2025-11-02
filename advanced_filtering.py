# Kailash Boggaram
# Signals RTP
# Spectrum Interpolation Notch + Zero-Phase Bandpass
# 10/05/2025 (rev)

from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt

def _as_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim > 1:
        raise ValueError("Expected 1D array (single channel).")
    return x.astype(float, copy=False)

def spectrum_interpolation_notch(
    signal: np.ndarray,
    fs: float,
    notch_freq: float = 60.0,
    width_hz: float = 2.0,
    harmonics: tuple[float, ...] | None = None,
    edge_guard: int = 1,
) -> np.ndarray:
    """
    Remove narrowband line noise via complex-spectrum interpolation (rFFT).
    - No ringing or phase distortion.
    - Works on each specified frequency (60 Hz by default) and any harmonics.

    Parameters
    ----------
    signal : (n,) array
        Single-channel time series.
    fs : float
        Sampling rate (Hz).
    notch_freq : float
        Base frequency to suppress (e.g., 50 or 60 Hz).
    width_hz : float
        Total suppression band width around each target freq (±width_hz/2).
    harmonics : tuple of float, optional
        Additional frequencies to remove (e.g., (120, 180)).
        If None, no extra harmonics beyond notch_freq.
    edge_guard : int
        Keep this many bins untouched outside each notched region to stabilize interpolation.

    Returns
    -------
    filtered : (n,) array
    """
    x = _as_1d(signal)
    n = x.size
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)

    # Build list of target frequencies
    targets = [float(notch_freq)] if notch_freq is not None else []
    if harmonics:
        targets.extend(map(float, harmonics))

    if not targets:
        return np.fft.irfft(X, n=n)

    # Create mask of bins to replace
    mask = np.zeros_like(freqs, dtype=bool)
    half = width_hz / 2.0
    for f0 in targets:
        mask |= (np.abs(freqs - f0) <= half)

    # Identify contiguous masked blocks
    # For each block, do complex linear interpolation across the gap using the two nearest unmasked bins.
    def contiguous_regions(m: np.ndarray):
        idx = np.flatnonzero(m)
        if idx.size == 0:
            return []
        breaks = np.where(np.diff(idx) > 1)[0]
        starts = np.r_[idx[0], idx[breaks + 1]]
        ends = np.r_[idx[breaks], idx[-1]]
        return list(zip(starts, ends))

    blocks = contiguous_regions(mask)

    X_new = X.copy()
    for start, end in blocks:
        # Expand with edge_guard so we don't interpolate *right* at the edge of the untouched bins
        left = max(0, start - edge_guard)
        right = min(len(freqs) - 1, end + edge_guard)

        # Find left/right anchor bins that are NOT masked
        # Left anchor: the nearest unmasked bin < left
        la = left - 1
        while la >= 0 and mask[la]:
            la -= 1
        # Right anchor: the nearest unmasked bin > right
        ra = right + 1
        L = len(freqs)
        while ra < L and mask[ra]:
            ra += 1

        if la < 0 and ra >= L:
            # Everything masked (degenerate). Zero them out.
            X_new[start:end+1] = 0.0
            continue
        elif la < 0:
            # Only right anchor exists -> flat fill with right anchor's value
            X_new[start:end+1] = X_new[ra]
            continue
        elif ra >= L:
            # Only left anchor exists -> flat fill with left anchor's value
            X_new[start:end+1] = X_new[la]
            continue

        # Complex-linear interpolation from la -> ra over indices [left..right]
        f_la, f_ra = freqs[la], freqs[ra]
        X_la, X_ra = X_new[la], X_new[ra]

        # Interpolate only across the masked range [start..end]
        k = np.arange(start, end + 1)
        t = (freqs[k] - f_la) / (f_ra - f_la + 1e-15)  # avoid 0-div
        X_interp = (1.0 - t) * X_la + t * X_ra
        X_new[k] = X_interp

    return np.fft.irfft(X_new, n=n)

def zero_phase_bandpass(
    signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
    dc_block: bool = False,
) -> np.ndarray:
    """
    Zero-phase bandpass using SOS + sosfiltfilt.
    Stable for long signals and tight bands.

    Parameters
    ----------
    signal : (n,) array
    lowcut, highcut : float
        Band edges in Hz (0 < lowcut < highcut < fs/2).
    fs : float
        Sampling rate.
    order : int
        Analog prototype order (Butterworth). Total bandpass order is 2*order.
    dc_block : bool
        If True and lowcut == 0, raise; DC cannot be kept with zero-phase bandpass.
        If True and lowcut < 0.1 Hz, we gently push lowcut up to 0.1 Hz for numeric stability.

    Returns
    -------
    filtered : (n,) array
    """
    x = _as_1d(signal)

    if dc_block and lowcut <= 0.0:
        raise ValueError("lowcut must be > 0 when dc_block=True.")
    if lowcut < 0.1 and dc_block:
        lowcut = 0.1

    if not (0 < lowcut < highcut < fs * 0.5):
        raise ValueError("Need 0 < lowcut < highcut < Nyquist.")

    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)
