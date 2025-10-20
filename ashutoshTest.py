"""
Data validation utilities for EEG streams.

Implements:
- detect_nan_inf
- detect_flatline
- detect_extreme_noise
- detect_channel_duplication
- ValidationPackage

Functions use numpy and expect data shaped (channels, samples) or (samples,) for single channel.
"""

from typing import List, Tuple
import numpy as np


def detect_nan_inf(data: np.ndarray) -> Tuple[bool, List[int]]:
    """Detect NaN or Inf values in EEG data.

    Returns (is_valid, bad_channel_indices).
    """
    data = np.asarray(data)

    if data.size == 0:
        return True, []

    if data.ndim == 1:
        bad_mask = np.isnan(data) | np.isinf(data)
        if np.any(bad_mask):
            return False, [0]
        return True, []

    nan_mask = np.isnan(data)
    inf_mask = np.isinf(data)
    combined = nan_mask | inf_mask

    bad_per_channel = np.any(combined, axis=1)
    bad_indices = np.nonzero(bad_per_channel)[0].tolist()

    is_valid = len(bad_indices) == 0
    return is_valid, bad_indices


def detect_flatline(data: np.ndarray, threshold: float = 0.1) -> Tuple[List[int], np.ndarray]:
    """Detect flatline channels where per-channel std < threshold.

    Returns (flatline_channel_indices, channel_stds).
    """
    data = np.asarray(data)

    if data.size == 0:
        return [], np.array([])

    if data.ndim == 1:
        stds = np.array([float(np.std(data))])
        flat = np.nonzero(stds < threshold)[0].tolist()
        return flat, stds

    stds = np.std(data, axis=1)
    flat = np.nonzero(stds < threshold)[0].tolist()
    return flat, stds


def detect_extreme_noise(data: np.ndarray, z_threshold: float = 5.0) -> Tuple[bool, List[int], float]:
    """Detect spike artifacts per-channel using robust MAD-based z-score.

    Uses median and MAD (scaled to approximate std) so large spikes do not
    inflate the baseline. Marks channel bad if >1% samples exceed threshold.
    Returns (has_spikes, spike_channel_indices, overall_spike_percentage).
    """

    data = np.asarray(data)

    if data.size == 0:
        return False, [], 0.0

    if data.ndim == 1:
        data = data.reshape(1, -1)

    med = np.median(data, axis=1)
    mad = np.median(np.abs(data - med[:, None]), axis=1)
    mad_std = mad * 1.4826
    mad_std[mad_std == 0] = np.inf

    z = np.abs((data - med[:, None]) / mad_std[:, None])
    spikes_mask = z > z_threshold

    spike_counts = spikes_mask.sum(axis=1)
    samples = data.shape[1]
    spike_pct_per_channel = 100.0 * spike_counts / samples

    spike_channels = np.nonzero(spike_pct_per_channel > 1.0)[0].tolist()
    total_spikes = spikes_mask.sum()
    overall_spike_pct = 100.0 * total_spikes / data.size
    has_spikes = len(spike_channels) > 0

    return has_spikes, spike_channels, overall_spike_pct


def detect_channel_duplication(data: np.ndarray, correlation_threshold: float = 0.99) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Detect pairs of channels with Pearson correlation > threshold.

    Returns (duplicate_pairs, correlation_matrix).
    """
    data = np.asarray(data)

    if data.size == 0:
        return [], np.array([[]])

    if data.ndim == 1:
        return [], np.array([[1.0]])

    corr = np.corrcoef(data)
    corr = np.nan_to_num(corr)

    c = corr.shape[0]
    dup_pairs: List[Tuple[int, int]] = []
    for i in range(c):
        for j in range(i + 1, c):
            if corr[i, j] > correlation_threshold:
                dup_pairs.append((i, j))

    return dup_pairs, corr


class ValidationPackage:

    def __init__(self, data: np.ndarray, timestamp: float, sampling_rate: float | None = None):
        self.data = np.asarray(data)
        self.timestamp = timestamp
        self.sampling_rate = sampling_rate

        is_valid_nan, nan_ch = detect_nan_inf(self.data)
        flat_ch, stds = detect_flatline(self.data)
        has_spikes, spike_ch, spike_pct = detect_extreme_noise(self.data)
        dup_pairs, corr = detect_channel_duplication(self.data)

        self.quality_metrics = {
            'has_nan': not is_valid_nan,
            'nan_channels': nan_ch,
            'has_inf': not is_valid_nan,
            'flatline_channels': flat_ch,
            'noisy_channels': spike_ch,
            'spike_percentage': spike_pct,
            'duplicate_pairs': dup_pairs,
            'correlation_matrix': corr,
            'channel_stds': stds,
            'sampling_rate': sampling_rate,
        }

        self.is_valid = not (
            self.quality_metrics['has_nan'] or
            len(self.quality_metrics['flatline_channels']) > 0 or
            len(self.quality_metrics['noisy_channels']) > 0 or
            len(self.quality_metrics['duplicate_pairs']) > 0
        )
