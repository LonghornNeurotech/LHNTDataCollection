# Kailash Boggaram
#!/usr/bin/env python3
"""
LDA Spatial Filter (Fisher LDA) for multichannel signals
- fit_lda_filter(X, y, reg): learn a projection w that maximizes separation
- apply_lda_filter(X, w): apply w^T * X to get a 1D discriminant signal

Testing/Visualization:
- Synthesizes 2 classes of 8-channel signals with different narrowband activity
- Trains LDA spatial filter
- Plots time & frequency before vs after, and a 1D scatter showing separation

Dependencies: numpy, scipy, matplotlib
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# -------------------------------
# LDA spatial filter (2-class)
# -------------------------------

def fit_lda_filter(
    X: np.ndarray,
    y: np.ndarray,
    reg: float = 1e-3,
    center: bool = True,
) -> np.ndarray:
    """
    Fit a Fisher LDA spatial filter for TWO classes.
    Parameters
    ----------
    X : array, shape (n_samples, n_channels)  or (n_trials, n_channels)
        Each row is one observation vector across channels.
        For time-series training, you can concatenate samples from class A and B
        (with labels y marking class), or use trial-averaged features.
    y : array, shape (n_samples,)
        Binary class labels {0,1} (or {1,2}); other values will be remapped.
    reg : float
        Tikhonov regularization added to the within-class covariance: Sw + reg*I.
        Increase if ill-conditioned.
    center : bool
        If True, mean-center X before computing covariances.

    Returns
    -------
    w : array, shape (n_channels,)
        LDA projection vector (not normalized to unit length).
    """
    X = np.asarray(X, float)
    y = np.asarray(y).ravel()

    # Normalize labels to {0,1}
    labels = np.unique(y)
    if labels.size != 2:
        raise ValueError("fit_lda_filter requires exactly two classes.")
    y_bin = (y == labels[1]).astype(int)

    if center:
        Xc = X - X.mean(axis=0, keepdims=True)
    else:
        Xc = X

    X0 = Xc[y_bin == 0]
    X1 = Xc[y_bin == 1]

    m0 = X0.mean(axis=0)
    m1 = X1.mean(axis=0)
    dm = (m1 - m0)

    # Within-class covariance (pooled)
    # Using unbiased covariance via row-vectors -> cov = (X^T X) / (n-1)
    def cov_rows(A):
        # A: (n, p), returns (p, p)
        n = A.shape[0]
        if n <= 1:
            return np.zeros((A.shape[1], A.shape[1]))
        return (A.T @ A) / (n - 1)

    S0 = cov_rows(X0 - m0)
    S1 = cov_rows(X1 - m1)
    Sw = S0 + S1

    # Regularization for numeric stability
    p = Sw.shape[0]
    Sw_reg = Sw + reg * np.eye(p)

    # Solve Sw_reg * w = dm  (equivalent to w ∝ Sw^{-1} dm)
    # Use solve rather than explicit inverse
    w = np.linalg.solve(Sw_reg, dm)

    # Optional: normalize w for interpretability (not required)
    # Scale so that projected class means are ~±1
    proj0 = X0 @ w
    proj1 = X1 @ w
    s = (np.abs(proj0.mean()) + np.abs(proj1.mean()))
    if s > 0:
        w = w / s

    return w


def apply_lda_filter(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Apply LDA spatial filter.
    Parameters
    ----------
    X : array, shape (n_samples, n_channels) or (n_time, n_channels)
    w : array, shape (n_channels,)
    Returns
    -------
    y : array, shape (n_samples,)
        1D discriminant signal.
    """
    X = np.asarray(X, float)
    w = np.asarray(w, float).ravel()
    if X.ndim != 2:
        raise ValueError("X must be 2D: (n_samples, n_channels)")
    if X.shape[1] != w.size:
        raise ValueError(f"w length {w.size} != n_channels {X.shape[1]}")
    return X @ w


# -------------------------------
# Utilities for testing/plots
# -------------------------------

def synth_two_class_timeseries(
    n_channels: int = 8,
    fs: float = 250.0,
    seconds_per_class: float = 8.0,
    f0: float = 10.0,     # dominant freq for class 0
    f1: float = 12.0,     # dominant freq for class 1
    snr_db: float = -5.0, # negative = quite noisy
    active_ch: tuple[int, ...] = (0, 1, 2),
    seed: int = 7,
):
    """
    Create two blocks of multichannel data with different narrowband activity.
    Returns concatenated data X (time, ch) and labels y per-sample.
    """
    rng = np.random.default_rng(seed)
    n0 = int(seconds_per_class * fs)
    n1 = int(seconds_per_class * fs)
    t0 = np.arange(n0) / fs
    t1 = np.arange(n1) / fs

    # Base noise for all channels
    def colored_noise(n, p, beta=1.0):
        # 1/f^beta-ish noise via frequency shaping
        # Build a random complex spectrum directly in the rFFT domain
        # (avoids calling rfft() on a complex time-domain array).
        f = np.fft.rfftfreq(n, d=1/fs)
        # avoid division by zero at f==0 by flooring to 1/(n/fs)
        shape = 1.0 / np.maximum(f, 1.0 / (n / fs)) ** (beta / 2.0)
        # random complex coefficients for the rfft output length
        Xf_r = rng.normal(size=(f.size, p)) + 1j * rng.normal(size=(f.size, p))
        Xf_r *= shape[:, None]
        # invert back to time domain using irfft
        x = np.fft.irfft(Xf_r, n=n, axis=0).real
        x /= np.std(x, axis=0, keepdims=True) + 1e-12
        return x

    X0 = colored_noise(n0, n_channels, beta=1.0)
    X1 = colored_noise(n1, n_channels, beta=1.0)

    # Inject class-specific narrowband on a subset of channels
    amp = 1.0
    s0 = np.sin(2*np.pi*f0*t0)[:, None] * amp
    s1 = np.sin(2*np.pi*f1*t1)[:, None] * amp
    for ch in active_ch:
        X0[:, ch] += s0[:, 0]
        X1[:, ch] += s1[:, 0]

    # Mix SNR
    def set_snr(x, target_snr_db):
        sig_pow = np.mean(x**2, axis=0, keepdims=True)
        noise = rng.normal(size=x.shape)
        noise_pow = np.mean(noise**2, axis=0, keepdims=True)
        k = np.sqrt(sig_pow / (noise_pow * 10**(target_snr_db/10.0)))
        return x + k * noise

    X0 = set_snr(X0, snr_db)
    X1 = set_snr(X1, snr_db)

    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(n0, int), np.ones(n1, int)]
    t = np.arange(X.shape[0]) / fs
    return X, y, t, fs


def welch_amp(x, fs, nperseg=512):
    """Return frequencies and sqrt(PSD) for amplitude-like plotting."""
    f, pxx = welch(x, fs=fs, nperseg=min(nperseg, len(x)))
    return f, np.sqrt(pxx)


# -------------------------------
# Demo / Visualization
# -------------------------------

def demo():
    # --- Synthesize training data ---
    X, y, t, fs = synth_two_class_timeseries(
        n_channels=8,
        fs=250.0,
        seconds_per_class=8.0,
        f0=10.0,
        f1=12.0,
        snr_db=-5.0,
        active_ch=(0, 1, 2),
        seed=7,
    )

    # --- Fit LDA spatial filter on a sub-sampled set (for speed) ---
    # Using per-sample features here; for real EEG, prefer epoch averages/features.
    idx_train = np.r_[np.arange(0, len(y), 2)]  # take every other sample
    w = fit_lda_filter(X[idx_train], y[idx_train], reg=1e-2, center=True)

    # --- Apply to full time-series ---
    y_lda = apply_lda_filter(X, w)

    # --- Visualizations ---
    # 1) Time-domain: pick a representative raw channel vs LDA output
    raw_ch = 0
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_t, ax_f, ax_sep, ax_t_zoom = axes.ravel()

    ax_t.plot(t, X[:, raw_ch], lw=0.8, label=f"Raw ch {raw_ch}")
    ax_t.plot(t, y_lda, lw=0.9, alpha=0.9, label="LDA projection")
    ax_t.set_title("Time domain (full)")
    ax_t.set_xlabel("Time (s)")
    ax_t.set_ylabel("Amplitude (a.u.)")
    ax_t.legend(loc="upper right")
    ax_t.axvline(t[len(t)//2], ls="--", alpha=0.5, label="Class switch")

    # 2) Frequency-domain: √PSD of raw vs LDA over the whole record
    fr, Ar = welch_amp(X[:, raw_ch], fs)
    fl, Al = welch_amp(y_lda, fs)
    ax_f.plot(fr, Ar, lw=1.0, label=f"Raw ch {raw_ch}")
    ax_f.plot(fl, Al, lw=1.0, label="LDA projection")
    ax_f.set_title("Frequency domain (Welch √PSD)")
    ax_f.set_xlabel("Frequency (Hz)")
    ax_f.set_ylabel("Amplitude (a.u.)")
    ax_f.set_xlim(0, 40)
    ax_f.legend()

    # 3) Class separation in 1D (project short windows)
    # We'll downsample y_lda and color by class to show separation
    step = int(fs * 0.05)  # 50 ms
    idx = np.arange(0, len(y), max(1, step))
    ax_sep.scatter(idx / fs, y_lda[idx], c=y[idx], s=12, cmap="coolwarm", alpha=0.8)
    ax_sep.set_title("Projected values over time (color = class)")
    ax_sep.set_xlabel("Time (s)")
    ax_sep.set_ylabel("LDA output")
    ax_sep.axvline(t[len(t)//2], ls="--", alpha=0.5)

    # 4) Short zoom to compare raw vs LDA around the transition
    margin = int(2 * fs)
    mid = len(t) // 2
    sl = slice(max(0, mid - margin), min(len(t), mid + margin))
    ax_t_zoom.plot(t[sl], X[sl, raw_ch], lw=0.9, label=f"Raw ch {raw_ch}")
    ax_t_zoom.plot(t[sl], y_lda[sl], lw=1.0, label="LDA projection")
    ax_t_zoom.set_title("Time domain (zoom around class switch)")
    ax_t_zoom.set_xlabel("Time (s)")
    ax_t_zoom.set_ylabel("Amplitude (a.u.)")
    ax_t_zoom.legend()

    plt.suptitle("LDA Spatial Filter: Before vs After")
    plt.tight_layout()
    plt.show()

    # Print the learned weights for inspection
    print("LDA spatial weights (per channel):")
    for i, wi in enumerate(w):
        print(f"  ch {i}: {wi:+.4f}")


if __name__ == "__main__":
    demo()
