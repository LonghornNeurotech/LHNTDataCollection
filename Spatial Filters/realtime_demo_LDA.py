# Kailash Boggaram
#!/usr/bin/env python3
"""
Realtime simulation demo for the LDA spatial filter.
- Uses `synth_two_class_timeseries()` and `fit_lda_filter()` from `lda_filter.py`.
- Simulates a streaming source by yielding one sample at a time from synthetic data.
- Trains LDA on a warm-up period, then displays a scrolling plot of a raw channel
  and the LDA projection updated in (near) real-time.

Run:
    python3 realtime_demo.py

Close the matplotlib window to end the demo.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lda_filter import synth_two_class_timeseries, fit_lda_filter


def stream_gen(X: np.ndarray, y: np.ndarray):
    """Yield one (sample, label) pair at a time to simulate streaming."""
    for i in range(len(y)):
        yield X[i], int(y[i])


def run_realtime_demo(
    seconds_per_class: float = 8.0,
    fs: float = 250.0,
    warmup_seconds: float = 1.0,
    display_seconds: float = 3.0,
    raw_ch: int = 0,
    update_interval_ms: int = 50,
):
    # --- generate synthetic dataset ---
    X, y, t, fs = synth_two_class_timeseries(
        n_channels=8,
        fs=fs,
        seconds_per_class=seconds_per_class,
        f0=10.0,
        f1=12.0,
        snr_db=-5.0,
        active_ch=(0, 1, 2),
        seed=7,
    )

    n_samples, n_channels = X.shape
    # We'll sample warm-up training points evenly across the recording so the
    # initial training contains examples from both classes (the synthetic data
    # places classes in contiguous blocks, so taking only the first second
    # could yield a single-class training set).
    warmup_samples = int(warmup_seconds * fs)
    buf_len = int(display_seconds * fs)

    # Circular buffers for plotting
    buf_raw = np.zeros(buf_len)
    buf_lda = np.zeros(buf_len)

    # Build warm-up training set by sampling evenly across the full recording
    warmup_samples = min(warmup_samples, n_samples)
    warm_idx = np.linspace(0, n_samples - 1, warmup_samples, dtype=int)
    warm_X = X[warm_idx]
    warm_y = y[warm_idx]

    w = fit_lda_filter(warm_X, warm_y, reg=1e-2, center=True)

    # create the stream generator (start streaming from the beginning)
    it = stream_gen(X, y)

    # Setup plotting
    fig, ax = plt.subplots(figsize=(10, 4))
    x_axis = np.linspace(-display_seconds, 0.0, buf_len)
    line_raw, = ax.plot(x_axis, buf_raw, label=f"Raw ch {raw_ch}")
    line_lda, = ax.plot(x_axis, buf_lda, label="LDA projection")
    ax.set_xlim(-display_seconds, 0.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title("Realtime LDA projection (simulated stream)")
    ax.legend(loc="upper right")

    # Optionally print initial weights
    print("Initial LDA weights:")
    for i, wi in enumerate(w):
        print(f"  ch {i}: {wi:+.4f}")

    # Update function called by FuncAnimation
    def update(frame):
        nonlocal w, buf_raw, buf_lda
        try:
            x_sample, lab = next(it)
        except StopIteration:
            # stop animation gracefully
            return line_raw, line_lda

        # push new sample into buffers (scrolling)
        buf_raw = np.roll(buf_raw, -1)
        buf_raw[-1] = x_sample[raw_ch]

        buf_lda = np.roll(buf_lda, -1)
        buf_lda[-1] = float(np.dot(x_sample, w))

        # Update lines
        line_raw.set_ydata(buf_raw)
        line_lda.set_ydata(buf_lda)

        # Optionally: update w every N samples (commented out by default)
        # Could accumulate labeled data and retrain periodically
        return line_raw, line_lda

    ani = FuncAnimation(fig, update, interval=update_interval_ms, blit=True)
    plt.show()


if __name__ == "__main__":
    run_realtime_demo()
