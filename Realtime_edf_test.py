# Real-time EDF filter tester + live comparisons
# Usage examples:
#   python realtime_test_edf.py --edf ./data/sample.edf --channel 0 --fs 250 \
#       --low 1 --high 40 --notch 60 --width 2 --harmonics 120 180 --block 1.0
#
#   python realtime_test_edf.py --edf ./data/sample.edf --channel-name "Cz" \
#       --low 0.5 --high 45 --notch 60 --width 1.5

from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import welch
import time

# Choose one reader: MNE (recommended) or pyedflib fallback
# pip install mne pyedflib numpy scipy matplotlib
try:
    import mne
    HAVE_MNE = True
except Exception:
    HAVE_MNE = False

try:
    import pyedflib
    HAVE_PYEDF = True
except Exception:
    HAVE_PYEDF = False

from advanced_filtering import spectrum_interpolation_notch, zero_phase_bandpass

def read_edf_all(edf_path: str, preload: bool = True):
    """
    Returns (data [n_channels, n_samples], fs, channel_names)
    """
    if HAVE_MNE:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        fs = float(raw.info["sfreq"])
        data = raw.get_data()  # shape (n_channels, n_samples)
        ch_names = raw.ch_names
        return data, fs, ch_names
    elif HAVE_PYEDF:
        f = pyedflib.EdfReader(edf_path)
        n = f.signals_in_file
        ch_names = [f.getLabel(i) for i in range(n)]
        fs = float(f.getSampleFrequency(0))
        lens = [f.getNSamples()[i] for i in range(n)]
        N = min(lens)
        data = np.vstack([f.readSignal(i)[:N] for i in range(n)])
        f.close()
        return data, fs, ch_names
    else:
        raise RuntimeError(
            "Neither mne nor pyedflib available. Please pip install mne or pyedflib."
        )

def main():
    p = argparse.ArgumentParser(description="Real-time EDF filter tester")
    p.add_argument("--edf", required=True, help="Path to .edf file")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--channel", type=int, help="Channel index (0-based)")
    group.add_argument("--channel-name", type=str, help="Channel name (e.g., 'Cz')")
    p.add_argument("--fs", type=float, default=None, help="Override sampling rate (Hz)")
    p.add_argument("--block", type=float, default=1.0, help="Block seconds (stream step)")
    p.add_argument("--low", type=float, default=None, help="Bandpass lowcut (Hz)")
    p.add_argument("--high", type=float, default=None, help="Bandpass highcut (Hz)")
    p.add_argument("--order", type=int, default=4, help="Bandpass Butterworth order")
    p.add_argument("--notch", type=float, default=None, help="Notch center freq (e.g., 60)")
    p.add_argument("--width", type=float, default=2.0, help="Notch total width in Hz")
    p.add_argument("--harmonics", type=float, nargs="*", default=[], help="Extra freqs to remove")
    p.add_argument("--welch_sec", type=float, default=1.0, help="Welch segment length seconds")
    p.add_argument("--fft_n", type=int, default=None, help="Override FFT length (power of 2 recommended)")
    p.add_argument("--x_lim", type=float, nargs=2, default=None, help="Time xlim (sec), e.g., 0 5")
    p.add_argument("--f_lim", type=float, nargs=2, default=None, help="Freq xlim, e.g., 0 80")
    p.add_argument("--max_seconds", type=float, default=None, help="Stop after N seconds (for batch demo)")
    args = p.parse_args()

    data, fs_file, ch_names = read_edf_all(args.edf, preload=True)
    fs = float(args.fs) if args.fs is not None else fs_file

    # Pick channel
    if args.channel_name is not None:
        if args.channel_name not in ch_names:
            raise ValueError(f"Channel name '{args.channel_name}' not in {ch_names}")
        ch_idx = ch_names.index(args.channel_name)
    else:
        ch_idx = int(args.channel) if args.channel is not None else 0

    x = data[ch_idx].astype(float, copy=False)
    n = x.size
    block = int(round(args.block * fs))
    if block <= 0:
        raise ValueError("block seconds too small.")
    if block > n:
        block = n

    # Prepare figure
    plt.figure(figsize=(12, 7))
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], wspace=0.25, hspace=0.35)

    ax_t = plt.subplot(gs[0, 0])
    ax_f = plt.subplot(gs[0, 1])
    ax_t2 = plt.subplot(gs[1, 0])
    ax_f2 = plt.subplot(gs[1, 1])

    ax_t.set_title("Raw (time)")
    ax_t2.set_title("Filtered (time)")
    ax_f.set_title("Raw (frequency)")
    ax_f2.set_title("Filtered (frequency)")

    t_line, = ax_t.plot([], [], lw=1)
    t2_line, = ax_t2.plot([], [], lw=1)
    f_line, = ax_f.plot([], [], lw=1)
    f2_line, = ax_f2.plot([], [], lw=1)

    # x-limits
    ax_t.set_xlabel("Time (s)")
    ax_t2.set_xlabel("Time (s)")
    ax_t.set_ylabel("μV (raw units)")
    ax_t2.set_ylabel("μV (raw units)")

    if args.x_lim:
        ax_t.set_xlim(args.x_lim)
        ax_t2.set_xlim(args.x_lim)

    if args.f_lim:
        ax_f.set_xlim(args.f_lim)
        ax_f2.set_xlim(args.f_lim)

    # State for streaming
    start_idx = 0
    t0 = time.time()
    stopped = False

    # Precompute time vector for block
    t_block = np.arange(block) / fs

    # Helper: filter one vector
    def apply_filters(x_block):
        y = x_block.copy()
        # Optional bandpass first (recommended), then notch interpolation
        if args.low is not None and args.high is not None:
            y = zero_phase_bandpass(y, args.low, args.high, fs=fs, order=args.order, dc_block=True)
        if args.notch is not None:
            y = spectrum_interpolation_notch(y, fs, notch_freq=args.notch, width_hz=args.width, harmonics=tuple(args.harmonics))
        return y

    # Helper: compute Welch PSD or simple FFT magnitude
    def power_spectrum(x_block):
        if args.welch_sec and args.welch_sec > 0:
            nperseg = max(8, int(round(args.welch_sec * fs)))
            f, pxx = welch(x_block, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
            # Convert to amplitude-like scale (sqrt) for visual comparability with FFT magnitude
            return f, np.sqrt(pxx)
        else:
            nfft = args.fft_n if args.fft_n else int(2 ** np.ceil(np.log2(len(x_block))))
            X = np.fft.rfft(x_block, n=nfft)
            f = np.fft.rfftfreq(nfft, d=1.0/fs)
            return f, np.abs(X) / max(1, len(x_block))

    def init():
        # initialize empty lines
        t_line.set_data([], [])
        t2_line.set_data([], [])
        f_line.set_data([], [])
        f2_line.set_data([], [])
        return t_line, t2_line, f_line, f2_line

    def update(_frame):
        nonlocal start_idx, stopped
        if stopped:
            return t_line, t2_line, f_line, f2_line

        end_idx = start_idx + block
        if end_idx > n:
            end_idx = n
        x_block = x[start_idx:end_idx]

        # If last block is shorter, pad for nicer plotting
        if x_block.size < block:
            pad = block - x_block.size
            x_block = np.pad(x_block, (0, pad), mode="edge")

        y_block = apply_filters(x_block)

        # Time plots
        t_line.set_data(t_block, x_block)
        t2_line.set_data(t_block, y_block)

        if not args.x_lim:
            ax_t.set_xlim(0, t_block[-1])
            ax_t2.set_xlim(0, t_block[-1])

        # Frequency plots
        fr, Ar = power_spectrum(x_block)
        ff, Af = power_spectrum(y_block)

        f_line.set_data(fr, Ar)
        f2_line.set_data(ff, Af)

        if not args.f_lim:
            # Autoscale once per update
            ax_f.set_xlim(fr[0], fr[-1])
            ax_f2.set_xlim(ff[0], ff[-1])

        # Y autoscale each frame
        for ax in (ax_t, ax_t2, ax_f, ax_f2):
            ax.relim()
            ax.autoscale_view()

        start_idx += block

        # Optional stop after max_seconds
        if args.max_seconds is not None:
            if (time.time() - t0) >= args.max_seconds:
                stopped = True

        return t_line, t2_line, f_line, f2_line

    ani = FuncAnimation(plt.gcf(), update, init_func=init, interval=max(1, int(1000 * args.block * 0.9)), blit=False)
    plt.suptitle(f"EDF: {args.edf} | Channel: {ch_names[ch_idx] if ch_names else ch_idx} | fs={fs:.1f} Hz")
    plt.show()

if __name__ == "__main__":
    main()
