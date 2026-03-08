"""
Phase 1: Signal Transformation — Generalized S-Transform (GST)
================================================================
Paper: Dalai & Kumar (2025) — Hybrid Quantum-Deep Learning Pipeline
       for Seismic P-wave Detection

The Generalized S-Transform extends the standard S-Transform by introducing
a tunable parameter (lambda / alpha) that controls the width of the Gaussian
window in the time-frequency plane.

Standard S-Transform window:  w(t, f) = |f| / sqrt(2π) * exp(-t²f²/2)
Generalized version:          w(t, f) = |f|^λ / (α·sqrt(2π)) * exp(-t²|f|^(2λ) / 2α²)

This gives better frequency resolution for low-frequency seismic content
(P-waves typically appear at 1–10 Hz) while preserving time localization
at higher frequencies.

Output spectrograms: shape (1200, 64, 200) — (samples, freq_bins, time_bins)
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import resample
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time


# ─────────────────────────────────────────────
#  GST CORE
# ─────────────────────────────────────────────

def generalized_s_transform(signal: np.ndarray,
                             sample_rate: float = 100.0,
                             freq_bins: int = 64,
                             lam: float = 1.0,
                             alpha: float = 1.0) -> np.ndarray:
    """
    Compute the Generalized S-Transform of a 1D seismic signal.

    Parameters
    ----------
    signal      : 1D numpy array — raw waveform
    sample_rate : sampling rate in Hz
    freq_bins   : number of frequency bins in output (rows of spectrogram)
    lam         : lambda — controls Gaussian window shape (1.0 = standard ST)
    alpha       : alpha  — controls window width scaling

    Returns
    -------
    amplitude   : 2D numpy array of shape (freq_bins, len(signal))
                  |GST coefficients| at each (frequency, time) point
    freqs       : 1D array of frequency axis values (Hz)
    """
    N = len(signal)
    # FFT of the signal
    H = fft(signal)

    # Frequency axis (full FFT resolution)
    f_full = fftfreq(N, d=1.0 / sample_rate)  # shape: (N,)

    # Select positive frequencies only, mapped to freq_bins
    f_max = sample_rate / 2.0                  # Nyquist
    f_positive = np.linspace(0.0, f_max, freq_bins + 1)[1:]  # exclude DC

    # Pre-allocate output: (freq_bins, N)
    gst_matrix = np.zeros((freq_bins, N), dtype=np.complex64)

    for k, f in enumerate(f_positive):
        if f == 0:
            # DC component: just the mean of the signal
            gst_matrix[k, :] = np.mean(signal)
            continue

        # Generalized Gaussian window in frequency domain
        # G(α, f) = exp(-2π²α²n²/|f|^(2λ))  where n = f_full
        # (derived from the time-domain Gaussian via Fourier shift theorem)
        exp_arg = -2.0 * (np.pi ** 2) * (alpha ** 2) * (f_full ** 2) / (np.abs(f) ** (2 * lam))
        window_freq = np.exp(exp_arg.astype(np.float32))

        # Shift H by frequency f, multiply by window, inverse FFT → one row
        shift = int(round(f * N / sample_rate))
        H_shifted = np.roll(H, -shift)
        voice = ifft(H_shifted * window_freq)
        gst_matrix[k, :] = voice.astype(np.complex64)

    amplitude = np.abs(gst_matrix).astype(np.float32)
    return amplitude, f_positive


def resize_time_axis(spectrogram: np.ndarray, target_time_bins: int) -> np.ndarray:
    """
    Downsample the time axis of a spectrogram to target_time_bins using
    scipy.signal.resample (sinc interpolation — preserves spectral content).
    """
    if spectrogram.shape[1] == target_time_bins:
        return spectrogram
    resampled = resample(spectrogram, target_time_bins, axis=1)
    return resampled.astype(np.float32)


def normalize_spectrogram(spec: np.ndarray) -> np.ndarray:
    """
    Per-sample min-max normalization → [0, 1].
    Avoids division by zero for flat signals.
    """
    s_min, s_max = spec.min(), spec.max()
    if s_max - s_min < 1e-10:
        return np.zeros_like(spec)
    return (spec - s_min) / (s_max - s_min)


# ─────────────────────────────────────────────
#  BATCH PROCESSING
# ─────────────────────────────────────────────

def transform_dataset(seismograms: np.ndarray,
                      sample_rate: float = 100.0,
                      freq_bins: int = 64,
                      time_bins: int = 200,
                      lam: float = 1.2,
                      alpha: float = 1.0,
                      verbose: bool = True) -> np.ndarray:
    """
    Apply GST to every waveform in the dataset.

    Parameters
    ----------
    seismograms : shape (N, signal_length)
    sample_rate : Hz
    freq_bins   : frequency resolution of output spectrograms
    time_bins   : time resolution of output spectrograms
    lam         : GST lambda parameter (>1 = better low-freq resolution)
    alpha       : GST alpha parameter
    verbose     : print progress

    Returns
    -------
    spectrograms : shape (N, freq_bins, time_bins), float32, normalized [0,1]
    """
    N = seismograms.shape[0]
    spectrograms = np.zeros((N, freq_bins, time_bins), dtype=np.float32)

    t0 = time.time()
    for i in range(N):
        spec, _ = generalized_s_transform(
            seismograms[i],
            sample_rate=sample_rate,
            freq_bins=freq_bins,
            lam=lam,
            alpha=alpha
        )
        spec = resize_time_axis(spec, time_bins)
        spectrograms[i] = normalize_spectrogram(spec)

        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (N - i - 1) / rate
            print(f"  [{i+1:4d}/{N}]  elapsed: {elapsed:.1f}s  "
                  f"rate: {rate:.1f} sig/s  eta: {remaining:.1f}s")

    return spectrograms


# ─────────────────────────────────────────────
#  VISUALIZATION
# ─────────────────────────────────────────────

def visualize_sample(seismogram: np.ndarray,
                     spectrogram: np.ndarray,
                     label: int,
                     p_arrival: int,
                     sample_rate: float = 100.0,
                     freq_bins: int = 64,
                     time_bins: int = 200,
                     save_path: str = None):
    """
    Plot a raw waveform alongside its GST spectrogram for visual verification.
    """
    duration = len(seismogram) / sample_rate
    t_axis = np.linspace(0, duration, len(seismogram))
    f_axis = np.linspace(0, sample_rate / 2, freq_bins)
    t_spec_axis = np.linspace(0, duration, time_bins)

    label_str = "Earthquake" if label == 1 else "Noise"
    color = "#e74c3c" if label == 1 else "#3498db"

    fig = plt.figure(figsize=(14, 7), facecolor="#0f1117")
    fig.suptitle(f"Phase 1 — GST Output | Sample: {label_str}",
                 color="white", fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 1, hspace=0.45)

    # ── Top: Raw waveform ──
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t_axis, seismogram, color=color, linewidth=0.7, alpha=0.9)
    if label == 1 and p_arrival >= 0:
        p_time = p_arrival / sample_rate
        ax1.axvline(p_time, color="#f1c40f", linewidth=1.5,
                    linestyle="--", label=f"P-arrival ({p_time:.2f}s)")
        ax1.legend(fontsize=9, facecolor="#1a1d2e", labelcolor="white",
                   framealpha=0.8)
    ax1.set_title("Raw Waveform (1D Seismogram)", color="white", fontsize=11)
    ax1.set_xlabel("Time (s)", color="#aaaaaa")
    ax1.set_ylabel("Amplitude", color="#aaaaaa")
    ax1.set_facecolor("#1a1d2e")
    ax1.tick_params(colors="#aaaaaa")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333355")

    # ── Bottom: GST Spectrogram ──
    ax2 = fig.add_subplot(gs[1])
    img = ax2.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        extent=[0, duration, f_axis[0], f_axis[-1]],
        cmap="inferno",
        interpolation="bilinear",
        vmin=0, vmax=1
    )
    if label == 1 and p_arrival >= 0:
        p_time = p_arrival / sample_rate
        ax2.axvline(p_time, color="#f1c40f", linewidth=1.5,
                    linestyle="--", label=f"P-arrival ({p_time:.2f}s)")
        ax2.legend(fontsize=9, facecolor="#1a1d2e", labelcolor="white",
                   framealpha=0.8)
    ax2.set_title("GST Spectrogram (2D Time-Frequency)", color="white", fontsize=11)
    ax2.set_xlabel("Time (s)", color="#aaaaaa")
    ax2.set_ylabel("Frequency (Hz)", color="#aaaaaa")
    ax2.set_facecolor("#1a1d2e")
    ax2.tick_params(colors="#aaaaaa")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333355")

    cbar = fig.colorbar(img, ax=ax2, pad=0.02)
    cbar.set_label("Normalized Amplitude", color="#aaaaaa", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#aaaaaa")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aaaaaa")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Visualization saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    # ── Config ──────────────────────────────
    DATA_DIR        = "data"
    OUTPUT_DIR      = "data"
    VIZ_DIR         = "outputs"
    SAMPLE_RATE     = 100.0   # Hz
    FREQ_BINS       = 64      # frequency resolution
    TIME_BINS       = 200     # time resolution (downsampled from 2000)
    GST_LAMBDA      = 1.2     # >1 improves low-freq resolution (good for P-waves)
    GST_ALPHA       = 1.0     # window width scaling

    os.makedirs(VIZ_DIR, exist_ok=True)

    # ── Load data ───────────────────────────
    print("=" * 55)
    print("  PHASE 1 — Generalized S-Transform")
    print("=" * 55)
    print(f"\n[1/4] Loading dataset from '{DATA_DIR}/'...")

    seismograms = np.load(os.path.join(DATA_DIR, "seismograms.npy"))
    labels      = np.load(os.path.join(DATA_DIR, "labels.npy"))
    p_arrivals  = np.load(os.path.join(DATA_DIR, "p_arrivals.npy"))

    print(f"      seismograms : {seismograms.shape}")
    print(f"      labels      : {labels.shape}  "
          f"(noise={( labels==0).sum()}, eq={(labels==1).sum()})")
    print(f"      p_arrivals  : {p_arrivals.shape}")

    # ── Transform ───────────────────────────
    print(f"\n[2/4] Applying GST  "
          f"(λ={GST_LAMBDA}, α={GST_ALPHA}, "
          f"freq_bins={FREQ_BINS}, time_bins={TIME_BINS})...")
    spectrograms = transform_dataset(
        seismograms,
        sample_rate=SAMPLE_RATE,
        freq_bins=FREQ_BINS,
        time_bins=TIME_BINS,
        lam=GST_LAMBDA,
        alpha=GST_ALPHA,
        verbose=True
    )
    print(f"\n      spectrograms: {spectrograms.shape}  dtype={spectrograms.dtype}")

    # ── Save ────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, "spectrograms.npy")
    print(f"\n[3/4] Saving → {out_path}")
    np.save(out_path, spectrograms)
    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    print(f"      File size: {size_mb:.1f} MB")

    # ── Visualize ───────────────────────────
    print(f"\n[4/4] Generating visualizations...")

    # Pick one earthquake and one noise sample
    eq_idx    = np.where(labels == 1)[0][0]
    noise_idx = np.where(labels == 0)[0][0]

    for idx, tag in [(eq_idx, "earthquake"), (noise_idx, "noise")]:
        visualize_sample(
            seismogram   = seismograms[idx],
            spectrogram  = spectrograms[idx],
            label        = labels[idx],
            p_arrival    = p_arrivals[idx],
            sample_rate  = SAMPLE_RATE,
            freq_bins    = FREQ_BINS,
            time_bins    = TIME_BINS,
            save_path    = os.path.join(VIZ_DIR, f"gst_sample_{tag}.png")
        )

    print("\n" + "=" * 55)
    print("  Phase 1 complete!")
    print(f"  Output : data/spectrograms.npy  {spectrograms.shape}")
    print(f"  Plots  : {VIZ_DIR}/gst_sample_*.png")
    print("=" * 55)


if __name__ == "__main__":
    main()
