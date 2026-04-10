from __future__ import annotations

import tempfile

from definers.runtime_numpy import get_numpy_module

np = get_numpy_module()

from definers.logger import init_logger

from .dependencies import librosa_module

_logger = init_logger()


def create_spectrum_visualization(audio_path: str) -> str | None:
    import importlib

    plt = importlib.import_module("matplotlib.pyplot")
    librosa = librosa_module()

    try:
        y, sr = librosa.load(audio_path, sr=None)
        n_fft = 8192
        hop_length = 512
        stft_result = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft_result)
        avg_magnitude = np.mean(magnitude, axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        magnitude_db = librosa.amplitude_to_db(avg_magnitude, ref=np.max)
        fig, ax = plt.subplots(figsize=(8, 5), facecolor="#f0f0f0")
        ax.set_facecolor("white")
        ax.fill_between(
            freqs,
            magnitude_db,
            y2=np.min(magnitude_db) - 1,
            color="#7c3aed",
            alpha=0.8,
            zorder=2,
        )
        ax.plot(freqs, magnitude_db, color="#4c2a8c", linewidth=1, zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(20, sr / 2)
        ax.set_ylim(np.min(magnitude_db) - 1, np.max(magnitude_db) + 5)
        xticks = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        xtick_labels = [
            "50",
            "100",
            "200",
            "500",
            "1k",
            "2k",
            "5k",
            "10k",
            "20k",
        ]
        ax.set_xticks([x for x in xticks if x < sr / 2])
        ax.set_xticklabels(
            [label for (x, label) in zip(xticks, xtick_labels) if x < sr / 2]
        )
        ax.grid(True, which="both", ls="--", color="gray", alpha=0.6, zorder=1)
        ax.set_title("Frequency Analysis", color="black")
        ax.set_xlabel("Frequency (Hz)", color="black")
        ax.set_ylabel("Amplitude (dB)", color="black")
        ax.tick_params(colors="black", which="both")
        audible_mask = freqs > 20
        if np.any(audible_mask):
            peak_idx = int(np.argmax(magnitude_db[audible_mask]))
            peak_freq = freqs[audible_mask][peak_idx]
            peak_db = magnitude_db[audible_mask][peak_idx]
            peak_text = f"Peak: {peak_freq:.0f} Hz at {peak_db:.1f} dB"
            ax.text(
                0.98,
                0.95,
                peak_text,
                transform=ax.transAxes,
                color="black",
                ha="right",
                va="top",
            )
        fig.tight_layout()
        with tempfile.NamedTemporaryFile(
            suffix=".png", delete=False
        ) as tmpfile:
            temp_path = tmpfile.name
        fig.savefig(temp_path, facecolor=fig.get_facecolor())
        plt.close(fig)
        return temp_path
    except Exception:
        _logger.exception("Error creating spectrum")
        return None
