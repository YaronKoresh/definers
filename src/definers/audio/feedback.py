from __future__ import annotations

import numpy as np

from definers.logger import init_logger

from .dependencies import librosa_module

_logger = init_logger()


def get_color_palette(name: str) -> list[tuple[int, int, int]]:
    palettes = {
        "Cyberpunk": [(0, 255, 255), (255, 0, 128), (128, 0, 255)],
        "Sunset": [(255, 94, 77), (255, 195, 0), (199, 0, 57)],
        "Ocean": [(0, 105, 148), (0, 168, 107), (72, 209, 204)],
        "Toxic": [(57, 255, 20), (170, 255, 0), (20, 20, 20)],
        "Gold": [(255, 215, 0), (218, 165, 32), (50, 50, 50)],
        "Israel": [(0, 56, 184), (255, 255, 255), (200, 200, 255)],
        "Matrix": [(0, 255, 0), (0, 128, 0), (0, 50, 0)],
        "Neon Red": [(255, 0, 0), (100, 0, 0), (20, 0, 0)],
        "Deep Space": [(10, 10, 30), (138, 43, 226), (75, 0, 130)],
    }
    return palettes.get(name, palettes["Cyberpunk"])


def get_audio_feedback(audio_path: str) -> str | None:

    if not audio_path:
        return None

    try:
        librosa = librosa_module()
        (y_stereo, sr) = librosa.load(audio_path, sr=None, mono=False)
        y_mono = librosa.to_mono(y_stereo) if y_stereo.ndim > 1 else y_stereo
        rms = librosa.feature.rms(y=y_mono)[0]
        stft = librosa.stft(y_mono)
        freqs = librosa.fft_frequencies(sr=sr)
        bass_energy = np.mean(np.abs(stft[(freqs >= 20) & (freqs < 250)]))
        high_energy = np.mean(np.abs(stft[(freqs >= 5000) & (freqs < 20000)]))
        peak_amp = np.max(np.abs(y_mono))
        mean_rms = np.mean(rms)
        crest_factor = 20 * np.log10(peak_amp / mean_rms) if mean_rms > 0 else 0
        stereo_width = 0
        if y_stereo.ndim > 1 and y_stereo.shape[0] == 2:
            from scipy.stats import pearsonr

            (corr, _) = pearsonr(y_stereo[0], y_stereo[1])
            stereo_width = (1 - corr) * 100

        feedback = "### AI Track Feedback\n\n"
        feedback += "#### Technical Analysis\n"
        feedback += f"- **Loudness & Dynamics:** The track has a crest factor of **{crest_factor:.2f} dB**. "
        if crest_factor > 14:
            feedback += "This suggests the track is very dynamic and punchy.\n"
        elif crest_factor > 8:
            feedback += "This is a good balance between punch and loudness, typical for many genres.\n"
        else:
            feedback += "This suggests the track is heavily compressed or limited, prioritizing loudness over dynamic range.\n"
        feedback += f"- **Stereo Image:** The stereo width is estimated at **{stereo_width:.1f}%**. "
        if stereo_width > 60:
            feedback += "The mix feels wide and immersive.\n"
        elif stereo_width > 20:
            feedback += "The mix has a balanced stereo field.\n"
        else:
            feedback += "The mix is narrow or mostly mono.\n"
        feedback += f"- **Frequency Balance:** Bass energy is at **{bass_energy:.2f}** and high-frequency energy is at **{high_energy:.2f}**. "
        if bass_energy > high_energy * 2:
            feedback += "The track is bass-heavy.\n"
        elif high_energy > bass_energy * 2:
            feedback += "The track is bright or treble-heavy.\n"
        else:
            feedback += (
                "The track has a relatively balanced frequency spectrum.\n"
            )
        feedback += "\n#### Advice\n"
        if crest_factor < 8:
            feedback += "- **Compression:** The track might be over-compressed. Consider reducing the amount of compression to bring back some life and punch to the transients.\n"
        if stereo_width < 20 and y_stereo.ndim > 1:
            feedback += "- **Stereo Width:** To make the mix sound bigger, try using stereo widening tools or panning instruments differently to create more space.\n"
        if bass_energy > high_energy * 2.5:
            feedback += "- **Bass Management:** The low-end might be overpowering. Ensure it's not masking other instruments. A high-pass filter on non-bass elements can clean up muddiness.\n"
        if high_energy > bass_energy * 2.5:
            feedback += "- **Tame the Highs:** The track is very bright, which can be fatiguing. Check for harshness in cymbals or vocals, and consider using a de-esser or a gentle high-shelf cut.\n"
        if mean_rms < 0.05:
            feedback += "- **Mastering:** The overall volume is low. The track would benefit from mastering to increase its loudness and competitiveness with commercial tracks.\n"
        else:
            feedback += "- **General Mix:** The track has a solid technical foundation. Focus on creative choices, arrangement, and ensuring all elements have their own space in the mix.\n"
        return feedback
    except Exception:
        _logger.exception("Audio feedback analysis failed")
        return None
