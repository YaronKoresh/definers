from __future__ import annotations

import importlib

import numpy as np

from definers.logger import init_logger
from definers.system.paths import full_path, tmp

_logger = init_logger()


def _load_librosa_backend():
    return importlib.import_module("librosa")


def _load_audio_analysis_backend():
    return importlib.import_module("definers.audio.analysis")


def _load_array_backend():
    return importlib.import_module("definers.data.arrays")


def _load_model_introspection_backend():
    return importlib.import_module("definers.ml.introspection")


def extract_audio_features(
    file_path: str, n_mfcc: int = 20
) -> np.ndarray | None:
    librosa_backend = _load_librosa_backend()
    try:
        (y, sr) = librosa_backend.load(file_path, sr=None)
    except Exception:
        _logger.exception(
            "Failed to load audio for feature extraction: %s", file_path
        )
        return None

    try:
        mfccs = librosa_backend.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, n_mels=80
        ).flatten()
        spectral_centroid = librosa_backend.feature.spectral_centroid(
            y=y, sr=sr
        ).flatten()
        spectral_bandwidth = librosa_backend.feature.spectral_bandwidth(
            y=y, sr=sr
        ).flatten()
        spectral_rolloff = librosa_backend.feature.spectral_rolloff(
            y=y, sr=sr
        ).flatten()
        spectral_features = np.concatenate(
            (spectral_centroid, spectral_bandwidth, spectral_rolloff)
        )
        zero_crossing_rate = librosa_backend.feature.zero_crossing_rate(
            y=y
        ).flatten()
        chroma = librosa_backend.feature.chroma_stft(y=y, sr=sr).flatten()
        all_features = np.concatenate(
            (mfccs, spectral_features, zero_crossing_rate, chroma)
        ).astype(np.float32)
        return all_features
    except Exception:
        _logger.exception("Failed to extract audio features")
        return None


def features_to_audio(
    predicted_features,
    sr: int = 32000,
    n_mfcc: int = 20,
    n_mels: int = 80,
    n_fft: int = 2048,
    hop_length: int = 512,
):
    expected_freq_bins = n_fft // 2 + 1
    librosa_backend = _load_librosa_backend()

    try:
        predicted_features = np.asarray(predicted_features)
        remainder = predicted_features.size % n_mfcc
        if remainder != 0:
            padding_needed = n_mfcc - remainder
            _logger.debug(
                "Padding with %d zeros to make the predicted features (%d) a multiple of n_mfcc (%d).",
                padding_needed,
                predicted_features.size,
                n_mfcc,
            )
            predicted_features = np.pad(
                predicted_features,
                (0, padding_needed),
                mode="constant",
                constant_values=0,
            )
        mfccs = predicted_features.reshape((n_mfcc, -1))
        if mfccs.shape[1] == 0:
            _logger.error(
                "Reshaped MFCCs have zero frames; cannot proceed with audio reconstruction."
            )
            return None

        mel_spectrogram_db = librosa_backend.feature.inverse.mfcc_to_mel(
            mfccs, n_mels=n_mels
        )
        mel_spectrogram = librosa_backend.db_to_amplitude(mel_spectrogram_db)
        mel_spectrogram = np.nan_to_num(
            mel_spectrogram,
            nan=0.0,
            posinf=np.finfo(np.float16).max,
            neginf=np.finfo(np.float16).min,
        )
        mel_spectrogram = np.maximum(0, mel_spectrogram)
        magnitude_spectrogram = librosa_backend.feature.inverse.mel_to_stft(
            M=mel_spectrogram, sr=sr, n_fft=n_fft
        )
        magnitude_spectrogram = np.nan_to_num(
            magnitude_spectrogram,
            nan=0.0,
            posinf=np.finfo(np.float16).max,
            neginf=np.finfo(np.float16).min,
        )
        magnitude_spectrogram = np.maximum(0, magnitude_spectrogram)
        magnitude_spectrogram = np.nan_to_num(
            magnitude_spectrogram,
            nan=0.0,
            posinf=np.finfo(np.float16).max,
            neginf=np.finfo(np.float16).min,
        )

        if magnitude_spectrogram.shape[0] != expected_freq_bins:
            _logger.error(
                "Magnitude spectrogram has incorrect frequency bin count (%d) for n_fft (%d); expected %d. Cannot perform Griffin-Lim.",
                magnitude_spectrogram.shape[0],
                n_fft,
                expected_freq_bins,
            )
            return None

        if magnitude_spectrogram.shape[1] == 0:
            _logger.error(
                "Magnitude spectrogram has zero frames; skipping Griffin-Lim."
            )
            return None

        griffin_lim_iterations = [12, 32]
        for n_iter in griffin_lim_iterations:
            try:
                audio_waveform = librosa_backend.griffinlim(
                    magnitude_spectrogram,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_iter=n_iter,
                )
                if audio_waveform.size > 0:
                    _logger.debug("Griffin-Lim finished %d iterations", n_iter)
                    audio_waveform = np.nan_to_num(
                        audio_waveform,
                        nan=0.0,
                        posinf=np.finfo(np.float16).max,
                        neginf=np.finfo(np.float16).min,
                    )
                    audio_waveform = np.clip(audio_waveform, -1.0, 1.0)
                    if not np.all(np.isfinite(audio_waveform)):
                        _logger.warning(
                            "Audio waveform contains non-finite values after clipping; returning None."
                        )
                        return None
                    return audio_waveform
                else:
                    _logger.warning(
                        "Griffin-Lim with n_iter=%d produced an empty output.",
                        n_iter,
                    )
            except Exception:
                _logger.warning("Griffin-Lim with n_iter=%d failed", n_iter)
                if n_iter == griffin_lim_iterations[-1]:
                    _logger.warning("Griffin-Lim failed; returning None")
                    return None
                _logger.info("Trying again with more Griffin-Lim iterations...")
        return None

    except Exception:
        _logger.exception("Failed to convert features to audio")
        return None


def predict_audio(model, audio_file):
    import os

    import librosa
    import soundfile as sf

    audio_analysis_backend = _load_audio_analysis_backend()
    array_backend = _load_array_backend()
    model_introspection_backend = _load_model_introspection_backend()

    audio_file = full_path(audio_file)

    if not os.path.exists(audio_file):
        return None

    try:
        (audio_data, sr) = librosa.load(audio_file, sr=32000, mono=True)
        timeline = audio_analysis_backend.get_active_audio_timeline(audio_file)
        _logger.info("Audio shape: %s", audio_data.shape)
        _logger.info("Active audio timeline: %s", timeline)
        predicted_audio = np.zeros_like(audio_data)
        if not timeline:
            _logger.info("Silent timeline: no active audio segments found.")
        for i, (start_time, end_time) in enumerate(timeline):
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)
            active_audio_part_np = audio_data[start_sample:end_sample]
            if active_audio_part_np.size == 0:
                _logger.info(
                    "Segment skipped: empty audio segment from %.2fs to %.2fs",
                    start_time,
                    end_time,
                )
                continue
            active_audio_part_model_input = array_backend.numpy_to_cupy(
                active_audio_part_np
            )
            _logger.info(
                "Predicting segment %d/%d with shape %s",
                i + 1,
                len(timeline),
                active_audio_part_model_input.shape,
            )
            prediction = model.predict(active_audio_part_model_input)
            if model_introspection_backend.is_clusters_model(model):
                _logger.info(
                    "Getting prediction cluster content for segment %d", i + 1
                )
                part_feat = array_backend.cupy_to_numpy(
                    model_introspection_backend.get_cluster_content(
                        model, int(prediction[0])
                    )
                )
            else:
                part_feat = array_backend.cupy_to_numpy(prediction)
            _logger.info(
                "Predicted features shape for segment %d: %s",
                i + 1,
                part_feat.shape,
            )
            part_aud = features_to_audio(part_feat)
            if part_aud is None:
                _logger.warning(
                    "Failed to convert features to audio for segment %d. Skipping.",
                    i + 1,
                )
                continue
            part_length = end_sample - start_sample
            min_len = min(part_aud.shape[0], part_length)
            predicted_audio[start_sample : start_sample + min_len] = part_aud[
                :min_len
            ]
        output_file = tmp("wav")
        sf.write(output_file, predicted_audio, sr)
        _logger.info("Predicted audio saved to: %s", output_file)
        return output_file
    except Exception:
        _logger.exception("Failed to predict audio")
        return None
