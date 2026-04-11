from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from definers.runtime_numpy import get_numpy_module

np = get_numpy_module()

from .dependencies import librosa_module

DEFAULT_TTS_MODEL_NAME = "facebook/mms-tts-eng"


def _to_mono_audio(audio_signal: np.ndarray) -> np.ndarray:
    normalized_signal = np.asarray(audio_signal, dtype=np.float32)
    if normalized_signal.ndim == 0:
        return normalized_signal.reshape(1)
    if normalized_signal.ndim == 1:
        return normalized_signal
    return np.mean(normalized_signal, axis=0, dtype=np.float32)


def _normalize_peak(
    audio_signal: np.ndarray, peak_limit: float = 0.98
) -> np.ndarray:
    normalized_signal = np.nan_to_num(
        _to_mono_audio(audio_signal),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    max_amplitude = (
        float(np.max(np.abs(normalized_signal)))
        if normalized_signal.size
        else 0.0
    )
    if max_amplitude <= 0.0:
        return normalized_signal.astype(np.float32, copy=False)
    scale = max(max_amplitude / peak_limit, 1.0)
    return (normalized_signal / scale).astype(np.float32, copy=False)


def _rms(audio_signal: np.ndarray) -> float:
    normalized_signal = _to_mono_audio(audio_signal).astype(np.float64)
    if normalized_signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(normalized_signal**2)))


def _estimate_pitch(audio_signal: np.ndarray, sample_rate: int) -> float:
    librosa = librosa_module()
    normalized_signal = _to_mono_audio(audio_signal)
    if normalized_signal.size < 2048 or sample_rate <= 0:
        return 0.0
    try:
        pitch_values = librosa.yin(
            normalized_signal,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sample_rate,
        )
    except Exception:
        return 0.0
    finite_pitch_values = pitch_values[np.isfinite(pitch_values)]
    if finite_pitch_values.size == 0:
        return 0.0
    return float(np.median(finite_pitch_values))


def _apply_reference_style(
    audio_signal: np.ndarray,
    sample_rate: int,
    reference_audio_path: str | None,
) -> np.ndarray:
    librosa = librosa_module()
    if not reference_audio_path:
        return _normalize_peak(audio_signal)
    try:
        reference_audio, _ = librosa.load(
            reference_audio_path,
            sr=sample_rate,
            mono=True,
        )
    except Exception:
        return _normalize_peak(audio_signal)

    conditioned_audio = _to_mono_audio(audio_signal)
    reference_pitch = _estimate_pitch(reference_audio, sample_rate)
    generated_pitch = _estimate_pitch(conditioned_audio, sample_rate)
    if reference_pitch > 0.0 and generated_pitch > 0.0:
        pitch_shift_steps = float(
            np.clip(
                12.0 * math.log2(reference_pitch / generated_pitch),
                -4.0,
                4.0,
            )
        )
        if abs(pitch_shift_steps) >= 0.05:
            try:
                conditioned_audio = librosa.effects.pitch_shift(
                    conditioned_audio,
                    sr=sample_rate,
                    n_steps=pitch_shift_steps,
                )
            except Exception:
                pass

    reference_rms = _rms(reference_audio)
    generated_rms = _rms(conditioned_audio)
    if reference_rms > 0.0 and generated_rms > 0.0:
        conditioned_audio = conditioned_audio * (reference_rms / generated_rms)

    return _normalize_peak(conditioned_audio)


@dataclass(slots=True)
class LocalTextToSpeech:
    model: Any
    tokenizer: Any
    sample_rate: int
    device_name: str

    @classmethod
    def from_pretrained(
        cls,
        device_name: str = "cpu",
        model_name: str = DEFAULT_TTS_MODEL_NAME,
    ) -> LocalTextToSpeech:
        from transformers import AutoTokenizer, VitsModel

        from definers.model_installation import hf_snapshot_download

        local_model_path = hf_snapshot_download(
            repo_id=model_name,
            item_label=model_name,
            detail="Downloading text-to-speech model source files.",
        )
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        model = VitsModel.from_pretrained(local_model_path).to(device_name)
        model.eval()
        sample_rate = int(getattr(model.config, "sampling_rate", 16000))
        return cls(
            model=model,
            tokenizer=tokenizer,
            sample_rate=sample_rate,
            device_name=device_name,
        )

    def generate(
        self,
        *,
        text: str,
        reference_audio_path: str | None = None,
    ) -> tuple[np.ndarray, int]:
        import torch

        normalized_text = " ".join(str(text).split())
        if not normalized_text:
            raise ValueError("Text is required for speech synthesis")

        encoded_input = self.tokenizer(normalized_text, return_tensors="pt")
        encoded_input = {
            name: value.to(self.device_name)
            for name, value in encoded_input.items()
        }
        with torch.no_grad():
            waveform = self.model(**encoded_input).waveform
        audio_signal = np.asarray(
            waveform.squeeze().cpu().numpy(),
            dtype=np.float32,
        )
        if audio_signal.size == 0:
            raise ValueError("Speech synthesis returned empty audio")
        audio_signal = _apply_reference_style(
            audio_signal,
            self.sample_rate,
            reference_audio_path,
        )
        return audio_signal, self.sample_rate
