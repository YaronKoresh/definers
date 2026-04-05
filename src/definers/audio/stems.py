from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np

from definers.constants import MODELS
from definers.logger import init_logger
from definers.system import catch, delete, run, tmp
from definers.text import random_string

from .io import read_audio, save_audio
from .utils import normalize_audio_to_peak

_logger = init_logger()


def _normalize_demucs_option(value: str, *, option_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"Missing {option_name}")
    if any(
        not (character.isalnum() or character in {"_", "-"})
        for character in normalized
    ):
        raise ValueError(f"Invalid {option_name}: {value}")
    return normalized


def separate_stem_layers(
    audio_path: str,
    *,
    model_name: str = "htdemucs_6s",
    shifts: int = 2,
    two_stems: str | None = None,
    output_dir: str | None = None,
) -> tuple[dict[str, str], str]:
    resolved_model_name = _normalize_demucs_option(
        model_name,
        option_name="DEMUCS model name",
    )
    resolved_two_stems = (
        None
        if two_stems is None
        else _normalize_demucs_option(
            two_stems,
            option_name="DEMUCS two-stems value",
        )
    )
    resolved_shifts = max(int(shifts), 1)
    resolved_output_dir = output_dir or tmp(dir=True)
    owns_output_dir = output_dir is None
    command = (
        f'"{sys.executable}" -m demucs.separate -n {resolved_model_name} '
        f"--shifts={resolved_shifts}"
    )
    if resolved_two_stems is not None:
        command += f" --two-stems={resolved_two_stems}"
    command += f' -o "{resolved_output_dir}" "{audio_path}"'

    try:
        run(command)
        separated_dir = (
            Path(resolved_output_dir)
            / resolved_model_name
            / Path(audio_path).stem
        )
        stem_paths = {
            stem_path.stem.lower(): str(stem_path)
            for stem_path in sorted(separated_dir.glob("*.wav"))
        }
        if not stem_paths:
            raise FileNotFoundError("Stem separation failed")
        return stem_paths, str(resolved_output_dir)
    except Exception:
        if owns_output_dir:
            delete(resolved_output_dir)
        raise


def separate_stems(
    audio_path: str,
    separation_type=None,
    format_choice: str = "wav",
):
    output_dir = tmp(dir=True)
    try:
        stem_paths, _resolved_output_dir = separate_stem_layers(
            audio_path,
            model_name="hdemucs_mmi",
            shifts=2,
            two_stems="vocals",
            output_dir=output_dir,
        )
    except Exception:
        delete(output_dir)
        catch("Stem separation failed.")
        return None
    vocals_path = stem_paths.get("vocals")
    accompaniment_path = stem_paths.get("no_vocals")
    if vocals_path is None or accompaniment_path is None:
        delete(output_dir)
        catch("Stem separation failed.")
        return None

    def export_stem(chosen_stem_path, suffix):
        sr, sound = read_audio(chosen_stem_path)
        output_stem = str(
            Path(audio_path).with_name(Path(audio_path).stem + suffix)
        )
        return save_audio(
            audio_signal=sound,
            destination_path=output_stem,
            sample_rate=sr,
            output_format=format_choice,
        )

    if separation_type == "acapella":
        voice = export_stem(vocals_path, "_acapella")
        delete(output_dir)
        return normalize_audio_to_peak(voice)
    if separation_type == "karaoke":
        music = export_stem(accompaniment_path, "_karaoke")
        delete(output_dir)
        return normalize_audio_to_peak(music)

    voice = normalize_audio_to_peak(export_stem(vocals_path, "_acapella"))
    music = normalize_audio_to_peak(export_stem(accompaniment_path, "_karaoke"))
    delete(output_dir)
    return voice, music


def stem_mixer(files, format_choice):
    import pydub
    from scipy.io.wavfile import write as write_wav

    if not files or len(files) < 2:
        catch("Please upload at least two stem files.")
        return None
    processed_stems = []
    target_sr = None
    max_length = 0
    _logger.info("Processing stems for simple mixing")
    for index, file_path in enumerate(files):
        file_obj = Path(file_path)
        _logger.info(
            "Processing file %d/%d: %s",
            index + 1,
            len(files),
            file_obj.name,
        )
        try:
            y, sr = librosa.load(file_path, sr=None)
        except Exception as error:
            catch(f"Could not load file: {file_obj.name}. Error: {error}")
            continue
        if target_sr is None:
            target_sr = sr
        if sr != target_sr:
            _logger.info(
                "Resampling %s from %dHz to %dHz",
                file_obj.name,
                sr,
                target_sr,
            )
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        processed_stems.append(y)
        if len(y) > max_length:
            max_length = len(y)
    if not processed_stems:
        catch("No audio files were successfully processed.")
        return None
    _logger.info("Mixing stems")
    mixed_y = np.zeros(max_length, dtype=np.float32)
    for stem_audio in processed_stems:
        mixed_y[: len(stem_audio)] += stem_audio
    _logger.info("Mixing complete; normalizing volume")
    peak_amplitude = np.max(np.abs(mixed_y))
    if peak_amplitude > 0:
        mixed_y = mixed_y / peak_amplitude * 0.99
    _logger.info("Exporting final mix")
    temp_wav_path = tmp(".wav", keep=False)
    write_wav(temp_wav_path, target_sr, (mixed_y * 32767).astype(np.int16))
    sound = pydub.AudioSegment.from_file(temp_wav_path)
    output_stem = Path(temp_wav_path).with_name(f"stem_mix_{random_string()}")
    output_path = save_audio(
        audio_signal=sound,
        destination_path=output_stem,
        output_format=format_choice,
    )
    delete(temp_wav_path)
    _logger.info("Success! Mix saved to: %s", output_path)
    return output_path


def identify_instruments(audio_path):
    if MODELS["audio-classification"] is None:
        catch("Audio identification model is not available.")
        return None
    predictions = MODELS["audio-classification"](audio_path, top_k=10)
    instrument_list = [
        "guitar",
        "piano",
        "violin",
        "drum",
        "bass",
        "saxophone",
        "trumpet",
        "flute",
        "cello",
        "clarinet",
        "synthesizer",
        "organ",
        "accordion",
        "banjo",
        "harp",
        "voice",
        "speech",
    ]
    detected_instruments = "### Detected Instruments\n\n"
    found = False
    for prediction in predictions:
        label = prediction["label"].lower()
        if any(instrument in label for instrument in instrument_list):
            detected_instruments += f"- **{prediction['label'].title()}** (Score: {prediction['score']:.2f})\n"
            found = True
    if not found:
        detected_instruments += "Could not identify specific instruments with high confidence. Top sound events:\n"
        for prediction in predictions[:3]:
            detected_instruments += f"- {prediction['label'].title()} (Score: {prediction['score']:.2f})\n"
    return detected_instruments
