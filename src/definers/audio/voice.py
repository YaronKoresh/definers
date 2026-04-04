from __future__ import annotations

import os
from pathlib import Path

import librosa
import numpy as np

from definers.constants import MODELS, language_codes
from definers.logger import init_logger
from definers.system import catch, delete, exist, run, tmp

from .analysis import analyze_audio_features
from .io import save_audio
from .utils import get_scale_notes, normalize_audio_to_peak

_logger = init_logger()


def _separate_stems(*args, **kwargs):
    from .stems import separate_stems

    return separate_stems(*args, **kwargs)


def value_to_keys(dictionary: dict, target_value) -> list:
    return [key for (key, value) in dictionary.items() if value == target_value]


def humanize_vocals(audio_path: str, amount: float = 0.5) -> str | None:
    import soundfile as sf

    temp_dir = None
    try:
        if not exist(audio_path):
            return None

        temp_dir = tmp(dir=True)
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            frame_length=2048,
            hop_length=1024,
        )
        f0 = np.nan_to_num(f0)

        target_f0 = np.copy(f0)
        if amount > 0:
            deviation_scale = amount * 5.0
            max_deviation_cents = 20.0
            for index in range(len(f0)):
                if voiced_flag[index] and f0[index] > 0:
                    cents_dev = np.random.normal(0, scale=deviation_scale)
                    cents_dev = np.clip(
                        cents_dev,
                        -max_deviation_cents,
                        max_deviation_cents,
                    )
                    target_f0[index] = f0[index] * 2 ** (cents_dev / 1200)

        freq_map_path = os.path.join(temp_dir, "freqmap.txt")
        with open(freq_map_path, "w") as handle:
            for index in range(len(f0)):
                if (
                    voiced_flag[index]
                    and f0[index] > 0
                    and target_f0[index] > 0
                ):
                    sample_num = index * 1024
                    ratio = target_f0[index] / f0[index]
                    handle.write(f"{sample_num} {ratio:.6f}\n")

        if os.path.getsize(freq_map_path) > 0:
            _logger.info("Applying pitch variations with Rubberband")
            temp_input_wav = os.path.join(temp_dir, "input.wav")
            temp_output_wav = os.path.join(temp_dir, "output.wav")
            sf.write(temp_input_wav, y, sr)
            run(
                f"rubberband --formant --freqmap {freq_map_path} {temp_input_wav} {temp_output_wav}"
            )
            y_humanized, _ = librosa.load(temp_output_wav, sr=sr)
            _logger.info("Humanization complete")
        else:
            _logger.warning("No voiced frames detected; skipping humanization.")
            y_humanized = np.copy(y)

        input_path = Path(audio_path)
        output_path = input_path.parent / f"{input_path.stem}_humanized.wav"
        sf.write(str(output_path), y_humanized, sr)
        _logger.info("Humanized audio saved to: %s", output_path)
        return str(output_path)
    except Exception:
        _logger.exception("An error occurred during humanize_vocals")
        return None
    finally:
        _logger.info("Cleaning up temporary files")
        if temp_dir is not None:
            delete(temp_dir)


def transcribe_audio(audio_path: str, language: str) -> str | None:
    from definers.ml import init_pretrained_model

    if MODELS["speech-recognition"] is None:
        init_pretrained_model("speech-recognition")

    audio_path = normalize_audio_to_peak(audio_path)
    vocal, _ = _separate_stems(audio_path)
    lang_code = value_to_keys(language_codes, language)[0]
    lang_code = lang_code.replace("iw", "he")
    return MODELS["speech-recognition"](
        vocal,
        generate_kwargs={"language": lang_code},
        return_timestamps=True,
    )["text"]


def generate_voice(
    text: str,
    reference_audio: str,
    format_choice: str,
) -> str | None:
    import pydub
    import soundfile as sf

    from definers.ml import init_pretrained_model

    if not MODELS["tts"]:
        init_pretrained_model("tts")

    try:
        temp_wav_path = tmp("wav", False)
        wav = MODELS["tts"].generate(
            text=text,
            audio_prompt_path=reference_audio,
        )
        wav = normalize_audio_to_peak(wav)
        sf.write(temp_wav_path, wav, 24000)
        sound = pydub.AudioSegment.from_file(temp_wav_path)
        output_stem = tmp(keep=False).replace(".data", "")
        return save_audio(
            destination_path=output_stem,
            audio_signal=sound,
            sample_rate=24000,
            output_format=format_choice,
        )
    except Exception as error:
        catch(f"Generation failed: {error}")
        return None


def pitch_shift_vocals(
    audio_path: str,
    pitch_shift,
    format_choice: str = "wav",
    seperated: bool = False,
):
    import pydub
    import soundfile as sf

    if seperated:
        y_vocals, sr = librosa.load(str(audio_path), sr=None)
        y_shifted = librosa.effects.pitch_shift(
            y=y_vocals,
            sr=sr,
            n_steps=float(pitch_shift),
        )
        output_path = tmp(format_choice, keep=False)
        sf.write(output_path, y_shifted, sr)
        return output_path

    vocals_path, instrumental_path = _separate_stems(audio_path)
    vocals_file = Path(vocals_path)
    instrumental_file = Path(instrumental_path)
    if not vocals_file.exists() or not instrumental_file.exists():
        delete(str(vocals_file.parent))
        catch("Vocal separation failed.")
        return None

    y_vocals, sr = librosa.load(str(vocals_file), sr=None)
    y_shifted = librosa.effects.pitch_shift(
        y=y_vocals,
        sr=sr,
        n_steps=float(pitch_shift),
    )
    shifted_vocals_path = tmp("wav", keep=False)
    sf.write(shifted_vocals_path, y_shifted, sr)
    instrumental = pydub.AudioSegment.from_file(instrumental_file)
    shifted_vocals = pydub.AudioSegment.from_file(shifted_vocals_path)
    combined = instrumental.overlay(shifted_vocals)
    output_stem = str(
        Path(audio_path).with_name(
            f"{Path(audio_path).stem}_vocal_pitch_shifted"
        )
    )
    final_output_path = save_audio(
        audio_signal=combined,
        destination_path=output_stem,
        output_format=format_choice,
    )
    delete(str(vocals_file.parent))
    delete(shifted_vocals_path)
    return final_output_path


def autotune_song(
    audio_path: str,
    output_path: str | None = None,
    strength: float = 0.7,
    correct_timing: bool = True,
    quantize_grid_strength: int = 16,
    tolerance_cents: int = 15,
    attack_smoothing_ms: float = 0.1,
) -> str | None:
    import madmom
    import pydub
    import soundfile as sf
    from scipy.signal import medfilt

    if output_path is None:
        output_path = tmp("wav", keep=False)
    audio_path = normalize_audio_to_peak(audio_path)
    if not exist(audio_path):
        catch("Input audio file not found.")
        return None

    temp_files: list[str] = []
    try:
        detected_key, detected_mode, detected_bpm = analyze_audio_features(
            audio_path,
            txt=False,
        )
        _ = detected_bpm
        if not detected_key:
            catch("Could not determine song key. Aborting.")
            return None
        vocals_path, instrumental_path = _separate_stems(audio_path)
        if not vocals_path or not instrumental_path:
            catch("Vocal separation failed.")
            return None
        temp_files.extend([vocals_path, instrumental_path])
        y_vocals, sr = librosa.load(vocals_path, sr=None, mono=True)
        n_fft = 16384
        hop_length = 8192
        processed_vocals_path = vocals_path
        if correct_timing:
            beat_proc = madmom.features.beats.RNNBeatProcessor()
            beat_act = beat_proc(instrumental_path)
            beat_times = madmom.features.beats.BeatTrackingProcessor(fps=100)(
                beat_act
            )
            if quantize_grid_strength > 1 and len(beat_times) > 1:
                quantized_beat_times: list[float] = []
                for index in range(len(beat_times) - 1):
                    quantized_beat_times.extend(
                        np.linspace(
                            beat_times[index],
                            beat_times[index + 1],
                            quantize_grid_strength,
                            endpoint=False,
                        )
                    )
                quantized_beat_times.append(beat_times[-1])
                beat_times = np.array(sorted(quantized_beat_times))
            onsets = librosa.onset.onset_detect(
                y=y_vocals,
                sr=sr,
                hop_length=hop_length,
                units="time",
            )
            if len(onsets) > 1 and len(beat_times) > 0:
                time_map_data: list[str] = []
                for onset_time in onsets:
                    closest_beat_index = int(
                        np.argmin(np.abs(beat_times - onset_time))
                    )
                    target_time = beat_times[closest_beat_index]
                    time_map_data.append(f"{onset_time:.6f} {target_time:.6f}")
                time_map_path = tmp(".txt")
                temp_files.append(time_map_path)
                with open(time_map_path, "w") as handle:
                    handle.write("\n".join(time_map_data))
                quantized_vocals_path = tmp(".wav")
                command = [
                    "rubberband",
                    "--formant",
                    "--freqmap",
                    time_map_path,
                    vocals_path,
                    quantized_vocals_path,
                ]
                run(command)
                if exist(quantized_vocals_path):
                    y_vocals, sr = librosa.load(quantized_vocals_path, sr=sr)
                    processed_vocals_path = quantized_vocals_path
                    temp_files.append(quantized_vocals_path)
        allowed_notes_midi = get_scale_notes(
            key=detected_key,
            scale=detected_mode,
        )
        f0, voiced_flag, _ = librosa.pyin(
            y_vocals,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            frame_length=n_fft,
            hop_length=hop_length,
        )
        f0 = np.nan_to_num(f0, nan=0.0)
        target_f0 = np.copy(f0)
        voiced_mask = voiced_flag & (f0 > 0)
        if np.any(voiced_mask):
            voiced_f0 = f0[voiced_mask]
            voiced_midi = librosa.hz_to_midi(voiced_f0)
            note_diffs = np.abs(allowed_notes_midi.reshape(-1, 1) - voiced_midi)
            closest_note_indices = np.argmin(note_diffs, axis=0)
            target_midi = allowed_notes_midi[closest_note_indices]
            cents_deviation = np.abs(voiced_midi - target_midi) * 100
            correction_mask = cents_deviation > tolerance_cents
            ideal_f0 = librosa.midi_to_hz(target_midi[correction_mask])
            original_f0_to_correct = voiced_f0[correction_mask]
            corrected_f0_subset = (
                original_f0_to_correct
                + (ideal_f0 - original_f0_to_correct) * strength
            )
            temp_voiced_f0 = np.copy(voiced_f0)
            temp_voiced_f0[correction_mask] = corrected_f0_subset
            if attack_smoothing_ms > 0:
                smoothing_window_size = int(
                    sr / hop_length * (attack_smoothing_ms / 1000.0)
                )
                if smoothing_window_size % 2 == 0:
                    smoothing_window_size += 1
                if smoothing_window_size > 1:
                    temp_voiced_f0 = medfilt(
                        temp_voiced_f0,
                        kernel_size=smoothing_window_size,
                    )
            target_f0[voiced_mask] = temp_voiced_f0
        freq_map_path = tmp(".txt")
        temp_files.append(freq_map_path)
        with open(freq_map_path, "w") as handle:
            ratios = target_f0 / f0
            ratios[~voiced_flag | (f0 == 0)] = 1.0
            for index in range(len(ratios)):
                sample_num = index * hop_length
                handle.write(f"{sample_num} {ratios[index]:.6f}\n")
        tuned_vocals_path = tmp(".wav")
        command = [
            "rubberband",
            "--formant",
            "--freqmap",
            freq_map_path,
            processed_vocals_path,
            tuned_vocals_path,
        ]
        run(command)
        if not exist(tuned_vocals_path):
            catch("Pitch correction with rubberband failed.")
            return None
        temp_files.append(tuned_vocals_path)
        instrumental_audio = pydub.AudioSegment.from_file(instrumental_path)
        tuned_vocals_audio = pydub.AudioSegment.from_file(tuned_vocals_path)
        tuned_vocals_audio = tuned_vocals_audio.set_frame_rate(
            instrumental_audio.frame_rate
        )
        combined = instrumental_audio.overlay(tuned_vocals_audio)
        output_format = Path(output_path).suffix.strip(".")
        combined.export(output_path, format=output_format)
        normalize_audio_to_peak(output_path)
        return output_path
    finally:
        for path in temp_files:
            delete(path)
