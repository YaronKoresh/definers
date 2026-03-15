from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
from scipy.io.wavfile import write as write_wav

from definers.constants import MODELS, PROCESSORS, language_codes
from definers.cuda import device
from definers.logger import init_logger
from definers.system import catch, delete, exist, get_ext, run, tmp
from definers.text import random_string

from .analysis import analyze_audio_features
from .io import read_audio, save_audio
from .utils import get_scale_notes, normalize_audio_to_peak, stretch_audio

_logger = init_logger()


def value_to_keys(dictionary: dict, target_value) -> list:
    return [key for (key, value) in dictionary.items() if value == target_value]


def humanize_vocals(audio_path: str, amount: float = 0.5) -> str | None:
    import soundfile as sf

    temp_dir = None
    try:
        if not exist(audio_path):
            return None

        temp_dir = tmp(dir=True)
        (y, sr) = librosa.load(audio_path, sr=None, mono=True)
        (f0, voiced_flag, _) = librosa.pyin(
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
            for i in range(len(f0)):
                if voiced_flag[i] and f0[i] > 0:
                    cents_dev = np.random.normal(0, scale=deviation_scale)
                    cents_dev = np.clip(
                        cents_dev, -max_deviation_cents, max_deviation_cents
                    )
                    target_f0[i] = f0[i] * 2 ** (cents_dev / 1200)

        freq_map_path = os.path.join(temp_dir, "freqmap.txt")
        with open(freq_map_path, "w") as f:
            for i in range(len(f0)):
                if voiced_flag[i] and f0[i] > 0 and (target_f0[i] > 0):
                    sample_num = i * 1024
                    ratio = target_f0[i] / f0[i]
                    f.write(f"{sample_num} {ratio:.6f}\n")

        if os.path.getsize(freq_map_path) > 0:
            _logger.info("Applying pitch variations with Rubberband")
            temp_input_wav = os.path.join(temp_dir, "input.wav")
            temp_output_wav = os.path.join(temp_dir, "output.wav")
            sf.write(temp_input_wav, y, sr)
            run(
                f"rubberband --formant --freqmap {freq_map_path} {temp_input_wav} {temp_output_wav}"
            )
            (y_humanized, _) = librosa.load(temp_output_wav, sr=sr)
            _logger.info("Humanization complete")
        else:
            _logger.warning("No voiced frames detected; skipping humanization.")
            y_humanized = np.copy(y)

        input_p = Path(audio_path)
        output_path = input_p.parent / f"{input_p.stem}_humanized.wav"
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
    (vocal, _) = separate_stems(audio_path)
    lang_code = value_to_keys(language_codes, language)[0]
    lang_code = lang_code.replace("iw", "he")
    return MODELS["speech-recognition"](
        vocal, generate_kwargs={"language": lang_code}, return_timestamps=True
    )["text"]


def generate_voice(
    text: str, reference_audio: str, format_choice: str
) -> str | None:
    import pydub
    import soundfile as sf

    from definers.ml import init_pretrained_model

    if not MODELS["tts"]:
        init_pretrained_model("tts")

    try:
        temp_wav_path = tmp("wav", False)
        wav = MODELS["tts"].generate(
            text=text, audio_prompt_path=reference_audio
        )
        wav = normalize_audio_to_peak(wav)
        sf.write(temp_wav_path, wav, 24000)
        sound = pydub.AudioSegment.from_file(temp_wav_path)
        output_stem = tmp(keep=False).replace(".data", "")
        final_output_path = save_audio(
            destination_path=output_stem,
            audio_signal=sound,
            sample_rate=24000,
            output_format=format_choice,
        )
        return final_output_path
    except Exception as e:
        catch(f"Generation failed: {e}")


def generate_music(prompt: str, duration_s: float, format_choice: str) -> str:
    import pydub
    from scipy.io.wavfile import write as write_wav

    inputs = PROCESSORS["music"](
        text=[prompt], padding=True, return_tensors="pt"
    ).to(device())
    max_new_tokens = int(duration_s * 50)
    audio_values = MODELS["music"].generate(
        **inputs,
        do_sample=True,
        guidance_scale=3,
        max_new_tokens=max_new_tokens,
    )
    sampling_rate = MODELS["music"].config.audio_encoder.sampling_rate
    wav_output = audio_values[0, 0].cpu().numpy()
    temp_wav_path = tmp("wav", keep=False)
    write_wav(temp_wav_path, rate=sampling_rate, data=wav_output)
    sound = pydub.AudioSegment.from_file(temp_wav_path)
    output_stem = Path(temp_wav_path).with_name(f"generated_{random_string()}")
    output_path = save_audio(
        destination_path=output_stem,
        audio_signal=sound,
        sample_rate=32000,
        output_format=format_choice,
    )
    delete(temp_wav_path)
    return output_path


def change_audio_speed(
    audio_path: str,
    speed_factor: float,
    preserve_pitch: bool,
    format_choice: str,
):
    import pydub

    sound_out = None
    if preserve_pitch:
        audio_path_out = tmp(Path(audio_path).suffix)
        stretched = stretch_audio(audio_path, audio_path_out, speed_factor)
        if stretched:
            sound_out = pydub.AudioSegment.from_file(audio_path_out)
            delete(audio_path_out)
        else:
            catch("Failed to stretch audio while preserving pitch.")
            return None
    else:
        sound = pydub.AudioSegment.from_file(audio_path)
        new_frame_rate = int(sound.frame_rate * speed_factor)
        sound_out = sound._spawn(
            sound.raw_data, overrides={"frame_rate": new_frame_rate}
        ).set_frame_rate(sound.frame_rate)
    if sound_out:
        output_stem = str(
            Path(audio_path).with_name(
                f"{Path(audio_path).stem}_speed_{speed_factor}x"
            )
        )
        return save_audio(
            destination_path=output_stem,
            audio_signal=sound_out,
            sample_rate=24000,
            output_format=format_choice,
        )
    else:
        catch("Could not process audio speed change.")
        return None


def separate_stems(
    audio_path: str, separation_type=None, format_choice: str = "wav"
):
    output_dir = tmp(dir=True)
    run(
        f'"{sys.executable}" -m demucs.separate -n hdemucs_mmi --shifts=2 --two-stems=vocals -o "{output_dir}" "{audio_path}"'
    )
    separated_dir = Path(output_dir) / "hdemucs_mmi" / Path(audio_path).stem
    vocals_path = separated_dir / "vocals.wav"
    accompaniment_path = separated_dir / "no_vocals.wav"
    if not vocals_path.exists() or not accompaniment_path.exists():
        delete(output_dir)
        catch("Stem separation failed.")
        return None

    def _export_stem(chosen_stem_path, suffix):
        sr, sound = read_audio(chosen_stem_path)
        output_stem = str(
            Path(audio_path).with_name(Path(audio_path).stem + suffix)
        )
        final_output_path = save_audio(
            audio_signal=sound,
            destination_path=output_stem,
            sample_rate=sr,
            output_format=format_choice,
        )
        return final_output_path

    if separation_type == "acapella":
        voice = _export_stem(vocals_path, "_acapella")
        delete(output_dir)
        return normalize_audio_to_peak(voice)
    if separation_type == "karaoke":
        music = _export_stem(accompaniment_path, "_karaoke")
        delete(output_dir)
        return normalize_audio_to_peak(music)

    (voice, music) = (
        normalize_audio_to_peak(_export_stem(vocals_path, "_acapella")),
        normalize_audio_to_peak(_export_stem(accompaniment_path, "_karaoke")),
    )
    delete(output_dir)
    return (voice, music)


def pitch_shift_vocals(
    audio_path: str,
    pitch_shift,
    format_choice: str = "wav",
    seperated: bool = False,
):
    import pydub
    import soundfile as sf

    if seperated:
        (y_vocals, sr) = librosa.load(str(audio_path), sr=None)
        y_shifted = librosa.effects.pitch_shift(
            y=y_vocals, sr=sr, n_steps=float(pitch_shift)
        )
        output_path = tmp(format_choice, keep=False)
        sf.write(output_path, y_shifted, sr)
        return output_path

    (vocals_path, instrumental_path) = separate_stems(audio_path)
    vocals_path = Path(vocals_path)
    instrumental_path = Path(instrumental_path)
    if not vocals_path.exists() or not instrumental_path.exists():
        delete(str(vocals_path.parent))
        catch("Vocal separation failed.")
        return None
    (y_vocals, sr) = librosa.load(str(vocals_path), sr=None)
    y_shifted = librosa.effects.pitch_shift(
        y=y_vocals, sr=sr, n_steps=float(pitch_shift)
    )
    shifted_vocals_path = tmp("wav", keep=False)
    sf.write(shifted_vocals_path, y_shifted, sr)
    instrumental = pydub.AudioSegment.from_file(instrumental_path)
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
    delete(str(vocals_path.parent))
    delete(shifted_vocals_path)
    return final_output_path


def create_spectrum_visualization(audio_path: str) -> str | None:
    import matplotlib.pyplot as plt

    try:
        (y, sr) = librosa.load(audio_path, sr=None)
        n_fft = 8192
        hop_length = 512
        stft_result = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft_result)
        avg_magnitude = np.mean(magnitude, axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        magnitude_db = librosa.amplitude_to_db(avg_magnitude, ref=np.max)
        (fig, ax) = plt.subplots(figsize=(8, 5), facecolor="#f0f0f0")
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
            peak_idx = np.argmax(magnitude_db[audible_mask])
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


def stem_mixer(files, format_choice):
    import madmom
    import pydub
    import soundfile as sf

    if not files or len(files) < 2:
        catch("Please upload at least two stem files.")
        return None
    processed_stems = []
    target_sr = None
    max_length = 0
    _logger.info("Processing stems for simple mixing")
    for i, _file in enumerate(files):
        file_obj = Path(_file)
        _logger.info(
            "Processing file %d/%d: %s", i + 1, len(files), file_obj.name
        )
        try:
            (y, sr) = librosa.load(_file, sr=None)
        except Exception as e:
            catch(f"Could not load file: {file_obj.name}. Error: {e}")
            continue
        if target_sr is None:
            target_sr = sr
        if sr != target_sr:
            _logger.info(
                "Resampling %s from %dHz to %dHz", file_obj.name, sr, target_sr
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
    for p in predictions:
        label = p["label"].lower()
        if any(instrument in label for instrument in instrument_list):
            detected_instruments += (
                f"- **{p['label'].title()}** (Score: {p['score']:.2f})\n"
            )
            found = True
    if not found:
        detected_instruments += "Could not identify specific instruments with high confidence. Top sound events:\n"
        for p in predictions[:3]:
            detected_instruments += (
                f"- {p['label'].title()} (Score: {p['score']:.2f})\n"
            )
    return detected_instruments


def extend_audio(audio_path: str, extend_duration_s: float, format_choice: str):
    import pydub
    import soundfile as sf

    if MODELS["music"] is None or PROCESSORS["music"] is None:
        catch("MusicGen model is not available for audio extension.")
        return None
    (y, sr) = librosa.load(audio_path, sr=None, mono=True)
    prompt_duration_s = min(15.0, len(y) / sr)
    prompt_wav = y[-int(prompt_duration_s * sr) :]
    inputs = PROCESSORS["music"](
        audio=prompt_wav, sampling_rate=sr, return_tensors="pt"
    ).to(device())
    total_duration_s = prompt_duration_s + extend_duration_s
    max_new_tokens = int(total_duration_s * 50)
    generated_audio_values = MODELS["music"].generate(
        **inputs,
        do_sample=True,
        guidance_scale=3,
        max_new_tokens=max_new_tokens,
    )
    generated_wav = generated_audio_values[0, 0].cpu().numpy()
    extension_start_sample = int(
        prompt_duration_s * MODELS["music"].config.audio_encoder.sampling_rate
    )
    extension_wav = generated_wav[extension_start_sample:]
    temp_extension_path = tmp(".wav")
    sf.write(
        temp_extension_path,
        extension_wav,
        MODELS["music"].config.audio_encoder.sampling_rate,
    )
    original_sound = pydub.AudioSegment.from_file(audio_path)
    extension_sound = pydub.AudioSegment.from_file(temp_extension_path)
    if original_sound.channels != extension_sound.channels:
        extension_sound = extension_sound.set_channels(original_sound.channels)
    final_sound = original_sound + extension_sound
    output_stem = str(
        Path(audio_path).with_name(f"{Path(audio_path).stem}_extended")
    )
    final_output_path = save_audio(
        audio_signal=final_sound,
        destination_path=output_stem,
        output_format=format_choice,
    )
    delete(temp_extension_path)
    return final_output_path


def audio_to_midi(audio_path: str):
    import madmom
    from basic_pitch.inference import predict

    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(audio_path)
    bpm = np.median(60 / np.diff(proc(act)))
    (model_output, midi_data, note_events) = predict(
        audio_path,
        midi_tempo=bpm,
        onset_threshold=0.95,
        frame_threshold=0.25,
        minimum_note_length=80,
        minimum_frequency=60,
        maximum_frequency=4200,
    )
    name = random_string() + ".mid"
    midi_data.write(f"./{name}")
    return name


def midi_to_audio(midi_path: str, format_choice: str):
    import pydub
    from midi2audio import FluidSynth

    soundfont_paths = [
        os.path.join(
            os.path.expanduser("~"),
            "app_dependencies",
            "soundfonts",
            "VintageDreamsWaves-v2.sf3",
        ),
        "/usr/share/sounds/sf2/VintageDreamsWaves-v2.sf3",
        "C:/Windows/System32/drivers/gm.dls",
    ]
    soundfont_file = None
    for path in soundfont_paths:
        if os.path.exists(path):
            soundfont_file = path
            break
    if soundfont_file is None:
        catch(
            "SoundFont file not found. MIDI to Audio conversion cannot proceed. Please re-run the dependency installer."
        )
        return None
    fs = FluidSynth(sound_font=soundfont_file)
    temp_wav_path = tmp(".wav")
    fs.midi_to_audio(midi_path, temp_wav_path)
    sr, sound = read_audio(temp_wav_path)
    output_stem = str(
        Path(midi_path).with_name(f"{Path(midi_path).stem}_render")
    )
    final_output_path = save_audio(
        audio_signal=sound,
        destination_path=output_stem,
        sample_rate=sr,
        output_format=format_choice,
    )
    delete(temp_wav_path)
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
    import librosa
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
        (detected_key, detected_mode, detected_bpm) = analyze_audio_features(
            audio_path, txt=False
        )
        if not detected_key:
            catch("Could not determine song key. Aborting.")
            return None
        (vocals_path, instrumental_path) = separate_stems(audio_path)
        if not vocals_path or not instrumental_path:
            catch("Vocal separation failed.")
            return None
        temp_files.extend([vocals_path, instrumental_path])
        (y_vocals, sr) = librosa.load(vocals_path, sr=None, mono=True)
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
                beat_interval = np.mean(np.diff(beat_times))
                beat_interval / quantize_grid_strength
                quantized_beat_times: list[float] = []
                for i in range(len(beat_times) - 1):
                    quantized_beat_times.extend(
                        np.linspace(
                            beat_times[i],
                            beat_times[i + 1],
                            quantize_grid_strength,
                            endpoint=False,
                        )
                    )
                quantized_beat_times.append(beat_times[-1])
                beat_times = np.array(sorted(quantized_beat_times))
            onsets = librosa.onset.onset_detect(
                y=y_vocals, sr=sr, hop_length=hop_length, units="time"
            )
            if len(onsets) > 1 and len(beat_times) > 0:
                time_map_data: list[str] = []
                for onset_time in onsets:
                    closest_beat_index = np.argmin(
                        np.abs(beat_times - onset_time)
                    )
                    target_time = beat_times[closest_beat_index]
                    time_map_data.append(f"{onset_time:.6f} {target_time:.6f}")
                time_map_path = tmp(".txt")
                temp_files.append(time_map_path)
                with open(time_map_path, "w") as f:
                    f.write("\n".join(time_map_data))
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
                    (y_vocals, sr) = librosa.load(quantized_vocals_path, sr=sr)
                    processed_vocals_path = quantized_vocals_path
                    temp_files.append(quantized_vocals_path)
        allowed_notes_midi = get_scale_notes(
            key=detected_key, scale=detected_mode
        )
        (f0, voiced_flag, _) = librosa.pyin(
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
                        temp_voiced_f0, kernel_size=smoothing_window_size
                    )
            target_f0[voiced_mask] = temp_voiced_f0
        freq_map_path = tmp(".txt")
        temp_files.append(freq_map_path)
        with open(freq_map_path, "w") as f:
            ratios = target_f0 / f0
            ratios[~voiced_flag | (f0 == 0)] = 1.0
            for i in range(len(ratios)):
                sample_num = i * hop_length
                f.write(f"{sample_num} {ratios[i]:.6f}\n")
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
        output_format = get_ext(output_path)
        combined.export(output_path, format=output_format)
        normalize_audio_to_peak(output_path)
        return output_path
    finally:
        for path in temp_files:
            delete(path)
