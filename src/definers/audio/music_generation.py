from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from definers.constants import MODELS, PROCESSORS
from definers.cuda import device
from definers.system import catch, delete, exist, tmp
from definers.text import random_string

from .dependencies import librosa_module
from .io import read_audio, save_audio


def generate_music(prompt: str, duration_s: float, format_choice: str) -> str:
    import pydub
    from scipy.io.wavfile import write as write_wav

    from definers.ml import init_pretrained_model
    from definers.system.output_paths import managed_output_path

    if MODELS["music"] is None or PROCESSORS["music"] is None:
        init_pretrained_model("music")
    inputs = PROCESSORS["music"](
        text=[prompt],
        padding=True,
        return_tensors="pt",
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
    output_stem = managed_output_path(
        format_choice,
        section="audio",
        stem=f"generated_{random_string()}",
    )
    output_path = save_audio(
        destination_path=output_stem,
        audio_signal=sound,
        sample_rate=32000,
    )
    delete(temp_wav_path)
    return output_path


def extend_audio(
    audio_path: str,
    extend_duration_s: float,
    format_choice: str,
):
    import pydub
    import soundfile as sf

    from definers.ml import init_pretrained_model
    from definers.system.output_paths import managed_output_path

    librosa = librosa_module()

    if MODELS["music"] is None or PROCESSORS["music"] is None:
        init_pretrained_model("music")
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    prompt_duration_s = min(15.0, len(y) / sr)
    prompt_wav = y[-int(prompt_duration_s * sr) :]
    inputs = PROCESSORS["music"](
        audio=prompt_wav,
        sampling_rate=sr,
        return_tensors="pt",
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
    output_stem = managed_output_path(
        format_choice,
        section="audio",
        stem=f"{Path(audio_path).stem}_extended",
    )
    final_output_path = save_audio(
        audio_signal=final_sound,
        destination_path=output_stem,
    )
    delete(temp_extension_path)
    return final_output_path


def audio_to_midi(audio_path: str):
    import madmom
    from basic_pitch.inference import predict

    from definers.system.output_paths import managed_output_path

    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(audio_path)
    bpm = np.median(60 / np.diff(proc(act)))
    model_output, midi_data, note_events = predict(
        audio_path,
        midi_tempo=bpm,
        onset_threshold=0.95,
        frame_threshold=0.25,
        minimum_note_length=80,
        minimum_frequency=60,
        maximum_frequency=4200,
    )
    _ = (model_output, note_events)
    output_path = managed_output_path(
        "mid",
        section="audio",
        stem=f"{Path(audio_path).stem}_{random_string()}",
    )
    midi_data.write(output_path)
    return output_path


def midi_to_audio(midi_path: str, format_choice: str):
    from midi2audio import FluidSynth

    from definers.system import install_audio_effects
    from definers.system.output_paths import managed_output_path

    install_audio_effects()

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
    fluid_synth = FluidSynth(sound_font=soundfont_file)
    temp_wav_path = tmp(".wav")
    fluid_synth.midi_to_audio(midi_path, temp_wav_path)
    sr, sound = read_audio(temp_wav_path)
    output_stem = managed_output_path(
        format_choice,
        section="audio",
        stem=f"{Path(midi_path).stem}_render",
    )
    final_output_path = save_audio(
        audio_signal=sound,
        destination_path=output_stem,
        sample_rate=sr,
    )
    delete(temp_wav_path)
    return final_output_path
