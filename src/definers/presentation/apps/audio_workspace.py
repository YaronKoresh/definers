from __future__ import annotations

AUDIO_FORMAT_CHOICES = ["MP3", "WAV", "FLAC"]

AUDIO_TOOL_MAP = {
    "Mastering Studio": "enhancer",
    "Vocal Finishing": "vocal_finish",
    "Audio Cleanup": "cleanup",
    "Preview & Split": "preview_split",
    "MIDI Tools": "midi_tools",
    "Audio Extender": "audio_extender",
    "Stem Mixer": "stem_mixer",
    "Track Feedback": "feedback",
    "Instrument ID": "instrument_id",
    "Music Clip Generation": "video_gen",
    "Speed & Pitch": "speed",
    "Stem Separation": "stem",
    "Vocal Pitch Shifter": "vps",
    "Voice Lab": "voice_lab",
    "DJ AutoMix": "dj",
    "Music Gen": "music_gen",
    "Voice Gen": "voice_gen",
    "Analysis": "analysis",
    "Speech-to-Text": "stt",
    "Spectrum": "spectrum",
    "Beat Visualizer": "beat_vis",
    "Lyric Video": "lyric_vid",
    "Support Chat": "chatbot",
}

AUDIO_ASSISTANT_RULES = [
    "guide users with the application usage",
    "explain the purpose of each tool in the application",
    "provide simple, step-by-step instructions on how to use the features based on their UI",
]

AUDIO_ASSISTANT_DATA = [
    "The name of the software you help with, is Definers Audio",
    "Definers Audio provides tools for audio transformation, generation, analysis, and presentation workflows",
    "The main AI models used by this workspace include openai/whisper-large-v3, MIT/ast-finetuned-audioset-10-10-0.4593, and facebook/musicgen-small",
    "The supported output formats, are: MP3 (320k), FLAC (16-bit), and WAV (16-bit PCM)",
    "The export process is by clicking on the small down-arrow download button",
    """The complete list of the application's features with usage instructions:
a mastering studio for high-end mastering and repair - upload a mix, choose preset and bass/volume/effects macros, decide whether to use stem-aware mastering, and click 'Master Audio'; the tool can also write a mastering diagnostics report and expose mastered stems;
a vocal finishing workspace with dedicated AutoTune and humanization tools - upload a full song for AutoTune or a vocal take for humanization, adjust the musical controls, and click the relevant action button;
an audio cleanup workspace for silence removal and compact exports - upload a track, choose the target format, and run either 'Remove Silence' or 'Compact Audio';
a preview and split workspace for smart excerpts and chunked exports - create a short preview clip from the most active region of a track or split long files into equal chunks with optional offsets and sample-rate conversion;
audio to midi converter - upload an audio file and click 'Convert to MIDI';
midi to audio converter - upload a MIDI file and click 'Convert to Audio';
an audio extender that uses AI to seamlessly continue a piece of music - upload your audio, use the 'Extend Duration' slider to choose how many seconds to add, and click 'Extend Audio';
a stem mixer that mixes individual instrument tracks (stems) together - upload multiple audio files (e.g., drums.wav, bass.wav). The tool automatically beatmatches them to the first track and mixes them;
a track feedbacks generator that provides an analysis and advice on your mix - upload your track and click 'Get Feedback' for written analysis on its dynamics, stereo width, and frequency balance;
an instrument identifier from an audio file - upload an audio file and click 'Identify Instruments';
a video generator which creates a simple and abstract music visualizer - upload an audio file and click 'Generate Video' to create a video with a pulsing circle that reacts to the music;
a speed & pitch changer which changes the playback speed of a track - upload audio, use the 'Speed Factor' slider (e.g., 1.5x is faster), and check 'Preserve Pitch' for a more natural sound;
a stem separation workspace which can output acapella, karaoke, both vocal and instrumental stems together, or the full mastering-layer split (vocals, drums, bass, other) used by the current mastering pipeline;
a vocal pitch shifter which changes the pitch of only the vocals in a song - upload a song and use the 'Vocal Pitch Shift' slider to raise or lower the vocal pitch in semitones;
a voice cloning and conversion tool for voice manipulation, preserving the melody - upload your training audio files, click 'Train' to create a voice model, then use the 'Convert' tab to apply that voice to a new audio input;
a dj tool which automatically mixes multiple songs together - upload two or more tracks. Choose 'Beatmatched Crossfade' for a smooth, tempo-synced mix and adjust the 'Transition Duration';
an AI music generator which creates original music from a text description - write a description of the music you want (e.g., 'upbeat synthwave'), set the duration, and click 'Generate Music';
an AI voice generator which clones a voice to say anything you type - upload a clean 5-15 second 'Reference Voice' sample, type the 'Text to Speak', and click 'Generate Voice';
a deep audio analysis workspace which combines bpm/key detection with a richer diagnostic summary and downloadable analysis report - upload your audio, optionally narrow the analysis window, and click 'Analyze Audio';
a speech-to-text tool which transcribes speech from an audio file into text - upload an audio file with speech, select the language, and click 'Transcribe Audio'.
a spectrum analyzer which creates a visual graph (spectrogram) of an audio's frequencies - upload an audio file and click 'Generate Spectrum'.
a beat visualizer which creates a video where an image pulses to the music's beat - upload an image and an audio file. Adjust the 'Beat Intensity' slider to control how much the image reacts.
a lyric video creation tool which creates a simple lyric video - upload a song and a background image/video. Then, paste your lyrics into the text box, with each line representing a new phrase on screen.
a support chat (that's you!) which answer questions like 'What is Stem Mixing?' or 'How do I use the Vocal Pitch Shifter?' based on his knowledge-base.""",
]


def get_audio_language_choices(language_codes):
    return sorted(set(language_codes.values()))


def train_voice_lab_model(experiment, inp, lvl):
    from definers.ml import train_model_rvc
    from definers.system import cwd

    with cwd():
        return train_model_rvc(experiment, inp, lvl), lvl + 1


def prepare_audio_workspace():
    from definers.ml import init_pretrained_model
    from definers.presentation.lyric_video_service import init_stable_whisper
    from definers.system import (
        cwd,
        exist,
        install_audio_effects,
        install_ffmpeg,
    )
    from definers.text import set_system_message

    install_audio_effects()
    install_ffmpeg()
    init_stable_whisper()
    init_pretrained_model("tts")

    svc_installed = False
    with cwd():
        if exist("./assets"):
            svc_installed = True

    if not svc_installed:
        init_pretrained_model("svc")

    for model_name in (
        "speech-recognition",
        "audio-classification",
        "music",
        "summary",
        "answer",
        "translate",
    ):
        init_pretrained_model(model_name)

    set_system_message(
        name="Definers Audio Assistant",
        role="the chat assistant for the Definers audio workspace",
        rules=AUDIO_ASSISTANT_RULES,
        data=AUDIO_ASSISTANT_DATA,
        formal=True,
        creative=False,
    )
    return {"svc_installed": svc_installed}
