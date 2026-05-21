from __future__ import annotations

import inspect
import json
from pathlib import Path
from shutil import copy2

from definers.ui.gui_shared import (
    bind_progress_click,
    init_chat,
    init_output_folder_controls,
    init_progress_tracker,
    init_status_card,
    launch_blocks,
    progress_update,
    qt_app,
    status_card_markdown,
)
from definers.ui.job_state import (
    create_job_dir,
    existing_path,
    manifest_markdown,
    read_manifest,
    scan_file_map,
    write_manifest,
)

AUDIO_FORMAT_CHOICES = ["MP3", "WAV", "FLAC"]

AUDIO_TOOL_MAP = {
    "Mastering Studio": "enhancer",
    "Vocal Finishing": "vocal_finish",
    "Audio Cleanup": "cleanup",
    "Preview and Split": "preview_split",
    "MIDI Tools": "midi_tools",
    "Audio Extender": "audio_extender",
    "Stem Mixer": "stem_mixer",
    "Track Feedback": "feedback",
    "Instrument ID": "instrument_id",
    "Music Clip Generation": "video_gen",
    "Speed and Pitch": "speed",
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
a speed and pitch changer which changes the playback speed of a track - upload audio, use the 'Speed Factor' slider (e.g., 1.5x is faster), and check 'Preserve Pitch' for a more natural sound;
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
    from definers.system import cwd, exist
    from definers.text import set_system_message

    svc_installed = False
    with cwd():
        if exist("./assets"):
            svc_installed = True

    set_system_message(
        name="Definers Audio Assistant",
        role="the chat assistant for the Definers audio workspace",
        rules=AUDIO_ASSISTANT_RULES,
        data=AUDIO_ASSISTANT_DATA,
        formal=True,
        creative=False,
    )
    return {"svc_installed": svc_installed}


MASTERING_PROFILE_CHOICES = (
    "Auto Analyze",
    "Balanced",
    "EDM",
    "Vocal Focus",
    "Custom Macro Blend",
)

STEM_MODEL_STRATEGY_CHOICES = (
    "Automatic mastering stack",
    "Demucs fine-tuned 4-stem",
    "HDemucs MMI 4-stem",
    "Custom checkpoint override",
)

MASTERING_STEM_PREVIEW_ORDER = (
    "vocals",
    "drums",
    "bass",
    "other",
    "guitar",
    "piano",
)

STEM_GLUE_REVERB_DEFAULT = 5.0
STEM_DRUM_EDGE_DEFAULT = 5.0
STEM_VOCAL_PULLBACK_DB_DEFAULT = 0.0

_AUDIO_JOB_SUFFIXES = (
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
)

_MASTERING_PROFILE_METADATA = {
    "auto": {
        "label": "Auto Analyze",
        "description": "Analyzes the mix first and chooses the most appropriate mastering preset automatically.",
        "macro_note": "Bass, Volume, and Effects are locked in Auto Analyze so the engine does not receive conflicting instructions.",
    },
    "balanced": {
        "label": "Balanced",
        "description": "Uses the neutral all-round mastering profile for mixed-genre material.",
        "macro_note": "Bass, Volume, and Effects are locked to the Balanced preset defaults. Switch to Custom Macro Blend for manual shaping.",
    },
    "edm": {
        "label": "EDM",
        "description": "Uses a louder low-end and excitement-focused profile for electronic and club-forward material.",
        "macro_note": "Bass, Volume, and Effects are locked to the EDM preset defaults. Switch to Custom Macro Blend for manual shaping.",
    },
    "vocal": {
        "label": "Vocal Focus",
        "description": "Uses a restrained low-end and clearer midrange profile for vocal-led material.",
        "macro_note": "Bass, Volume, and Effects are locked to the Vocal Focus preset defaults. Switch to Custom Macro Blend for manual shaping.",
    },
    "custom": {
        "label": "Custom Macro Blend",
        "description": "Starts from a neutral mastering base and unlocks Bass, Volume, and Effects for manual tuning.",
        "macro_note": "Bass, Volume, and Effects are active in Custom Macro Blend. The mastering engine uses a balanced base and then applies your manual macro values.",
    },
}

_MASTERING_PROFILE_ALIASES = {
    "auto": "auto",
    "auto analyze": "auto",
    "balanced": "balanced",
    "balanced preset": "balanced",
    "edm": "edm",
    "edm preset": "edm",
    "vocal": "vocal",
    "vocal focus": "vocal",
    "vocal focus preset": "vocal",
    "custom": "custom",
    "custom macro blend": "custom",
}

_STEM_MODEL_STRATEGY_METADATA = {
    "Automatic mastering stack": {
        "model_name": "mastering",
        "description": "Uses a BS-Roformer-first mastering split, then derives drums, bass, and other through the built-in multi-stage separator chain.",
    },
    "Demucs fine-tuned 4-stem": {
        "model_name": "htdemucs_ft.yaml",
        "description": "Forces the Demucs fine-tuned 4-stem separator for a direct modern layer split.",
    },
    "HDemucs MMI 4-stem": {
        "model_name": "hdemucs_mmi.yaml",
        "description": "Forces the HDemucs MMI 4-stem separator, often steadier on dense stereo mixes.",
    },
    "Custom checkpoint override": {
        "model_name": "__custom__",
        "description": "Lets you provide an exact separator checkpoint or YAML name manually.",
    },
}

_STEM_MODEL_STRATEGY_LABEL_BY_VALUE = {
    str(spec["model_name"]).lower(): label
    for label, spec in _STEM_MODEL_STRATEGY_METADATA.items()
    if str(spec["model_name"]).strip() != "__custom__"
}


def normalize_mastering_profile_selection(profile_name: str | None) -> str:
    normalized = str(profile_name or "").strip().lower()
    if normalized in _MASTERING_PROFILE_METADATA:
        return normalized
    return _MASTERING_PROFILE_ALIASES.get(normalized, "auto")


def _mastering_profile_defaults(profile_name: str) -> dict[str, float]:
    normalized = normalize_mastering_profile_selection(profile_name)
    if normalized == "custom":
        return {"bass": 0.5, "volume": 0.5, "effects": 0.5}

    from definers.audio.config import SmartMasteringConfig

    baseline = "balanced" if normalized == "auto" else normalized
    config = SmartMasteringConfig.from_preset(baseline)
    return {
        "bass": float(config.bass),
        "volume": float(config.volume),
        "effects": float(config.effects),
    }


def _coerce_macro_value(value: object, default_value: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default_value)


def _coerce_bounded_float(
    value: object,
    *,
    default_value: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        resolved = float(value)
    except Exception:
        resolved = float(default_value)
    return float(min(max(resolved, minimum), maximum))


def normalize_stem_glue_reverb_amount(value: object) -> float:
    return _coerce_bounded_float(
        value,
        default_value=STEM_GLUE_REVERB_DEFAULT,
        minimum=0.0,
        maximum=1.5,
    )


def normalize_stem_drum_edge_amount(value: object) -> float:
    return _coerce_bounded_float(
        value,
        default_value=STEM_DRUM_EDGE_DEFAULT,
        minimum=0.0,
        maximum=1.5,
    )


def normalize_stem_vocal_pullback_db(value: object) -> float:
    return _coerce_bounded_float(
        value,
        default_value=STEM_VOCAL_PULLBACK_DB_DEFAULT,
        minimum=0.0,
        maximum=3.0,
    )


def get_mastering_profile_ui_state(
    profile_name: str | None,
    bass: object = None,
    volume: object = None,
    effects: object = None,
) -> dict[str, object]:
    normalized = normalize_mastering_profile_selection(profile_name)
    metadata = _MASTERING_PROFILE_METADATA[normalized]
    defaults = _mastering_profile_defaults(normalized)
    controls_enabled = normalized == "custom"

    if controls_enabled:
        bass_value = _coerce_macro_value(bass, defaults["bass"])
        volume_value = _coerce_macro_value(volume, defaults["volume"])
        effects_value = _coerce_macro_value(effects, defaults["effects"])
    else:
        bass_value = defaults["bass"]
        volume_value = defaults["volume"]
        effects_value = defaults["effects"]

    return {
        "selection": normalized,
        "label": metadata["label"],
        "description": f"**{metadata['label']}:** {metadata['description']}",
        "macro_note": metadata["macro_note"],
        "controls_enabled": controls_enabled,
        "bass": bass_value,
        "volume": volume_value,
        "effects": effects_value,
    }


def resolve_mastering_request(
    profile_name: str | None,
    bass: object,
    volume: object,
    effects: object,
) -> tuple[str, dict[str, object]]:
    state = get_mastering_profile_ui_state(
        profile_name,
        bass,
        volume,
        effects,
    )
    normalized = str(state["selection"])
    request_kwargs: dict[str, object] = {}

    if normalized == "custom":
        request_kwargs.update(
            {
                "preset": "balanced",
                "bass": float(state["bass"]),
                "volume": float(state["volume"]),
                "effects": float(state["effects"]),
            }
        )
    elif normalized in {"balanced", "edm", "vocal"}:
        request_kwargs["preset"] = normalized

    return str(state["label"]), request_kwargs


def _normalized_stem_model_strategy_label(value: str | None) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        return STEM_MODEL_STRATEGY_CHOICES[0]

    normalized_lower = normalized.lower()
    for label in STEM_MODEL_STRATEGY_CHOICES:
        if normalized_lower == label.lower():
            return label
    mapped_label = _STEM_MODEL_STRATEGY_LABEL_BY_VALUE.get(normalized_lower)
    if mapped_label is not None:
        return mapped_label
    return STEM_MODEL_STRATEGY_CHOICES[-1]


def is_custom_stem_model_strategy(value: str | None) -> bool:
    return (
        _normalized_stem_model_strategy_label(value)
        == "Custom checkpoint override"
    )


def resolve_stem_model_name(
    model_selection: str | None,
    model_override: str | None = None,
) -> str:
    override = str(model_override or "").strip()
    if override:
        return override

    normalized = str(model_selection or "").strip()
    if not normalized:
        return "mastering"

    normalized_lower = normalized.lower()
    mapped_label = _STEM_MODEL_STRATEGY_LABEL_BY_VALUE.get(normalized_lower)
    if mapped_label is not None:
        return str(_STEM_MODEL_STRATEGY_METADATA[mapped_label]["model_name"])
    if normalized_lower not in {
        label.lower() for label in STEM_MODEL_STRATEGY_CHOICES
    }:
        return normalized

    strategy_label = _normalized_stem_model_strategy_label(normalized)
    strategy_value = str(
        _STEM_MODEL_STRATEGY_METADATA[strategy_label]["model_name"]
    )
    if strategy_value == "__custom__":
        return "mastering"
    if normalized_lower == strategy_label.lower():
        return strategy_value
    return normalized


def stem_model_choice_label(
    model_selection: str | None,
    model_override: str | None = None,
) -> str:
    override = str(model_override or "").strip()
    if override:
        return f"Custom checkpoint override ({override})"

    normalized = str(model_selection or "").strip()
    if not normalized:
        return STEM_MODEL_STRATEGY_CHOICES[0]

    normalized_lower = normalized.lower()
    if normalized_lower in _STEM_MODEL_STRATEGY_LABEL_BY_VALUE:
        return _STEM_MODEL_STRATEGY_LABEL_BY_VALUE[normalized_lower]

    strategy_label = _normalized_stem_model_strategy_label(normalized)
    if strategy_label == "Custom checkpoint override":
        return f"Custom checkpoint override ({normalized})"
    return strategy_label


def describe_stem_model_choice(
    model_selection: str | None,
    model_override: str | None = None,
) -> str:
    override = str(model_override or "").strip()
    if override:
        return (
            "**Stem Strategy:** Custom checkpoint override. "
            f"Running `{override}`."
        )

    normalized = str(model_selection or "").strip()
    if (
        normalized
        and normalized.lower()
        not in {label.lower() for label in STEM_MODEL_STRATEGY_CHOICES}
        and normalized.lower() not in _STEM_MODEL_STRATEGY_LABEL_BY_VALUE
    ):
        return (
            "**Stem Strategy:** Custom checkpoint override. "
            f"Running `{normalized}`."
        )

    strategy_label = _normalized_stem_model_strategy_label(normalized)
    strategy = _STEM_MODEL_STRATEGY_METADATA[strategy_label]
    if str(strategy["model_name"]) == "__custom__":
        return (
            "**Stem Strategy:** Custom checkpoint override. "
            "Enter an exact separator checkpoint or YAML name below."
        )
    return (
        f"**Stem Strategy:** {strategy_label}. "
        f"{strategy['description']} "
        f"(`{strategy['model_name']}`)"
    )


def normalize_audio_format_choice(format_choice: str) -> str:
    return str(format_choice).strip().lower().lstrip(".") or "wav"


def _temp_audio_output_path(format_choice: str) -> str:
    from definers.system.output_paths import managed_output_path

    return managed_output_path(
        normalize_audio_format_choice(format_choice),
        section="audio",
        stem="audio_output",
    )


def _report_audio_activity(
    item_label: str,
    *,
    detail: str,
    completed: int | None = None,
    total: int | None = None,
) -> None:
    from definers.system.download_activity import report_download_activity

    report_download_activity(
        item_label,
        detail=detail,
        phase="step",
        completed=completed,
        total=total,
    )


def _coerce_optional_int(value: float | int | None) -> int | None:
    if value is None:
        return None
    resolved_value = int(value)
    if resolved_value <= 0:
        return None
    return resolved_value


def _coerce_optional_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    resolved_value = float(value)
    if resolved_value <= 0.0:
        return None
    return resolved_value


def _supports_keyword_argument(function, argument_name: str) -> bool:
    try:
        signature = inspect.signature(function)
    except Exception:
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == argument_name:
            return True
    return False


def _write_json_payload(payload: dict[str, object]) -> str:
    from definers.system.output_paths import managed_output_path

    destination_path = Path(
        managed_output_path(
            "json",
            section="audio/reports",
            stem="report",
        )
    )
    destination_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(destination_path)


def _write_text_payload(text: str, *, extension: str = "md") -> str:
    from definers.system.output_paths import managed_output_path

    destination_path = Path(
        managed_output_path(
            extension,
            section="audio/reports",
            stem="report",
        )
    )
    destination_path.write_text(str(text), encoding="utf-8")
    return str(destination_path)


def _convert_audio_outputs(
    source_paths: list[str],
    format_choice: str,
) -> list[str]:
    normalized_format = normalize_audio_format_choice(format_choice)
    if normalized_format == "wav":
        return source_paths

    from definers.audio import read_audio, save_audio
    from definers.system.output_paths import managed_output_session_dir

    output_dir = Path(
        managed_output_session_dir(
            "audio/converted",
            stem="converted",
        )
    )
    converted_paths: list[str] = []
    for source_path in source_paths:
        sample_rate, audio_signal = read_audio(source_path)
        destination_path = (
            output_dir / f"{Path(source_path).stem}.{normalized_format}"
        )
        converted_path = save_audio(
            destination_path=str(destination_path),
            audio_signal=audio_signal,
            sample_rate=sample_rate,
        )
        converted_paths.append(str(converted_path))
    return converted_paths


def _format_metric_value(value: object, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}{suffix}"
    except Exception:
        return str(value)


def _metric_attr(metric: object, attribute_name: str) -> object:
    if metric is None:
        return None
    return getattr(metric, attribute_name, None)


def format_mastering_summary(
    report: object | None,
    *,
    control_mode: str | None = None,
    stem_mastering: bool,
    stem_strategy: str | None = None,
    stem_files: list[str],
) -> str:
    if report is None:
        return "Mastering completed without diagnostics."

    final_master_metrics = getattr(report, "final_in_memory_metrics", None)
    output_metrics = getattr(report, "output_metrics", None)
    decoded_metrics = getattr(report, "decoded_metrics", None)
    export_gain_applied_db = getattr(report, "export_gain_applied_db", None)
    export_peak_alignment_mode = (
        str(getattr(report, "export_peak_alignment_mode", "") or "")
        .strip()
        .lower()
    )
    export_peak_alignment_target_dbfs = getattr(
        report,
        "export_peak_alignment_target_dbfs",
        None,
    )
    peak_catch_events = tuple(getattr(report, "peak_catch_events", ()) or ())
    character_stage_decision = getattr(report, "character_stage_decision", None)
    summary_lines = []
    verdict = "Master ready."
    if final_master_metrics is not None and decoded_metrics is not None:
        final_lufs = _metric_attr(final_master_metrics, "integrated_lufs")
        decoded_lufs = _metric_attr(decoded_metrics, "integrated_lufs")
        if (
            final_lufs is not None
            and decoded_lufs is not None
            and abs(float(decoded_lufs) - float(final_lufs)) > 0.75
        ):
            verdict = "Master ready, but decoded playback shifts audibly from the final master."
    summary_lines.append(f"**Verdict:** {verdict}")
    if control_mode:
        summary_lines.append(f"**Control Mode:** {control_mode}")
    if stem_mastering and stem_strategy:
        summary_lines.append(f"**Stem Strategy:** {stem_strategy}")
    summary_lines.extend(
        [
            f"**Preset:** {getattr(report, 'preset_name', 'n/a')}",
            f"**Delivery Profile:** {getattr(report, 'delivery_profile_name', 'n/a')}",
            f"**Stem-aware Path:** {'On' if stem_mastering else 'Off'}",
            f"**Final Master:** {_format_metric_value(_metric_attr(final_master_metrics, 'integrated_lufs'), ' LUFS')} / {_format_metric_value(_metric_attr(final_master_metrics, 'true_peak_dbfs'), ' dBFS')}",
            f"**Delivered File:** {_format_metric_value(_metric_attr(output_metrics, 'integrated_lufs'), ' LUFS')} / {_format_metric_value(_metric_attr(output_metrics, 'true_peak_dbfs'), ' dBFS')}",
        ]
    )
    if decoded_metrics is not None:
        summary_lines.append(
            f"**Decoded Playback:** {_format_metric_value(_metric_attr(decoded_metrics, 'integrated_lufs'), ' LUFS')} / {_format_metric_value(_metric_attr(decoded_metrics, 'true_peak_dbfs'), ' dBFS')}"
        )
    if export_peak_alignment_mode == "align_to_ceil":
        summary_lines.append(
            f"**Ceiling Alignment:** Export used the available ceiling at {_format_metric_value(export_peak_alignment_target_dbfs, ' dBFS')} with {_format_metric_value(export_gain_applied_db, ' dB')} of gain."
        )
    processing_moves: list[str] = []
    if character_stage_decision is not None and bool(
        getattr(character_stage_decision, "applied", False)
    ):
        if bool(getattr(character_stage_decision, "reverted", False)):
            processing_moves.append("character pass rolled back")
        else:
            processing_moves.append("character pass kept")
    if peak_catch_events:
        processing_moves.append(f"peak catch x{len(peak_catch_events)}")
    headroom_recovery_mode = getattr(report, "headroom_recovery_mode", None)
    if headroom_recovery_mode is not None:
        processing_moves.append(f"headroom recovery: {headroom_recovery_mode}")
    if processing_moves:
        summary_lines.append(
            "**Processing Moves:** " + ", ".join(processing_moves)
        )
    if stem_files:
        summary_lines.append(f"**Mastered Stems:** {len(stem_files)} files")
    return "\n\n".join(summary_lines)


def resolve_mastering_stem_previews(
    stem_files: list[str],
) -> tuple[dict[str, str | None], list[str]]:
    previews = {key: None for key in MASTERING_STEM_PREVIEW_ORDER}
    extras: list[str] = []
    for stem_path in stem_files:
        resolved_path = str(stem_path)
        normalized_name = Path(resolved_path).stem.strip().lower()
        matched_key = None
        for key in MASTERING_STEM_PREVIEW_ORDER:
            if key in normalized_name and previews[key] is None:
                previews[key] = resolved_path
                matched_key = key
                break
        if matched_key is None:
            extras.append(resolved_path)
    return previews, extras


def _job_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _job_artifacts(manifest: dict[str, object]) -> dict[str, object]:
    return _job_dict(manifest.get("artifacts"))


def _job_settings(manifest: dict[str, object]) -> dict[str, object]:
    return _job_dict(manifest.get("settings"))


def _job_input(manifest: dict[str, object]) -> dict[str, object]:
    return _job_dict(manifest.get("input"))


def _job_analysis(manifest: dict[str, object]) -> dict[str, object]:
    return _job_dict(manifest.get("analysis"))


def _read_mastering_job(job_dir: str) -> dict[str, object]:
    manifest = read_manifest(job_dir)
    if str(manifest.get("job_type", "")).strip() != "audio-mastering-job":
        raise ValueError(
            "The selected job folder is not an audio mastering job."
        )
    return manifest


def _scan_audio_job_files(directory: object) -> dict[str, str]:
    return scan_file_map(directory, suffixes=_AUDIO_JOB_SUFFIXES)


def _refresh_mastering_job_artifacts(
    manifest: dict[str, object],
) -> dict[str, object]:
    artifacts = _job_artifacts(manifest)
    raw_stems_dir = existing_path(artifacts.get("raw_stems_dir"))
    processed_stems_dir = existing_path(artifacts.get("processed_stems_dir"))
    refreshed_artifacts = {
        **artifacts,
        "raw_stems_dir": raw_stems_dir,
        "raw_stems": _scan_audio_job_files(raw_stems_dir),
        "processed_stems_dir": processed_stems_dir,
        "processed_stems": _scan_audio_job_files(processed_stems_dir),
        "mixed_path": existing_path(artifacts.get("mixed_path")),
        "mastered_path": existing_path(artifacts.get("mastered_path")),
        "report_path": existing_path(artifacts.get("report_path")),
        "report_summary": str(artifacts.get("report_summary", "")).strip(),
    }
    manifest["artifacts"] = refreshed_artifacts
    return manifest


def refresh_mastering_job(job_dir: str) -> dict[str, object]:
    manifest = _read_mastering_job(job_dir)
    refreshed_manifest = _refresh_mastering_job_artifacts(manifest)
    return write_manifest(job_dir, refreshed_manifest)


def resolve_mastering_job_status(
    manifest: dict[str, object],
) -> tuple[str, str]:
    settings = _job_settings(manifest)
    artifacts = _job_artifacts(manifest)
    if existing_path(artifacts.get("mastered_path")) is not None:
        return (
            "Master ready",
            "Done. Download the final master, the report, and any saved stems.",
        )

    if bool(settings.get("stem_mastering")):
        if existing_path(artifacts.get("mixed_path")) is not None:
            return (
                "Stem mix ready",
                "Next: Finalize Master. This stage renders the delivery file and report from the stem mix.",
            )
        if _job_dict(artifacts.get("raw_stems")):
            return (
                "Stem separation finished",
                "Next: Build Stem Mix. Stem separation is the heavy stage; mix building reuses those artifacts.",
            )
        return (
            "Job prepared",
            "Next: Separate Stems. Stem separation is the heavy stage; later steps reuse the saved job artifacts.",
        )

    return (
        "Job prepared",
        "Next: Finalize Master. Stereo-only mastering skips stem separation and goes straight to final delivery.",
    )


def format_mastering_job_status(
    manifest: dict[str, object],
    *,
    title: str | None = None,
    detail: str | None = None,
) -> str:
    resolved_title, resolved_detail = resolve_mastering_job_status(manifest)
    settings = _job_settings(manifest)
    artifacts = _job_artifacts(manifest)
    analysis = _job_analysis(manifest)
    input_data = _job_input(manifest)
    resolved_kwargs = _job_dict(manifest.get("resolved_mastering_kwargs"))
    quality_flags = analysis.get("quality_flags") or ()
    status_items: list[tuple[str, object]] = [
        ("Job folder", manifest.get("job_dir", "")),
        ("Input", Path(str(input_data.get("path", "input"))).name),
        (
            "Mode",
            "Stem-Aware"
            if bool(settings.get("stem_mastering"))
            else "Stereo Only",
        ),
        (
            "Output format",
            str(settings.get("format_choice", "wav")).upper(),
        ),
        ("Requested profile", settings.get("profile_name", "Balanced")),
        ("Resolved control mode", settings.get("control_mode", "Balanced")),
        (
            "Suggested preset",
            analysis.get("suggested_preset", "balanced"),
        ),
        (
            "Resolved preset",
            resolved_kwargs.get("preset", "balanced"),
        ),
        (
            "Quality flags",
            ", ".join(str(flag) for flag in quality_flags)
            if quality_flags
            else "None",
        ),
    ]
    if bool(settings.get("stem_mastering")):
        status_items.extend(
            [
                (
                    "Glue reverb",
                    f"{normalize_stem_glue_reverb_amount(settings.get('stem_glue_reverb_amount')):.2f}x",
                ),
                (
                    "Drum edge",
                    f"{normalize_stem_drum_edge_amount(settings.get('stem_drum_edge_amount')):.2f}x",
                ),
                (
                    "Extra vocal pullback",
                    f"{normalize_stem_vocal_pullback_db(settings.get('stem_vocal_pullback_db')):.2f} dB",
                ),
            ]
        )
    status_items.extend(
        [
            ("Raw stems ready", len(_job_dict(artifacts.get("raw_stems")))),
            (
                "Processed stems ready",
                len(_job_dict(artifacts.get("processed_stems"))),
            ),
            (
                "Stem mix ready",
                existing_path(artifacts.get("mixed_path")) is not None,
            ),
            (
                "Final master ready",
                existing_path(artifacts.get("mastered_path")) is not None,
            ),
        ]
    )
    return status_card_markdown(
        title or resolved_title,
        detail or resolved_detail,
        status_items,
    )


def render_mastering_job_view(
    job_dir: str,
    *,
    title: str | None = None,
    detail: str | None = None,
) -> tuple[
    str,
    str,
    list[str] | None,
    list[str] | None,
    str | None,
    str | None,
    str | None,
    str,
    str,
]:
    manifest = refresh_mastering_job(job_dir)
    artifacts = _job_artifacts(manifest)
    raw_files = [
        file_path
        for file_path in _job_dict(artifacts.get("raw_stems")).values()
        if existing_path(file_path) is not None
    ]
    processed_files = [
        file_path
        for file_path in _job_dict(artifacts.get("processed_stems")).values()
        if existing_path(file_path) is not None
    ]
    report_summary = str(artifacts.get("report_summary", "")).strip()
    return (
        str(manifest.get("job_dir", job_dir)),
        format_mastering_job_status(
            manifest,
            title=title,
            detail=detail,
        ),
        raw_files or None,
        processed_files or None,
        existing_path(artifacts.get("mixed_path")),
        existing_path(artifacts.get("mastered_path")),
        existing_path(artifacts.get("report_path")),
        report_summary,
        manifest_markdown(manifest),
    )


def prepare_mastering_job(
    audio_path: str,
    format_choice: str,
    profile_name: str,
    bass: float,
    volume: float,
    effects: float,
    stem_mastering: bool,
    stem_model_name: str,
    stem_shifts: int,
    stem_mix_headroom_db: float,
    save_mastered_stems: bool,
    stem_model_override: str | None = None,
    stem_glue_reverb_amount: float = STEM_GLUE_REVERB_DEFAULT,
    stem_drum_edge_amount: float = STEM_DRUM_EDGE_DEFAULT,
    stem_vocal_pullback_db: float = STEM_VOCAL_PULLBACK_DB_DEFAULT,
) -> dict[str, object]:
    from definers.audio.io import read_audio
    from definers.audio.mastering.input_analysis import (
        _analyze_mastering_input,
        _resolve_mastering_kwargs_for_input,
    )

    source_path = Path(str(audio_path or "").strip())
    if not source_path.exists():
        raise FileNotFoundError(
            "Upload a mix before preparing a mastering job."
        )

    _report_audio_activity(
        "Prepare mastering job",
        detail="Creating the guided mastering workspace.",
        completed=1,
        total=4,
    )
    job_dir = create_job_dir("audio/mastering_jobs", stem=source_path.stem)
    copied_input_path = Path(job_dir) / (
        f"input{source_path.suffix.lower() or '.wav'}"
    )
    copy2(source_path, copied_input_path)

    _report_audio_activity(
        "Analyze mastering input",
        detail="Reading the source mix and analyzing the mastering route.",
        completed=2,
        total=4,
    )
    sample_rate, signal = read_audio(str(copied_input_path))
    control_mode, profile_kwargs = resolve_mastering_request(
        profile_name,
        bass,
        volume,
        effects,
    )
    input_analysis = _analyze_mastering_input(signal, sample_rate)

    _report_audio_activity(
        "Resolve mastering settings",
        detail="Resolving presets, quality flags, and stem options.",
        completed=3,
        total=4,
    )
    resolved_mastering_kwargs = _resolve_mastering_kwargs_for_input(
        signal,
        sample_rate,
        dict(profile_kwargs),
        input_analysis=input_analysis,
    )
    resolved_mastering_kwargs.setdefault(
        "resampling_target",
        int(input_analysis.target_sample_rate),
    )
    resolved_model_name = resolve_stem_model_name(
        stem_model_name,
        stem_model_override,
    )
    resolved_stem_glue_reverb_amount = normalize_stem_glue_reverb_amount(
        stem_glue_reverb_amount
    )
    resolved_stem_drum_edge_amount = normalize_stem_drum_edge_amount(
        stem_drum_edge_amount
    )
    resolved_stem_vocal_pullback_db = normalize_stem_vocal_pullback_db(
        stem_vocal_pullback_db
    )
    resolved_mastering_kwargs.update(
        {
            "stem_glue_reverb_amount": resolved_stem_glue_reverb_amount,
            "stem_drum_edge_amount": resolved_stem_drum_edge_amount,
            "stem_vocal_pullback_db": resolved_stem_vocal_pullback_db,
        }
    )
    manifest = {
        "job_type": "audio-mastering-job",
        "job_version": 1,
        "job_dir": job_dir,
        "input": {
            "path": str(copied_input_path),
            "sample_rate": int(sample_rate),
        },
        "settings": {
            "format_choice": normalize_audio_format_choice(format_choice),
            "profile_name": str(profile_name),
            "bass": float(bass),
            "volume": float(volume),
            "effects": float(effects),
            "control_mode": str(control_mode),
            "stem_mastering": bool(stem_mastering),
            "stem_model_name": str(resolved_model_name),
            "stem_model_label": stem_model_choice_label(
                stem_model_name,
                stem_model_override,
            ),
            "stem_shifts": int(stem_shifts),
            "stem_mix_headroom_db": float(stem_mix_headroom_db),
            "save_mastered_stems": bool(save_mastered_stems),
            "stem_glue_reverb_amount": resolved_stem_glue_reverb_amount,
            "stem_drum_edge_amount": resolved_stem_drum_edge_amount,
            "stem_vocal_pullback_db": resolved_stem_vocal_pullback_db,
        },
        "analysis": {
            "suggested_preset": str(input_analysis.preset_name),
            "quality_flags": list(input_analysis.quality_flags),
            "target_sample_rate": int(input_analysis.target_sample_rate),
        },
        "resolved_mastering_kwargs": dict(resolved_mastering_kwargs),
        "artifacts": {
            "raw_stems_dir": None,
            "raw_stems": {},
            "processed_stems_dir": None,
            "processed_stems": {},
            "mixed_path": None,
            "mastered_path": None,
            "report_path": None,
            "report_summary": "",
        },
    }

    _report_audio_activity(
        "Persist mastering job",
        detail="Writing the job manifest and copied input into the managed output workspace.",
        completed=4,
        total=4,
    )
    return write_manifest(job_dir, manifest)


def run_full_mastering_job(
    audio_path: str,
    format_choice: str,
    profile_name: str,
    bass: float,
    volume: float,
    effects: float,
    stem_mastering: bool,
    stem_model_name: str,
    stem_shifts: int,
    stem_mix_headroom_db: float,
    save_mastered_stems: bool,
    stem_model_override: str | None = None,
    stem_glue_reverb_amount: float = STEM_GLUE_REVERB_DEFAULT,
    stem_drum_edge_amount: float = STEM_DRUM_EDGE_DEFAULT,
    stem_vocal_pullback_db: float = STEM_VOCAL_PULLBACK_DB_DEFAULT,
) -> dict[str, object]:
    manifest = prepare_mastering_job(
        audio_path,
        format_choice,
        profile_name,
        bass,
        volume,
        effects,
        stem_mastering,
        stem_model_name,
        stem_shifts,
        stem_mix_headroom_db,
        save_mastered_stems,
        stem_model_override=stem_model_override,
        stem_glue_reverb_amount=stem_glue_reverb_amount,
        stem_drum_edge_amount=stem_drum_edge_amount,
        stem_vocal_pullback_db=stem_vocal_pullback_db,
    )
    job_dir = str(manifest["job_dir"])
    if bool(dict(manifest.get("settings", {})).get("stem_mastering")):
        separate_mastering_job_stems(job_dir)
        build_mastering_job_mix(job_dir)
    return finalize_mastering_job(job_dir)


def separate_mastering_job_stems(job_dir: str) -> dict[str, object]:
    from definers.audio.mastering.engine import _master_internal
    from definers.audio.stems import separate_stem_layers

    manifest = _read_mastering_job(job_dir)
    settings = _job_settings(manifest)
    if not bool(settings.get("stem_mastering")):
        return refresh_mastering_job(job_dir)

    input_path = str(_job_input(manifest).get("path", ""))
    raw_stems_dir = Path(str(job_dir)) / "raw_stems"
    raw_stems_dir.mkdir(parents=True, exist_ok=True)
    mastering_kwargs = dict(
        _job_dict(manifest.get("resolved_mastering_kwargs"))
    )
    pre_mastered_path = str(Path(str(job_dir)) / "pre_mastered.wav")
    _report_audio_activity(
        "Initial mastering pass",
        detail="Applying initial mix processing before stem separation.",
        completed=1,
        total=3,
    )
    _master_internal(
        input_path=input_path,
        output_path=pre_mastered_path,
        stem_mastering=False,
        raise_on_error=True,
        **mastering_kwargs,
    )
    manifest["artifacts"] = {
        **_job_artifacts(manifest),
        "pre_mastered_path": pre_mastered_path,
    }
    _report_audio_activity(
        "Separate stems",
        detail="Running the heavy stem-separation stage and saving raw layers.",
        completed=2,
        total=3,
    )
    stem_paths, resolved_output_dir = separate_stem_layers(
        pre_mastered_path,
        model_name=str(settings.get("stem_model_name", "mastering")),
        shifts=max(int(settings.get("stem_shifts", 1)), 1),
        quality_flags=tuple(
            str(flag).strip()
            for flag in _job_analysis(manifest).get("quality_flags", ())
            if str(flag).strip()
        ),
        output_dir=str(raw_stems_dir),
    )
    manifest["artifacts"] = {
        **_job_artifacts(manifest),
        "pre_mastered_path": pre_mastered_path,
        "raw_stems_dir": str(resolved_output_dir),
        "raw_stems": {
            str(stem_name): str(stem_path)
            for stem_name, stem_path in stem_paths.items()
        },
    }
    _report_audio_activity(
        "Publish raw stems",
        detail="Raw stems are ready for inspection and the mix-building step.",
        completed=3,
        total=3,
    )
    return write_manifest(job_dir, manifest)


def build_mastering_job_mix(job_dir: str) -> dict[str, object]:
    from definers.audio.io import read_audio, save_audio
    from definers.audio.mastering.engine import (
        SmartMastering,
        _process_stem_signal,
    )
    from definers.audio.mastering.stems import process_stem_layers

    manifest = _read_mastering_job(job_dir)
    settings = _job_settings(manifest)
    artifacts = _job_artifacts(manifest)
    if not bool(settings.get("stem_mastering")):
        return refresh_mastering_job(job_dir)
    raw_stems = _job_dict(artifacts.get("raw_stems"))
    if not raw_stems:
        raise ValueError("Run stem separation before building the stem mix.")

    processed_stems_dir = Path(str(job_dir)) / "processed_stems"
    processed_stems_dir.mkdir(parents=True, exist_ok=True)
    base_kwargs = dict(_job_dict(manifest.get("resolved_mastering_kwargs")))
    input_sample_rate = int(_job_input(manifest).get("sample_rate", 44100))
    base_mastering = SmartMastering(input_sample_rate, **base_kwargs)

    def reuse_existing_stems(_audio_path, **_kwargs):
        return raw_stems, str(
            artifacts.get("raw_stems_dir") or processed_stems_dir
        )

    _report_audio_activity(
        "Build stem mix",
        detail="Processing separated stems and building the guided stem mix.",
        completed=1,
        total=3,
    )
    mixed_sample_rate, mixed_signal = process_stem_layers(
        str(
            existing_path(artifacts.get("pre_mastered_path"))
            or _job_input(manifest).get("path", "")
        ),
        base_config=base_mastering.config,
        base_mastering_kwargs=base_kwargs,
        process_stem_fn=_process_stem_signal,
        separate_stems_fn=reuse_existing_stems,
        read_audio_fn=read_audio,
        delete_fn=lambda _path: None,
        model_name=str(settings.get("stem_model_name", "mastering")),
        shifts=max(int(settings.get("stem_shifts", 1)), 1),
        quality_flags=tuple(
            str(flag).strip()
            for flag in _job_analysis(manifest).get("quality_flags", ())
            if str(flag).strip()
        ),
        mix_headroom_db=float(settings.get("stem_mix_headroom_db", 6.0)),
        save_mastered_stems=bool(settings.get("save_mastered_stems", True)),
        mastered_stems_output_dir=(
            str(processed_stems_dir)
            if bool(settings.get("save_mastered_stems", True))
            else None
        ),
        save_audio_fn=save_audio,
        mastered_stems_format="wav",
        mastered_stems_bit_depth=32,
        mastered_stems_bitrate=320,
        mastered_stems_compression_level=9,
    )
    _report_audio_activity(
        "Write stem mix",
        detail="Saving the combined stem mix artifact.",
        completed=2,
        total=3,
    )
    mixed_path = str(Path(str(job_dir)) / "stem_mix.wav")
    save_audio(
        destination_path=mixed_path,
        audio_signal=mixed_signal,
        sample_rate=int(mixed_sample_rate),
        bit_depth=32,
    )
    manifest["artifacts"] = {
        **artifacts,
        "processed_stems_dir": str(processed_stems_dir),
        "processed_stems": _scan_audio_job_files(processed_stems_dir),
        "mixed_path": mixed_path,
    }
    _report_audio_activity(
        "Publish stem mix",
        detail="Processed stems and the stem mix are ready for final mastering.",
        completed=3,
        total=3,
    )
    return write_manifest(job_dir, manifest)


def finalize_mastering_job(job_dir: str) -> dict[str, object]:
    from definers.audio.io import read_audio, save_audio
    from definers.audio.mastering.engine import _render_master_output, master

    manifest = _read_mastering_job(job_dir)
    settings = _job_settings(manifest)
    artifacts = _job_artifacts(manifest)
    input_path = str(_job_input(manifest).get("path", ""))
    output_ext = normalize_audio_format_choice(
        str(settings.get("format_choice", "wav"))
    )
    output_path = str(
        Path(str(job_dir)) / f"{Path(input_path).stem}_mastered.{output_ext}"
    )
    report_path = str(
        Path(str(job_dir)) / f"{Path(input_path).stem}_mastering_report.md"
    )
    mastering_kwargs = dict(
        _job_dict(manifest.get("resolved_mastering_kwargs"))
    )

    _report_audio_activity(
        "Finalize master",
        detail="Rendering the final delivery file and mastering report.",
        completed=1,
        total=2,
    )
    if bool(settings.get("stem_mastering")):
        mixed_path = existing_path(artifacts.get("mixed_path"))
        if mixed_path is None:
            raise ValueError("Build the stem mix before finalizing the master.")
        input_signal_sample_rate, input_signal = read_audio(input_path)
        mixed_sample_rate, mixed_signal = read_audio(mixed_path)
        mastered_path, report = _render_master_output(
            input_path,
            input_signal=input_signal,
            processing_signal=mixed_signal,
            processing_sample_rate=int(mixed_sample_rate),
            output_path=output_path,
            report_path=report_path,
            report_indent=2,
            bit_depth=32,
            bitrate=320,
            compression_level=9,
            read_audio_fn=read_audio,
            save_audio_fn=save_audio,
            stem_mastered_input=True,
            **mastering_kwargs,
        )
        _ = input_signal_sample_rate
    else:
        mastered_path, report = master(
            input_path,
            output_path=output_path,
            report_path=report_path,
            raise_on_error=True,
            stem_mastering=False,
            **mastering_kwargs,
        )

    manifest["artifacts"] = {
        **artifacts,
        "mastered_path": mastered_path,
        "report_path": report_path
        if existing_path(report_path) is not None
        else None,
        "report_summary": (
            report.to_musician_markdown() if report is not None else ""
        ),
    }
    _report_audio_activity(
        "Publish final master",
        detail="Final master, report, and any saved stems are ready.",
        completed=2,
        total=2,
    )
    return write_manifest(job_dir, manifest)


def run_mastering_tool(
    audio_path: str,
    format_choice: str,
    profile_name: str,
    bass: float,
    volume: float,
    effects: float,
    stem_mastering: bool,
    stem_model_name: str,
    stem_shifts: int,
    stem_mix_headroom_db: float,
    save_mastered_stems: bool,
    stem_model_override: str | None = None,
    stem_glue_reverb_amount: float = STEM_GLUE_REVERB_DEFAULT,
    stem_drum_edge_amount: float = STEM_DRUM_EDGE_DEFAULT,
    stem_vocal_pullback_db: float = STEM_VOCAL_PULLBACK_DB_DEFAULT,
) -> tuple[str, str | None, str, list[str]]:
    from definers.audio import master

    _report_audio_activity(
        "Resolve mastering request",
        detail="Normalizing the mastering profile and output paths.",
        completed=1,
        total=4,
    )
    output_path = _temp_audio_output_path(format_choice)
    report_path = _write_text_payload("Mastering report pending.\n")
    should_collect_mastered_stems = bool(stem_mastering)
    control_mode, profile_kwargs = resolve_mastering_request(
        profile_name,
        bass,
        volume,
        effects,
    )
    resolved_stem_model_name = resolve_stem_model_name(
        stem_model_name,
        stem_model_override,
    )
    resolved_stem_glue_reverb_amount = normalize_stem_glue_reverb_amount(
        stem_glue_reverb_amount
    )
    resolved_stem_drum_edge_amount = normalize_stem_drum_edge_amount(
        stem_drum_edge_amount
    )
    resolved_stem_vocal_pullback_db = normalize_stem_vocal_pullback_db(
        stem_vocal_pullback_db
    )
    mastering_kwargs: dict[str, object] = {
        "report_path": report_path,
        "stem_mastering": bool(stem_mastering),
        "stem_model_name": resolved_stem_model_name,
        "stem_shifts": max(int(stem_shifts), 1),
        "stem_mix_headroom_db": float(stem_mix_headroom_db),
        "save_mastered_stems": should_collect_mastered_stems,
        "stem_glue_reverb_amount": resolved_stem_glue_reverb_amount,
        "stem_drum_edge_amount": resolved_stem_drum_edge_amount,
        "stem_vocal_pullback_db": resolved_stem_vocal_pullback_db,
    }
    mastering_kwargs.update(profile_kwargs)

    _report_audio_activity(
        "Run mastering pipeline",
        detail="Processing the mix through the mastering engine.",
        completed=2,
        total=4,
    )
    mastered_path, report = master(
        audio_path,
        output_path=output_path,
        raise_on_error=True,
        **mastering_kwargs,
    )
    if mastered_path is None:
        raise RuntimeError("Mastering failed")

    _report_audio_activity(
        "Collect mastering artifacts",
        detail="Loading the mastering report and any rendered stems.",
        completed=3,
        total=4,
    )
    stem_files: list[str] = []
    stem_output_dir = (
        Path(mastered_path).parent / f"{Path(mastered_path).stem}_stems"
    )
    if should_collect_mastered_stems and stem_output_dir.exists():
        stem_files = [
            str(path)
            for path in sorted(stem_output_dir.iterdir())
            if path.is_file()
        ]

    resolved_report_path = report_path if Path(report_path).exists() else None
    _report_audio_activity(
        "Publish mastered output",
        detail="Mastered audio is ready for the interface.",
        completed=4,
        total=4,
    )
    return (
        mastered_path,
        resolved_report_path,
        format_mastering_summary(
            report,
            control_mode=control_mode,
            stem_mastering=bool(stem_mastering),
            stem_strategy=stem_model_choice_label(
                stem_model_name,
                stem_model_override,
            ),
            stem_files=stem_files,
        ),
        stem_files,
    )


def run_stem_separation_tool(
    audio_path: str,
    separation_mode: str,
    format_choice: str,
    model_name: str,
    shifts: int,
    model_override: str | None = None,
) -> tuple[str | None, list[str], str]:
    from definers.audio import separate_stem_layers, separate_stems
    from definers.system.output_paths import managed_output_session_dir

    normalized_format = normalize_audio_format_choice(format_choice)
    normalized_mode = str(separation_mode).strip().lower()
    _report_audio_activity(
        "Validate stem request",
        detail="Resolving the separation mode and output format.",
        completed=1,
        total=3,
    )

    if normalized_mode == "mastering_layers":
        resolved_model_name = resolve_stem_model_name(
            model_name,
            model_override,
        )
        separation_kwargs = {
            "model_name": resolved_model_name,
            "shifts": max(int(shifts), 1),
        }
        if _supports_keyword_argument(separate_stem_layers, "output_dir"):
            separation_kwargs["output_dir"] = managed_output_session_dir(
                "audio/stems",
                stem=Path(audio_path).stem,
            )
        _report_audio_activity(
            "Resolve stem strategy",
            detail="Preparing the mastering-layers separation route.",
            completed=1,
            total=3,
        )
        stem_paths, _output_dir = separate_stem_layers(
            audio_path,
            **separation_kwargs,
        )
        ordered_names = ["vocals", "drums", "bass", "other", "guitar", "piano"]
        outputs = [
            stem_paths[name] for name in ordered_names if name in stem_paths
        ]
        outputs.extend(
            stem_path
            for stem_name, stem_path in stem_paths.items()
            if stem_name not in ordered_names
        )
        outputs = _convert_audio_outputs(outputs, normalized_format)
        _report_audio_activity(
            "Convert stem exports",
            detail="Finalizing the separated layer files.",
            completed=2,
            total=3,
        )
        _report_audio_activity(
            "Publish stem outputs",
            detail="Separated mastering layers are ready.",
            completed=3,
            total=3,
        )
        summary = (
            "Separated mastering layers with "
            + stem_model_choice_label(model_name, model_override)
            + ": "
            + ", ".join(Path(output_path).stem for output_path in outputs)
        )
        return (outputs[0] if outputs else None, outputs, summary)

    if normalized_mode == "vocals_karaoke":
        _report_audio_activity(
            "Separate vocals and instrumental",
            detail="Running the vocal-plus-karaoke split.",
            completed=2,
            total=3,
        )
        outputs = list(
            separate_stems(audio_path, format_choice=normalized_format) or ()
        )
        if not outputs:
            raise RuntimeError("Stem separation failed")
        _report_audio_activity(
            "Publish stem outputs",
            detail="Vocals and instrumental stems are ready.",
            completed=3,
            total=3,
        )
        return outputs[0], outputs, "Created vocals and instrumental stems"

    separation_type = "acapella" if normalized_mode == "acapella" else "karaoke"
    _report_audio_activity(
        "Separate requested stem",
        detail="Running the selected two-stem export.",
        completed=2,
        total=3,
    )
    output_path = separate_stems(
        audio_path,
        separation_type=separation_type,
        format_choice=normalized_format,
    )
    if output_path is None:
        raise RuntimeError("Stem separation failed")
    _report_audio_activity(
        "Publish stem output",
        detail="Requested stem export is ready.",
        completed=3,
        total=3,
    )
    summary = (
        "Created vocal-only stem"
        if separation_type == "acapella"
        else "Created instrumental-only stem"
    )
    return output_path, [output_path], summary


def run_autotune_song_tool(
    audio_path: str,
    format_choice: str,
    strength: float,
    correct_timing: bool,
    quantize_grid_strength: int,
    tolerance_cents: int,
    attack_smoothing_ms: float,
) -> str:
    from definers.audio import autotune_song

    _report_audio_activity(
        "Prepare autotune request",
        detail="Normalizing the AutoTune settings and output path.",
        completed=1,
        total=2,
    )
    output_path = _temp_audio_output_path(format_choice)
    tuned_path = autotune_song(
        audio_path,
        output_path=output_path,
        strength=float(strength),
        correct_timing=bool(correct_timing),
        quantize_grid_strength=int(quantize_grid_strength),
        tolerance_cents=int(tolerance_cents),
        attack_smoothing_ms=float(attack_smoothing_ms),
    )
    if tuned_path is None:
        raise RuntimeError("AutoTune failed")
    _report_audio_activity(
        "Publish autotuned audio",
        detail="AutoTuned audio is ready.",
        completed=2,
        total=2,
    )
    return tuned_path


def run_humanize_vocals_tool(
    audio_path: str,
    amount: float,
    format_choice: str,
) -> str:
    from definers.audio import humanize_vocals

    _report_audio_activity(
        "Humanize vocal take",
        detail="Applying vocal timing and pitch variation.",
        completed=1,
        total=2,
    )
    humanized_path = humanize_vocals(audio_path, amount=float(amount))
    if humanized_path is None:
        raise RuntimeError("Vocal humanization failed")
    converted_paths = _convert_audio_outputs(
        [humanized_path],
        normalize_audio_format_choice(format_choice),
    )
    _report_audio_activity(
        "Publish humanized vocals",
        detail="Humanized vocal output is ready.",
        completed=2,
        total=2,
    )
    return converted_paths[0]


def run_remove_silence_tool(audio_path: str, format_choice: str) -> str:
    from definers.audio import remove_silence

    _report_audio_activity(
        "Remove silence",
        detail="Applying silence-trimming to the audio.",
        completed=1,
        total=2,
    )
    output_path = _temp_audio_output_path(format_choice)
    cleaned_path = remove_silence(audio_path, output_path)
    if cleaned_path is None:
        raise RuntimeError("Silence removal failed")
    _report_audio_activity(
        "Publish cleaned audio",
        detail="Silence-reduced audio is ready.",
        completed=2,
        total=2,
    )
    return cleaned_path


def run_compact_audio_tool(audio_path: str, format_choice: str) -> str:
    from definers.audio import compact_audio

    _report_audio_activity(
        "Compact audio",
        detail="Rendering a lighter delivery version of the audio.",
        completed=1,
        total=2,
    )
    output_path = _temp_audio_output_path(format_choice)
    compacted_path = compact_audio(audio_path, output_path)
    if compacted_path is None:
        raise RuntimeError("Audio compaction failed")
    _report_audio_activity(
        "Publish compacted audio",
        detail="Compacted audio is ready.",
        completed=2,
        total=2,
    )
    return compacted_path


def run_audio_preview_tool(
    audio_path: str,
    max_duration: float,
    format_choice: str,
) -> tuple[str, str]:
    from definers.audio import audio_preview, get_audio_duration
    from definers.system.output_paths import managed_output_session_dir

    _report_audio_activity(
        "Prepare preview request",
        detail="Resolving preview duration and output directory.",
        completed=1,
        total=3,
    )
    preview_kwargs = {"max_duration": float(max_duration)}
    if _supports_keyword_argument(audio_preview, "output_folder"):
        preview_kwargs["output_folder"] = managed_output_session_dir(
            "audio/preview",
            stem=Path(audio_path).stem,
        )
    preview_path = audio_preview(audio_path, **preview_kwargs)
    if preview_path is None:
        raise RuntimeError("Preview generation failed")
    _report_audio_activity(
        "Convert preview clip",
        detail="Converting the preview to the requested delivery format.",
        completed=2,
        total=3,
    )
    converted_paths = _convert_audio_outputs(
        [preview_path],
        normalize_audio_format_choice(format_choice),
    )
    source_duration = get_audio_duration(audio_path)
    preview_duration = get_audio_duration(converted_paths[0])
    summary = "\n\n".join(
        [
            f"**Source Duration:** {_format_metric_value(source_duration, ' s')}",
            f"**Preview Duration:** {_format_metric_value(preview_duration, ' s')}",
        ]
    )
    _report_audio_activity(
        "Publish preview clip",
        detail="Preview clip and duration summary are ready.",
        completed=3,
        total=3,
    )
    return converted_paths[0], summary


def run_split_audio_tool(
    audio_path: str,
    chunk_duration: float,
    format_choice: str,
    chunks_limit: float | int | None,
    skip_time: float,
    target_sample_rate: float | int | None,
) -> tuple[str | None, list[str], str]:
    from definers.audio import split_audio
    from definers.system.output_paths import managed_output_session_dir

    _report_audio_activity(
        "Prepare split request",
        detail="Resolving chunk options and output directory.",
        completed=1,
        total=2,
    )
    output_dir = managed_output_session_dir(
        "audio/split",
        stem=Path(audio_path).stem,
    )
    outputs = split_audio(
        audio_path,
        chunk_duration=float(chunk_duration),
        audio_format=normalize_audio_format_choice(format_choice),
        chunks_limit=_coerce_optional_int(chunks_limit),
        skip_time=float(skip_time),
        target_sample_rate=_coerce_optional_int(target_sample_rate),
        output_folder=output_dir,
    )
    if not outputs:
        raise RuntimeError("Audio splitting failed")
    _report_audio_activity(
        "Publish split outputs",
        detail="Chunk files and preview are ready.",
        completed=2,
        total=2,
    )
    summary = "\n\n".join(
        [
            f"**Chunks Created:** {len(outputs)}",
            f"**Chunk Duration:** {_format_metric_value(chunk_duration, ' s')}",
            f"**Skip Offset:** {_format_metric_value(skip_time, ' s')}",
        ]
    )
    return outputs[0], outputs, summary


def run_audio_analysis_tool(
    audio_path: str,
    hop_length: int,
    duration: float | int | None,
    offset: float,
) -> tuple[str, str, str]:
    from definers.runtime_numpy import get_numpy_module

    np = get_numpy_module()

    from definers.audio import analyze_audio, analyze_audio_features

    _report_audio_activity(
        "Analyze track",
        detail="Computing tempo, key, and spectrum metrics.",
        completed=1,
        total=2,
    )
    resolved_duration = _coerce_optional_float(duration)
    payload = analyze_audio(
        audio_path,
        hop_length=max(int(hop_length), 128),
        duration=resolved_duration,
        offset=float(offset),
    )
    summary_text = analyze_audio_features(audio_path) or "Unavailable"
    analysis_payload = {
        "summary": summary_text,
        "sample_rate_hz": int(payload["sr"]),
        "duration_seconds": float(payload["duration"]),
        "hop_length": int(payload["hop_length"]),
        "bpm": int(payload["bpm"]),
        "beat_count": int(len(payload["beat_frames"])),
        "offset_seconds": float(offset),
        "requested_window_seconds": resolved_duration,
        "spectral_centroid_hz": {
            "mean": float(np.mean(payload["spectral_centroid"])),
            "max": float(np.max(payload["spectral_centroid"])),
        },
        "rms": {
            "mean": float(np.mean(payload["rms"])),
            "low_mean": float(np.mean(payload["rms_low"])),
            "mid_mean": float(np.mean(payload["rms_mid"])),
            "high_mean": float(np.mean(payload["rms_high"])),
        },
    }
    summary_markdown = "\n\n".join(
        [
            f"**Key / Tempo:** {summary_text}",
            f"**Duration:** {_format_metric_value(analysis_payload['duration_seconds'], ' s')}",
            f"**Sample Rate:** {analysis_payload['sample_rate_hz']} Hz",
            f"**Beat Count:** {analysis_payload['beat_count']}",
            f"**Average Spectral Centroid:** {_format_metric_value(analysis_payload['spectral_centroid_hz']['mean'], ' Hz')}",
            f"**Average RMS:** {_format_metric_value(analysis_payload['rms']['mean'])}",
            f"**Low / Mid / High RMS:** {analysis_payload['rms']['low_mean']:.4f} / {analysis_payload['rms']['mid_mean']:.4f} / {analysis_payload['rms']['high_mean']:.4f}",
        ]
    )
    _report_audio_activity(
        "Publish analysis report",
        detail="Analysis summary and JSON payload are ready.",
        completed=2,
        total=2,
    )
    return summary_text, summary_markdown, _write_json_payload(analysis_payload)


def launch_audio_app(
    tool_names=None,
    *,
    app_title="Definers Audio",
    hero_eyebrow="Production Workspace",
    hero_description=(
        "Master, clean, split, generate, and publish audio from one focused"
        " workspace."
    ),
    navigation_label="Choose Workflow",
):

    from html import escape

    from PySide6.QtWidgets import QPushButton

    from definers.audio import (
        audio_to_midi,
        beat_visualizer,
        change_audio_speed,
        create_spectrum_visualization,
        dj_mix,
        extend_audio,
        generate_music,
        generate_voice,
        get_audio_feedback,
        identify_instruments,
        midi_to_audio,
        pitch_shift_vocals,
        stem_mixer,
        transcribe_audio,
    )
    from definers.constants import language_codes
    from definers.ml import convert_vocal_rvc
    from definers.ui.lyric_video_service import lyric_video
    from definers.ui.music_video_service import music_video
    from definers.ui.qt_widgets import (
        AudioOutput,
        FileOutput,
        FilePicker,
        HeroCard,
        ImageInput,
        LabeledCheckbox,
        LabeledCombo,
        LabeledFloatSlider,
        LabeledLineEdit,
        LabeledSlider,
        LabeledTextbox,
        MarkdownLabel,
        MediaOutput,
        Row,
        Section,
        VideoOutput,
        build_main_window,
        build_tab_widget,
    )

    qt_app()
    prepare_audio_workspace()

    available_tool_names = tuple(AUDIO_TOOL_MAP)
    selected_tool_names = tuple(tool_names or available_tool_names)
    tool_map = {
        tool_name: AUDIO_TOOL_MAP[tool_name]
        for tool_name in selected_tool_names
        if tool_name in AUDIO_TOOL_MAP
    }
    if not tool_map:
        tool_map = {
            tool_name: AUDIO_TOOL_MAP[tool_name]
            for tool_name in available_tool_names
        }
    format_choices = list(AUDIO_FORMAT_CHOICES)
    language_choices = list(get_audio_language_choices(language_codes))

    body = Section()
    hero_label = HeroCard(
        f"<div>"
        f'<p style="color:#bdf6ff;font-size:11px;font-weight:700;'
        f'letter-spacing:3px;margin:0 0 10px 0;">'
        f"{escape(hero_eyebrow)}</p>"
        f'<h1 style="color:#f9fcff;font-size:28px;font-weight:700;'
        f'margin:0 0 10px 0;line-height:1.15;">{escape(app_title)}</h1>'
        f'<p style="color:#c8d3e2;font-size:14px;line-height:1.55;'
        f'margin:0 0 14px 0;">{escape(hero_description)}</p>'
        f'<p style="color:#8d9caf;font-size:11px;font-weight:700;'
        f'letter-spacing:2px;margin:0;">'
        f"{escape(navigation_label)}</p>"
        f"</div>"
    )
    body.addWidget(hero_label)
    audio_progress = init_progress_tracker(
        "Audio workspace ready",
        "Choose a tool from the tabs and run it.",
    )
    body.addWidget(audio_progress)
    _, output_toolbar = init_output_folder_controls(section="audio")
    body.addWidget(output_toolbar)

    tabs = build_tab_widget()
    body.addWidget(tabs)

    def _bind(
        button,
        handler,
        *,
        inputs=None,
        outputs=None,
        action_label,
        steps=None,
        running_detail=None,
        success_detail=None,
    ):
        return bind_progress_click(
            button,
            handler,
            progress_output=audio_progress,
            inputs=inputs,
            outputs=outputs,
            action_label=action_label,
            steps=steps,
            running_detail=running_detail,
            success_detail=success_detail,
        )

    def _add_tab(panel_section, tool_key):
        for display_name, key in AUDIO_TOOL_MAP.items():
            if key == tool_key and tool_key in tool_map.values():
                tabs.addTab(panel_section, display_name)
                return True
        return False

    if "enhancer" in tool_map.values():
        panel = Section()
        row = Row()
        left = Section()
        enhancer_input = FilePicker("Source Audio")
        enhancer_format = LabeledCombo(
            "Output Format", choices=format_choices, value=format_choices[0]
        )
        enhancer_preset = LabeledCombo(
            "Mastering Profile",
            choices=list(MASTERING_PROFILE_CHOICES),
            value=MASTERING_PROFILE_CHOICES[0],
        )
        enhancer_bass = LabeledFloatSlider(
            "Bass Boost", minimum=0.0, maximum=12.0, step=0.5, value=0.0
        )
        enhancer_volume = LabeledFloatSlider(
            "Volume Trim (dB)",
            minimum=-12.0,
            maximum=12.0,
            step=0.5,
            value=0.0,
        )
        enhancer_effects = LabeledCheckbox("Apply effects chain", value=True)
        enhancer_stem_mastering = LabeledCheckbox(
            "Use stem-aware mastering", value=False
        )
        enhancer_stem_strategy = LabeledCombo(
            "Stem Model Strategy",
            choices=list(STEM_MODEL_STRATEGY_CHOICES),
            value=STEM_MODEL_STRATEGY_CHOICES[0],
        )
        enhancer_custom_stem_model = LabeledLineEdit(
            "Custom Stem Model", placeholder="checkpoint path or model id"
        )
        enhancer_stem_shifts = LabeledSlider(
            "Stem Shifts", minimum=0, maximum=8, step=1, value=1
        )
        enhancer_stem_mix_headroom = LabeledFloatSlider(
            "Stem Mix Headroom (dB)",
            minimum=-6.0,
            maximum=6.0,
            step=0.5,
            value=0.0,
        )
        enhancer_save_mastered_stems = LabeledCheckbox(
            "Save mastered stems", value=False
        )
        enhancer_stem_glue_reverb_amount = LabeledFloatSlider(
            "Stem Glue Reverb",
            minimum=0.0,
            maximum=10.0,
            step=0.5,
            value=STEM_GLUE_REVERB_DEFAULT,
        )
        enhancer_stem_drum_edge_amount = LabeledFloatSlider(
            "Stem Drum Edge",
            minimum=0.0,
            maximum=10.0,
            step=0.5,
            value=STEM_DRUM_EDGE_DEFAULT,
        )
        enhancer_stem_vocal_pullback_db = LabeledFloatSlider(
            "Stem Vocal Pullback (dB)",
            minimum=-12.0,
            maximum=0.0,
            step=0.5,
            value=STEM_VOCAL_PULLBACK_DB_DEFAULT,
        )
        enhancer_btn = QPushButton("Master Track")
        enhancer_btn.setProperty("definersPrimary", True)
        for widget in (
            enhancer_input,
            enhancer_format,
            enhancer_preset,
            enhancer_bass,
            enhancer_volume,
            enhancer_effects,
            enhancer_stem_mastering,
            enhancer_stem_strategy,
            enhancer_custom_stem_model,
            enhancer_stem_shifts,
            enhancer_stem_mix_headroom,
            enhancer_save_mastered_stems,
            enhancer_stem_glue_reverb_amount,
            enhancer_stem_drum_edge_amount,
            enhancer_stem_vocal_pullback_db,
            enhancer_btn,
        ):
            left.addWidget(widget)

        right = Section()
        enhancer_output = AudioOutput("Mastered Audio")
        enhancer_report = MarkdownLabel("## Mastering\n- Status: idle")
        enhancer_stems_output = FileOutput("Mastered Stems Archive")
        right.addWidget(enhancer_output)
        right.addWidget(enhancer_report)
        right.addWidget(enhancer_stems_output)

        row.addWidget(left)
        row.addWidget(right)
        panel.addWidget(row)
        _add_tab(panel, "enhancer")

        _bind(
            enhancer_btn,
            run_mastering_tool,
            inputs=[
                enhancer_input,
                enhancer_format,
                enhancer_preset,
                enhancer_bass,
                enhancer_volume,
                enhancer_effects,
                enhancer_stem_mastering,
                enhancer_stem_strategy,
                enhancer_stem_shifts,
                enhancer_stem_mix_headroom,
                enhancer_save_mastered_stems,
                enhancer_custom_stem_model,
                enhancer_stem_glue_reverb_amount,
                enhancer_stem_drum_edge_amount,
                enhancer_stem_vocal_pullback_db,
            ],
            outputs=[enhancer_output, enhancer_report, enhancer_stems_output],
            action_label="Master Track",
            steps=(
                "Validate input",
                "Analyze source",
                "Render master",
                "Publish output",
            ),
            running_detail="Mastering the track.",
            success_detail="Mastered audio is ready.",
        )

    def _simple_audio_tab(
        tool_key,
        *,
        action_label,
        handler,
        extra_widgets=(),
        include_format=True,
        output_widget=None,
        report_message="- Status: idle",
        running_detail=None,
        success_detail=None,
        steps=None,
    ):
        if tool_key not in tool_map.values():
            return
        panel = Section()
        row = Row()
        left = Section()
        source = FilePicker("Source Audio")
        left.addWidget(source)
        widget_inputs = [source]
        for widget in extra_widgets:
            left.addWidget(widget)
            widget_inputs.append(widget)
        if include_format:
            fmt = LabeledCombo(
                "Output Format",
                choices=format_choices,
                value=format_choices[0],
            )
            left.addWidget(fmt)
            widget_inputs.append(fmt)
        button = QPushButton(action_label)
        button.setProperty("definersPrimary", True)
        left.addWidget(button)

        right = Section()
        out = (
            output_widget
            if output_widget is not None
            else AudioOutput("Output")
        )
        right.addWidget(out)
        status = MarkdownLabel(f"## {action_label}\n{report_message}")
        right.addWidget(status)

        row.addWidget(left)
        row.addWidget(right)
        panel.addWidget(row)
        _add_tab(panel, tool_key)
        _bind(
            button,
            handler,
            inputs=widget_inputs,
            outputs=[out, status],
            action_label=action_label,
            steps=steps,
            running_detail=running_detail or f"Running {action_label}.",
            success_detail=success_detail or f"{action_label} is complete.",
        )

    _simple_audio_tab(
        "vocal_finish",
        action_label="Polish Vocals",
        handler=run_autotune_song_tool,
        extra_widgets=(
            LabeledFloatSlider(
                "Autotune Strength",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.6,
            ),
            LabeledCheckbox("Correct Timing", value=True),
            LabeledCheckbox("Quantize Notes", value=False),
            LabeledFloatSlider(
                "Tolerance (cents)",
                minimum=0.0,
                maximum=50.0,
                step=1.0,
                value=20.0,
            ),
            LabeledFloatSlider(
                "Attack (ms)",
                minimum=0.0,
                maximum=500.0,
                step=5.0,
                value=80.0,
            ),
        ),
        steps=("Validate source", "Apply vocal polish", "Publish output"),
    )

    _simple_audio_tab(
        "cleanup",
        action_label="Remove Silence",
        handler=run_remove_silence_tool,
        steps=("Validate source", "Trim silence", "Publish output"),
    )

    if "preview_split" in tool_map.values():
        panel = Section()
        row = Row()
        left = Section()
        preview_input = FilePicker("Source Audio")
        preview_duration = LabeledSlider(
            "Preview Duration (s)",
            minimum=5,
            maximum=120,
            step=5,
            value=30,
        )
        preview_format = LabeledCombo(
            "Output Format", choices=format_choices, value=format_choices[0]
        )
        split_chunks = LabeledSlider(
            "Split Chunks", minimum=2, maximum=12, step=1, value=4
        )
        preview_btn = QPushButton("Create Preview")
        split_btn = QPushButton("Split Audio")
        preview_btn.setProperty("definersPrimary", True)
        for widget in (
            preview_input,
            preview_duration,
            preview_format,
            split_chunks,
            preview_btn,
            split_btn,
        ):
            left.addWidget(widget)
        right = Section()
        preview_output = AudioOutput("Preview")
        preview_status = MarkdownLabel("## Preview & Split\n- Status: idle")
        split_output = FileOutput("Split Archive")
        right.addWidget(preview_output)
        right.addWidget(preview_status)
        right.addWidget(split_output)
        row.addWidget(left)
        row.addWidget(right)
        panel.addWidget(row)
        _add_tab(panel, "preview_split")
        _bind(
            preview_btn,
            run_audio_preview_tool,
            inputs=[preview_input, preview_duration, preview_format],
            outputs=[preview_output, preview_status],
            action_label="Create Preview",
            steps=("Validate source", "Build preview clip", "Publish output"),
            running_detail="Building the preview clip.",
            success_detail="Preview clip is ready.",
        )
        _bind(
            split_btn,
            run_split_audio_tool,
            inputs=[preview_input, split_chunks, preview_format],
            outputs=[split_output, preview_status],
            action_label="Split Audio",
            steps=("Validate source", "Split audio", "Publish archive"),
            running_detail="Splitting the audio into chunks.",
            success_detail="Audio split archive is ready.",
        )

    if "midi_tools" in tool_map.values():
        panel = Section()
        row = Row()
        left = Section()
        a2m_input = FilePicker("Source Audio")
        a2m_btn = QPushButton("Audio To MIDI")
        m2a_input = FilePicker("Source MIDI")
        m2a_format = LabeledCombo(
            "Output Format", choices=format_choices, value=format_choices[0]
        )
        m2a_btn = QPushButton("MIDI To Audio")
        for widget in (a2m_input, a2m_btn, m2a_input, m2a_format, m2a_btn):
            left.addWidget(widget)
        right = Section()
        a2m_output = FileOutput("MIDI Output")
        m2a_output = AudioOutput("Audio Output")
        midi_status = MarkdownLabel("## MIDI Tools\n- Status: idle")
        right.addWidget(a2m_output)
        right.addWidget(m2a_output)
        right.addWidget(midi_status)
        row.addWidget(left)
        row.addWidget(right)
        panel.addWidget(row)
        _add_tab(panel, "midi_tools")
        _bind(
            a2m_btn,
            audio_to_midi,
            inputs=[a2m_input],
            outputs=[a2m_output, midi_status],
            action_label="Audio To MIDI",
            steps=("Validate source", "Transcribe MIDI", "Publish output"),
        )
        _bind(
            m2a_btn,
            midi_to_audio,
            inputs=[m2a_input, m2a_format],
            outputs=[m2a_output, midi_status],
            action_label="MIDI To Audio",
            steps=("Validate MIDI", "Synthesize audio", "Publish output"),
        )

    _simple_audio_tab(
        "audio_extender",
        action_label="Extend Audio",
        handler=extend_audio,
        extra_widgets=(
            LabeledSlider(
                "Target Duration (s)",
                minimum=5,
                maximum=600,
                step=5,
                value=60,
            ),
        ),
        steps=("Validate source", "Extend audio", "Publish output"),
    )

    if "stem_mixer" in tool_map.values():
        panel = Section()
        row = Row()
        left = Section()
        stem_mixer_files = FilePicker("Stem Files")
        stem_mixer_format = LabeledCombo(
            "Output Format", choices=format_choices, value=format_choices[0]
        )
        stem_mixer_btn = QPushButton("Mix Stems")
        stem_mixer_btn.setProperty("definersPrimary", True)
        left.addWidget(stem_mixer_files)
        left.addWidget(stem_mixer_format)
        left.addWidget(stem_mixer_btn)
        right = Section()
        stem_mixer_output = AudioOutput("Mix")
        stem_mixer_status = MarkdownLabel("## Stem Mixer\n- Status: idle")
        right.addWidget(stem_mixer_output)
        right.addWidget(stem_mixer_status)
        row.addWidget(left)
        row.addWidget(right)
        panel.addWidget(row)
        _add_tab(panel, "stem_mixer")
        _bind(
            stem_mixer_btn,
            stem_mixer,
            inputs=[stem_mixer_files, stem_mixer_format],
            outputs=[stem_mixer_output, stem_mixer_status],
            action_label="Mix Stems",
            steps=("Validate stems", "Mix stems", "Publish output"),
        )

    if "feedback" in tool_map.values():
        panel = Section()
        row = Row()
        left = Section()
        feedback_input = FilePicker("Source Audio")
        feedback_btn = QPushButton("Generate Feedback")
        feedback_btn.setProperty("definersPrimary", True)
        left.addWidget(feedback_input)
        left.addWidget(feedback_btn)
        right = Section()
        feedback_output = MarkdownLabel("## Track Feedback\n- Status: idle")
        right.addWidget(feedback_output)
        row.addWidget(left)
        row.addWidget(right)
        panel.addWidget(row)
        _add_tab(panel, "feedback")
        _bind(
            feedback_btn,
            get_audio_feedback,
            inputs=[feedback_input],
            outputs=[feedback_output],
            action_label="Generate Feedback",
            steps=("Validate source", "Analyze", "Publish feedback"),
        )

    if "instrument_id" in tool_map.values():
        panel = Section()
        row = Row()
        left = Section()
        instr_input = FilePicker("Source Audio")
        instr_btn = QPushButton("Identify Instruments")
        instr_btn.setProperty("definersPrimary", True)
        left.addWidget(instr_input)
        left.addWidget(instr_btn)
        right = Section()
        instr_output = MarkdownLabel("## Instrument ID\n- Status: idle")
        right.addWidget(instr_output)
        row.addWidget(left)
        row.addWidget(right)
        panel.addWidget(row)
        _add_tab(panel, "instrument_id")
        _bind(
            instr_btn,
            identify_instruments,
            inputs=[instr_input],
            outputs=[instr_output],
            action_label="Identify Instruments",
            steps=("Validate source", "Identify", "Publish results"),
        )

    if "video_gen" in tool_map.values():
        panel = Section()
        row = Row()
        left = Section()
        clip_audio = FilePicker("Audio")
        clip_image = ImageInput("Cover Image")
        clip_btn = QPushButton("Build Music Video")
        clip_btn.setProperty("definersPrimary", True)
        left.addWidget(clip_audio)
        left.addWidget(clip_image)
        left.addWidget(clip_btn)
        right = Section()
        clip_output = VideoOutput("Video")
        clip_status = MarkdownLabel("## Music Clip\n- Status: idle")
        right.addWidget(clip_output)
        right.addWidget(clip_status)
        row.addWidget(left)
        row.addWidget(right)
        panel.addWidget(row)
        _add_tab(panel, "video_gen")
        _bind(
            clip_btn,
            music_video,
            inputs=[clip_audio, clip_image],
            outputs=[clip_output, clip_status],
            action_label="Build Music Video",
            steps=("Validate inputs", "Render video", "Publish output"),
        )

    if "speed" in tool_map.values():
        panel = Section()
        row = Row()
        left = Section()
        speed_input = FilePicker("Source Audio")
        speed_value = LabeledFloatSlider(
            "Speed Factor",
            minimum=0.25,
            maximum=4.0,
            step=0.05,
            value=1.0,
        )
        speed_format = LabeledCombo(
            "Output Format", choices=format_choices, value=format_choices[0]
        )
        speed_btn = QPushButton("Change Speed")
        speed_btn.setProperty("definersPrimary", True)
        for widget in (speed_input, speed_value, speed_format, speed_btn):
            left.addWidget(widget)
        right = Section()
        speed_output = AudioOutput("Output")
        speed_status = MarkdownLabel("## Speed & Pitch\n- Status: idle")
        right.addWidget(speed_output)
        right.addWidget(speed_status)
        row.addWidget(left)
        row.addWidget(right)
        panel.addWidget(row)
        _add_tab(panel, "speed")
        _bind(
            speed_btn,
            change_audio_speed,
            inputs=[speed_input, speed_value, speed_format],
            outputs=[speed_output, speed_status],
            action_label="Change Speed",
            steps=("Validate source", "Resample", "Publish output"),
        )

    _simple_audio_tab(
        "stem",
        action_label="Separate Stems",
        handler=run_stem_separation_tool,
        steps=("Validate source", "Separate", "Publish stems"),
        output_widget=FileOutput("Stems Archive"),
    )

    if "vps" in tool_map.values():
        panel = Section()
        row = Row()
        left = Section()
        vps_input = FilePicker("Source Audio")
        vps_semitones = LabeledFloatSlider(
            "Pitch Shift (semitones)",
            minimum=-24.0,
            maximum=24.0,
            step=0.5,
            value=0.0,
        )
        vps_format = LabeledCombo(
            "Output Format", choices=format_choices, value=format_choices[0]
        )
        vps_btn = QPushButton("Shift Pitch")
        vps_btn.setProperty("definersPrimary", True)
        for widget in (vps_input, vps_semitones, vps_format, vps_btn):
            left.addWidget(widget)
        right = Section()
        vps_output = AudioOutput("Output")
        vps_status = MarkdownLabel("## Vocal Pitch Shifter\n- Status: idle")
        right.addWidget(vps_output)
        right.addWidget(vps_status)
        row.addWidget(left)
        row.addWidget(right)
        panel.addWidget(row)
        _add_tab(panel, "vps")
        _bind(
            vps_btn,
            pitch_shift_vocals,
            inputs=[vps_input, vps_semitones, vps_format],
            outputs=[vps_output, vps_status],
            action_label="Shift Pitch",
            steps=("Validate source", "Shift pitch", "Publish output"),
        )

    if "voice_lab" in tool_map.values():
        panel = Section()
        row = Row()
        left = Section()
        vlab_dataset = FilePicker("Voice Dataset Archive")
        vlab_model = LabeledLineEdit(
            "Model Name", placeholder="custom_voice_model"
        )
        vlab_btn = QPushButton("Train Voice Lab Model")
        vlab_btn.setProperty("definersPrimary", True)
        for widget in (vlab_dataset, vlab_model, vlab_btn):
            left.addWidget(widget)
        right = Section()
        vlab_status = MarkdownLabel("## Voice Lab\n- Status: idle")
        right.addWidget(vlab_status)
        row.addWidget(left)
        row.addWidget(right)
        panel.addWidget(row)
        _add_tab(panel, "voice_lab")
        _bind(
            vlab_btn,
            train_voice_lab_model,
            inputs=[vlab_dataset, vlab_model],
            outputs=[vlab_status],
            action_label="Train Voice Lab Model",
            steps=("Validate dataset", "Train model", "Publish artifact"),
        )

    if "dj" in tool_map.values():
        panel = Section()
        row = Row()
        left = Section()
        dj_files = FilePicker("Source Tracks")
        dj_format = LabeledCombo(
            "Output Format", choices=format_choices, value=format_choices[0]
        )
        dj_btn = QPushButton("Build DJ Mix")
        dj_btn.setProperty("definersPrimary", True)
        for widget in (dj_files, dj_format, dj_btn):
            left.addWidget(widget)
        right = Section()
        dj_output = AudioOutput("Mix")
        dj_status = MarkdownLabel("## DJ AutoMix\n- Status: idle")
        right.addWidget(dj_output)
        right.addWidget(dj_status)
        row.addWidget(left)
        row.addWidget(right)
        panel.addWidget(row)
        _add_tab(panel, "dj")
        _bind(
            dj_btn,
            dj_mix,
            inputs=[dj_files, dj_format],
            outputs=[dj_output, dj_status],
            action_label="Build DJ Mix",
            steps=("Validate tracks", "Mix", "Publish output"),
        )

    if "music_gen" in tool_map.values():
        panel = Section()
        row = Row()
        left = Section()
        mg_prompt = LabeledTextbox("Prompt", lines=5)
        mg_duration = LabeledSlider(
            "Duration (s)", minimum=5, maximum=300, step=5, value=30
        )
        mg_format = LabeledCombo(
            "Output Format", choices=format_choices, value=format_choices[0]
        )
        mg_btn = QPushButton("Generate Music")
        mg_btn.setProperty("definersPrimary", True)
        for widget in (mg_prompt, mg_duration, mg_format, mg_btn):
            left.addWidget(widget)
        right = Section()
        mg_output = AudioOutput("Music")
        mg_status = MarkdownLabel("## Music Gen\n- Status: idle")
        right.addWidget(mg_output)
        right.addWidget(mg_status)
        row.addWidget(left)
        row.addWidget(right)
        panel.addWidget(row)
        _add_tab(panel, "music_gen")
        _bind(
            mg_btn,
            generate_music,
            inputs=[mg_prompt, mg_duration, mg_format],
            outputs=[mg_output, mg_status],
            action_label="Generate Music",
            steps=("Validate prompt", "Generate", "Publish output"),
        )

    if "voice_gen" in tool_map.values():
        panel = Section()
        row = Row()
        left = Section()
        vg_text = LabeledTextbox("Text", lines=5)
        default_lang = language_choices[0] if language_choices else "en"
        vg_language = LabeledCombo(
            "Language", choices=language_choices, value=default_lang
        )
        vg_format = LabeledCombo(
            "Output Format", choices=format_choices, value=format_choices[0]
        )
        vg_btn = QPushButton("Generate Voice")
        vg_btn.setProperty("definersPrimary", True)
        for widget in (vg_text, vg_language, vg_format, vg_btn):
            left.addWidget(widget)
        right = Section()
        vg_output = AudioOutput("Voice")
        vg_status = MarkdownLabel("## Voice Gen\n- Status: idle")
        right.addWidget(vg_output)
        right.addWidget(vg_status)
        row.addWidget(left)
        row.addWidget(right)
        panel.addWidget(row)
        _add_tab(panel, "voice_gen")
        _bind(
            vg_btn,
            generate_voice,
            inputs=[vg_text, vg_language, vg_format],
            outputs=[vg_output, vg_status],
            action_label="Generate Voice",
            steps=("Validate text", "Synthesize", "Publish output"),
        )

    if "analysis" in tool_map.values():
        panel = Section()
        row = Row()
        left = Section()
        analysis_input = FilePicker("Source Audio")
        analysis_btn = QPushButton("Analyze Audio")
        analysis_btn.setProperty("definersPrimary", True)
        left.addWidget(analysis_input)
        left.addWidget(analysis_btn)
        right = Section()
        analysis_bpm = LabeledLineEdit("BPM / Key", placeholder="Ã¢â‚¬â€")
        analysis_summary = MarkdownLabel("## Analysis\n- Status: idle")
        analysis_json = FileOutput("Analysis JSON")
        right.addWidget(analysis_bpm)
        right.addWidget(analysis_summary)
        right.addWidget(analysis_json)
        row.addWidget(left)
        row.addWidget(right)
        panel.addWidget(row)
        _add_tab(panel, "analysis")
        _bind(
            analysis_btn,
            run_audio_analysis_tool,
            inputs=[analysis_input],
            outputs=[analysis_bpm, analysis_summary, analysis_json],
            action_label="Analyze Audio",
            steps=("Validate source", "Analyze", "Publish report"),
        )

    if "stt" in tool_map.values():
        panel = Section()
        row = Row()
        left = Section()
        stt_input = FilePicker("Source Audio")
        default_lang = language_choices[0] if language_choices else "en"
        stt_language = LabeledCombo(
            "Language", choices=language_choices, value=default_lang
        )
        stt_btn = QPushButton("Transcribe")
        stt_btn.setProperty("definersPrimary", True)
        for widget in (stt_input, stt_language, stt_btn):
            left.addWidget(widget)
        right = Section()
        stt_output = LabeledTextbox("Transcript", lines=8)
        stt_file = FileOutput("Transcript File")
        right.addWidget(stt_output)
        right.addWidget(stt_file)
        row.addWidget(left)
        row.addWidget(right)
        panel.addWidget(row)
        _add_tab(panel, "stt")
        _bind(
            stt_btn,
            transcribe_audio,
            inputs=[stt_input, stt_language],
            outputs=[stt_output, stt_file],
            action_label="Transcribe",
            steps=("Validate source", "Transcribe", "Publish output"),
        )

    if "spectrum" in tool_map.values():
        panel = Section()
        row = Row()
        left = Section()
        spec_input = FilePicker("Source Audio")
        spec_btn = QPushButton("Render Spectrum")
        spec_btn.setProperty("definersPrimary", True)
        left.addWidget(spec_input)
        left.addWidget(spec_btn)
        right = Section()
        spec_output = VideoOutput("Spectrum")
        spec_status = MarkdownLabel("## Spectrum\n- Status: idle")
        right.addWidget(spec_output)
        right.addWidget(spec_status)
        row.addWidget(left)
        row.addWidget(right)
        panel.addWidget(row)
        _add_tab(panel, "spectrum")
        _bind(
            spec_btn,
            create_spectrum_visualization,
            inputs=[spec_input],
            outputs=[spec_output, spec_status],
            action_label="Render Spectrum",
            steps=("Validate source", "Render spectrum", "Publish output"),
        )

    if "beat_vis" in tool_map.values():
        panel = Section()
        row = Row()
        left = Section()
        vis_image_input = ImageInput("Reference Image")
        vis_audio_input = FilePicker("Audio")
        vis_btn = QPushButton("Render Beat Visualizer")
        vis_btn.setProperty("definersPrimary", True)
        for widget in (vis_image_input, vis_audio_input, vis_btn):
            left.addWidget(widget)
        right = Section()
        vis_output = VideoOutput("Visualizer")
        vis_status = MarkdownLabel("## Beat Visualizer\n- Status: idle")
        right.addWidget(vis_output)
        right.addWidget(vis_status)
        row.addWidget(left)
        row.addWidget(right)
        panel.addWidget(row)
        _add_tab(panel, "beat_vis")
        _bind(
            vis_btn,
            beat_visualizer,
            inputs=[vis_image_input, vis_audio_input],
            outputs=[vis_output, vis_status],
            action_label="Render Beat Visualizer",
            steps=("Validate inputs", "Render video", "Publish output"),
        )

    if "lyric_vid" in tool_map.values():
        panel = Section()
        row = Row()
        left = Section()
        lyric_audio = FilePicker("Audio")
        lyric_bg = ImageInput("Background")
        default_lang = language_choices[0] if language_choices else "en"
        lyric_language = LabeledCombo(
            "Language", choices=language_choices, value=default_lang
        )
        lyric_text = LabeledTextbox("Lyrics", lines=6)
        load_transcript_btn = QPushButton("Load Transcript")
        lyric_btn = QPushButton("Build Lyric Video")
        lyric_btn.setProperty("definersPrimary", True)
        for widget in (
            lyric_audio,
            lyric_bg,
            lyric_language,
            lyric_text,
            load_transcript_btn,
            lyric_btn,
        ):
            left.addWidget(widget)
        right = Section()
        lyric_output = VideoOutput("Lyric Video")
        lyric_status = MarkdownLabel("## Lyric Video\n- Status: idle")
        right.addWidget(lyric_output)
        right.addWidget(lyric_status)
        row.addWidget(left)
        row.addWidget(right)
        panel.addWidget(row)
        _add_tab(panel, "lyric_vid")
        _bind(
            load_transcript_btn,
            transcribe_audio,
            inputs=[lyric_audio, lyric_language],
            outputs=[lyric_text],
            action_label="Load Transcript",
            steps=("Validate audio", "Transcribe", "Publish transcript"),
        )
        _bind(
            lyric_btn,
            lyric_video,
            inputs=[lyric_audio, lyric_bg, lyric_text, lyric_language],
            outputs=[lyric_output, lyric_status],
            action_label="Build Lyric Video",
            steps=("Validate inputs", "Render lyric video", "Publish output"),
        )

    if "chatbot" in tool_map.values():
        from definers.ui.chat_handlers import get_chat_response_stream

        panel = Section()
        chat_widget = init_chat(
            "Definers Audio Assistant", get_chat_response_stream
        )
        panel.addWidget(chat_widget)
        _add_tab(panel, "chatbot")

    _ = (convert_vocal_rvc,)

    window = build_main_window(app_title, body)
    launch_blocks(window)


__all__ = [
    "AUDIO_ASSISTANT_DATA",
    "AUDIO_ASSISTANT_RULES",
    "AUDIO_FORMAT_CHOICES",
    "AUDIO_TOOL_MAP",
    "MASTERING_PROFILE_CHOICES",
    "STEM_MODEL_STRATEGY_CHOICES",
    "launch_audio_app",
    "prepare_audio_workspace",
]
