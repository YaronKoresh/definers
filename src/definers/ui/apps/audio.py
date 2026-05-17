from __future__ import annotations

import inspect
import json
from pathlib import Path
from shutil import copy2

from definers.ui.gradio_shared import (
    bind_progress_click,
    init_chat,
    init_output_folder_controls,
    init_progress_tracker,
    init_status_card,
    launch_blocks,
    progress_update,
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

STEM_GLUE_REVERB_DEFAULT = 1.0
STEM_DRUM_EDGE_DEFAULT = 1.0
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
    from definers.audio.stems import separate_stem_layers

    manifest = _read_mastering_job(job_dir)
    settings = _job_settings(manifest)
    if not bool(settings.get("stem_mastering")):
        return refresh_mastering_job(job_dir)

    input_path = str(_job_input(manifest).get("path", ""))
    raw_stems_dir = Path(str(job_dir)) / "raw_stems"
    raw_stems_dir.mkdir(parents=True, exist_ok=True)
    _report_audio_activity(
        "Separate stems",
        detail="Running the heavy stem-separation stage and saving raw layers.",
        completed=1,
        total=2,
    )
    stem_paths, resolved_output_dir = separate_stem_layers(
        input_path,
        model_name=str(settings.get("stem_model_name", "mastering")),
        shifts=max(int(settings.get("stem_shifts", 2)), 1),
        quality_flags=tuple(
            str(flag).strip()
            for flag in _job_analysis(manifest).get("quality_flags", ())
            if str(flag).strip()
        ),
        output_dir=str(raw_stems_dir),
    )
    manifest["artifacts"] = {
        **_job_artifacts(manifest),
        "raw_stems_dir": str(resolved_output_dir),
        "raw_stems": {
            str(stem_name): str(stem_path)
            for stem_name, stem_path in stem_paths.items()
        },
    }
    _report_audio_activity(
        "Publish raw stems",
        detail="Raw stems are ready for inspection and the mix-building step.",
        completed=2,
        total=2,
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
        str(_job_input(manifest).get("path", "")),
        base_config=base_mastering.config,
        base_mastering_kwargs=base_kwargs,
        process_stem_fn=_process_stem_signal,
        separate_stems_fn=reuse_existing_stems,
        read_audio_fn=read_audio,
        delete_fn=lambda _path: None,
        model_name=str(settings.get("stem_model_name", "mastering")),
        shifts=max(int(settings.get("stem_shifts", 2)), 1),
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
    hero_description="Master, clean, split, generate, and publish audio from one focused workspace.",
    default_tool=None,
    navigation_label="Choose Workflow",
):
    from html import escape

    import gradio as gr

    from definers.audio import (
        audio_to_midi,
        beat_visualizer,
        change_audio_speed,
        create_share_links,
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
    from definers.cuda import device
    from definers.file_ops import save_temp_text as save_text_to_file
    from definers.ml import (
        convert_vocal_rvc,
    )
    from definers.system.download_activity import (
        create_download_activity_task,
        get_download_activity_snapshot,
        resolve_download_activity_task,
        wait_for_download_activity_task,
    )
    from definers.text import random_string
    from definers.ui.lyric_video_service import lyric_video
    from definers.ui.music_video_service import music_video

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
    default_tool_name = (
        default_tool
        if isinstance(default_tool, str) and default_tool in tool_map
        else next(iter(tool_map))
    )
    format_choices = AUDIO_FORMAT_CHOICES
    language_choices = get_audio_language_choices(language_codes)
    initial_mastering_state = get_mastering_profile_ui_state(
        MASTERING_PROFILE_CHOICES[0]
    )
    initial_stem_strategy_note = describe_stem_model_choice(
        STEM_MODEL_STRATEGY_CHOICES[0]
    )

    with gr.Blocks(title=app_title) as app:
        gr.HTML(
            f"""<div id=\"header\" class=\"audio-hero\"><p class=\"eyebrow\">{escape(hero_eyebrow)}</p><h1>{escape(app_title)}</h1><p>{escape(hero_description)}</p></div>"""
        )

        with gr.Row(elem_id="nav-dropdown-wrapper"):
            nav_dropdown = gr.Dropdown(
                choices=list(tool_map.keys()),
                value=default_tool_name,
                label=navigation_label,
                elem_id="nav-dropdown",
                visible=len(tool_map) > 1,
            )

        audio_progress = init_progress_tracker(
            "Audio workspace ready",
            "Choose a workflow and start processing.",
        )
        init_output_folder_controls(section="audio")

        def bind_audio_action(
            button,
            handler,
            *,
            inputs=None,
            outputs=None,
            action_label,
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
                running_detail=running_detail,
                success_detail=success_detail,
            )

        with gr.Row(elem_id="main-row"):
            with gr.Column(scale=1, elem_id="main-content"):
                with gr.Group(
                    visible=True, elem_classes="tool-container"
                ) as view_enhancer:
                    gr.Markdown("## Mastering Studio")
                    with gr.Row():
                        with gr.Column():
                            enhancer_input = gr.Audio(
                                label="Upload Mix", type="filepath"
                            )
                            enhancer_format = gr.Radio(
                                format_choices,
                                label="Output Format",
                                value="MP3",
                            )
                            enhancer_preset = gr.Dropdown(
                                MASTERING_PROFILE_CHOICES,
                                label="Mastering Strategy",
                                value=str(initial_mastering_state["label"]),
                            )
                            enhancer_profile_note = gr.Markdown(
                                value=str(
                                    initial_mastering_state["description"]
                                )
                            )
                            enhancer_stem_mastering = gr.Checkbox(
                                label="Use stem-aware mastering (experimental)",
                                value=False,
                            )
                            with gr.Accordion("Macro Controls", open=False):
                                enhancer_macro_note = gr.Markdown(
                                    value=str(
                                        initial_mastering_state["macro_note"]
                                    )
                                )
                                enhancer_bass = gr.Slider(
                                    0.0,
                                    1.0,
                                    float(initial_mastering_state["bass"]),
                                    step=0.05,
                                    label="Bass",
                                    interactive=bool(
                                        initial_mastering_state[
                                            "controls_enabled"
                                        ]
                                    ),
                                )
                                enhancer_volume = gr.Slider(
                                    0.0,
                                    1.0,
                                    float(initial_mastering_state["volume"]),
                                    step=0.05,
                                    label="Volume",
                                    interactive=bool(
                                        initial_mastering_state[
                                            "controls_enabled"
                                        ]
                                    ),
                                )
                                enhancer_effects = gr.Slider(
                                    0.0,
                                    1.0,
                                    float(initial_mastering_state["effects"]),
                                    step=0.05,
                                    label="Effects",
                                    interactive=bool(
                                        initial_mastering_state[
                                            "controls_enabled"
                                        ]
                                    ),
                                )
                            with gr.Accordion("Stem-Aware Path", open=False):
                                with gr.Group(
                                    visible=False
                                ) as enhancer_stem_settings:
                                    enhancer_stem_strategy = gr.Dropdown(
                                        STEM_MODEL_STRATEGY_CHOICES,
                                        label="Stem Separation Strategy",
                                        value=STEM_MODEL_STRATEGY_CHOICES[0],
                                    )
                                    enhancer_custom_stem_model = gr.Textbox(
                                        label="Custom separator checkpoint",
                                        placeholder="Example: htdemucs_6s or custom_model.yaml",
                                        visible=False,
                                    )
                                    enhancer_stem_strategy_note = gr.Markdown(
                                        value=initial_stem_strategy_note
                                    )
                                    enhancer_stem_shifts = gr.Slider(
                                        1,
                                        8,
                                        2,
                                        step=1,
                                        label="Stem Separation Shifts",
                                    )
                                    enhancer_stem_mix_headroom = gr.Slider(
                                        3.0,
                                        12.0,
                                        6.0,
                                        step=0.5,
                                        label="Stem Mix Headroom (dB)",
                                    )
                                    enhancer_save_mastered_stems = gr.Checkbox(
                                        label="Save mastered stems",
                                        value=True,
                                    )
                                    enhancer_stem_glue_reverb_amount = (
                                        gr.Slider(
                                            0.0,
                                            1.5,
                                            STEM_GLUE_REVERB_DEFAULT,
                                            step=0.05,
                                            label="Vocal/Other Glue Reverb",
                                        )
                                    )
                                    enhancer_stem_drum_edge_amount = gr.Slider(
                                        0.0,
                                        1.5,
                                        STEM_DRUM_EDGE_DEFAULT,
                                        step=0.05,
                                        label="Drum Edge / Expand-Compress",
                                    )
                                    enhancer_stem_vocal_pullback_db = gr.Slider(
                                        0.0,
                                        3.0,
                                        STEM_VOCAL_PULLBACK_DB_DEFAULT,
                                        step=0.1,
                                        label="Extra Vocal Pullback (dB)",
                                    )
                            gr.Markdown(
                                "Auto Analyze chooses a mastering profile after reading the mix. Named profiles lock macro controls to avoid conflict, and Custom Macro Blend unlocks manual shaping."
                            )
                            with gr.Row():
                                enhancer_btn = gr.Button(
                                    "Master Audio", variant="primary"
                                )
                                clear_enhancer_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column(
                            scale=1,
                            visible=False,
                            elem_id="enhancer-output-column",
                        ) as enhancer_output_column:
                            with gr.Group(elem_id="enhancer-output-box"):
                                enhancer_output = gr.Audio(
                                    label="Mastered Audio",
                                    interactive=False,
                                    buttons=["download"],
                                )
                                enhancer_report = gr.File(
                                    label="Mastering Report",
                                    interactive=False,
                                    visible=False,
                                )
                                enhancer_stems_output = gr.File(
                                    label="Mastered Stems",
                                    interactive=False,
                                    file_count="multiple",
                                    visible=False,
                                )
                                with gr.Group(
                                    visible=False
                                ) as enhancer_stem_preview_box:
                                    gr.Markdown("### Mastered Stem Previews")
                                    with gr.Row():
                                        with gr.Column():
                                            enhancer_vocals_stem = gr.Audio(
                                                label="Vocals Stem",
                                                interactive=False,
                                                buttons=["download"],
                                                visible=False,
                                            )
                                            enhancer_bass_stem = gr.Audio(
                                                label="Bass Stem",
                                                interactive=False,
                                                buttons=["download"],
                                                visible=False,
                                            )
                                            enhancer_guitar_stem = gr.Audio(
                                                label="Guitar Stem",
                                                interactive=False,
                                                buttons=["download"],
                                                visible=False,
                                            )
                                        with gr.Column():
                                            enhancer_drums_stem = gr.Audio(
                                                label="Drums Stem",
                                                interactive=False,
                                                buttons=["download"],
                                                visible=False,
                                            )
                                            enhancer_other_stem = gr.Audio(
                                                label="Other Stem",
                                                interactive=False,
                                                buttons=["download"],
                                                visible=False,
                                            )
                                            enhancer_piano_stem = gr.Audio(
                                                label="Piano Stem",
                                                interactive=False,
                                                buttons=["download"],
                                                visible=False,
                                            )
                                enhancer_diagnostics = gr.Markdown()
                                enhancer_share_links = gr.Markdown()
                    with gr.Accordion(
                        "Run In Stages Or Resume",
                        open=False,
                    ):
                        gr.Markdown(
                            "Use the same mastering settings above to prepare a resumable job, run one stage at a time, or execute the full staged flow in one pass."
                        )
                        enhancer_job_status = init_status_card(
                            "Staged mastering ready",
                            "Prepare a job to enable resumable mastering stages.",
                        )
                        enhancer_job_dir = gr.Textbox(
                            label="Job Folder",
                            placeholder="Filled after Prepare Staged Job or paste an existing job folder to resume.",
                            interactive=True,
                        )
                        with gr.Row():
                            enhancer_job_prepare_btn = gr.Button(
                                "Prepare Staged Job"
                            )
                            enhancer_job_full_btn = gr.Button(
                                "Run Full Job",
                                variant="primary",
                            )
                            enhancer_job_separate_btn = gr.Button(
                                "Separate Stems"
                            )
                            enhancer_job_mix_btn = gr.Button("Build Stem Mix")
                            enhancer_job_finalize_btn = gr.Button(
                                "Finalize Master"
                            )
                            enhancer_job_refresh_btn = gr.Button("Refresh Job")
                        with gr.Row():
                            enhancer_job_raw_stems = gr.File(
                                label="Raw Stems",
                                interactive=False,
                                file_count="multiple",
                            )
                            enhancer_job_processed_stems = gr.File(
                                label="Processed Stems",
                                interactive=False,
                                file_count="multiple",
                            )
                        with gr.Row():
                            enhancer_job_mixed_audio = gr.Audio(
                                label="Stem Mix Preview",
                                interactive=False,
                                buttons=["download"],
                            )
                            enhancer_job_mastered_audio = gr.Audio(
                                label="Staged Final Master",
                                interactive=False,
                                buttons=["download"],
                            )
                        enhancer_job_report = gr.File(
                            label="Staged Mastering Report",
                            interactive=False,
                        )
                        enhancer_job_summary = gr.Markdown()
                        with gr.Accordion("Advanced Job Details", open=False):
                            enhancer_job_manifest = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_vocal_finish:
                    gr.Markdown("## Vocal Finishing")
                    with gr.Tabs():
                        with gr.TabItem("AutoTune"):
                            with gr.Row():
                                with gr.Column():
                                    autotune_input = gr.Audio(
                                        label="Upload Song", type="filepath"
                                    )
                                    autotune_format = gr.Radio(
                                        format_choices,
                                        label="Output Format",
                                        value="MP3",
                                    )
                                    autotune_strength = gr.Slider(
                                        0.0,
                                        1.0,
                                        0.7,
                                        step=0.05,
                                        label="Pitch Correction Strength",
                                    )
                                    autotune_correct_timing = gr.Checkbox(
                                        label="Correct timing against detected beat grid",
                                        value=True,
                                    )
                                    autotune_quantize = gr.Slider(
                                        4,
                                        32,
                                        16,
                                        step=4,
                                        label="Beat Grid Density",
                                    )
                                    autotune_tolerance = gr.Slider(
                                        0,
                                        50,
                                        15,
                                        step=1,
                                        label="Tolerance (cents)",
                                    )
                                    autotune_attack = gr.Slider(
                                        0.0,
                                        20.0,
                                        0.1,
                                        step=0.1,
                                        label="Attack Smoothing (ms)",
                                    )
                                    with gr.Row():
                                        autotune_btn = gr.Button(
                                            "AutoTune Song",
                                            variant="primary",
                                        )
                                        clear_autotune_btn = gr.Button(
                                            "Clear",
                                            variant="secondary",
                                        )
                                with gr.Column():
                                    with gr.Group(
                                        visible=False
                                    ) as autotune_output_box:
                                        autotune_output = gr.Audio(
                                            label="AutoTuned Song",
                                            interactive=False,
                                            buttons=["download"],
                                        )
                                        autotune_share_links = gr.Markdown()
                        with gr.TabItem("Humanize Vocals"):
                            with gr.Row():
                                with gr.Column():
                                    humanize_input = gr.Audio(
                                        label="Upload Vocal Take",
                                        type="filepath",
                                    )
                                    humanize_format = gr.Radio(
                                        format_choices,
                                        label="Output Format",
                                        value="MP3",
                                    )
                                    humanize_amount = gr.Slider(
                                        0.0,
                                        1.0,
                                        0.5,
                                        step=0.05,
                                        label="Variation Amount",
                                    )
                                    with gr.Row():
                                        humanize_btn = gr.Button(
                                            "Humanize Vocals",
                                            variant="primary",
                                        )
                                        clear_humanize_btn = gr.Button(
                                            "Clear",
                                            variant="secondary",
                                        )
                                with gr.Column():
                                    with gr.Group(
                                        visible=False
                                    ) as humanize_output_box:
                                        humanize_output = gr.Audio(
                                            label="Humanized Vocals",
                                            interactive=False,
                                            buttons=["download"],
                                        )
                                        humanize_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_cleanup:
                    gr.Markdown("## Audio Cleanup")
                    with gr.Tabs():
                        with gr.TabItem("Remove Silence"):
                            with gr.Row():
                                with gr.Column():
                                    silence_input = gr.Audio(
                                        label="Upload Audio", type="filepath"
                                    )
                                    silence_format = gr.Radio(
                                        format_choices,
                                        label="Output Format",
                                        value="MP3",
                                    )
                                    with gr.Row():
                                        silence_btn = gr.Button(
                                            "Remove Silence",
                                            variant="primary",
                                        )
                                        clear_silence_btn = gr.Button(
                                            "Clear",
                                            variant="secondary",
                                        )
                                with gr.Column():
                                    with gr.Group(
                                        visible=False
                                    ) as silence_output_box:
                                        silence_output = gr.Audio(
                                            label="Silence-Reduced Audio",
                                            interactive=False,
                                            buttons=["download"],
                                        )
                                        silence_share_links = gr.Markdown()
                        with gr.TabItem("Compact Audio"):
                            with gr.Row():
                                with gr.Column():
                                    compact_input = gr.Audio(
                                        label="Upload Audio", type="filepath"
                                    )
                                    compact_format = gr.Radio(
                                        format_choices,
                                        label="Output Format",
                                        value="MP3",
                                    )
                                    gr.Markdown(
                                        "Creates a lighter export using the project's compact-audio preset."
                                    )
                                    with gr.Row():
                                        compact_btn = gr.Button(
                                            "Compact Audio",
                                            variant="primary",
                                        )
                                        clear_compact_btn = gr.Button(
                                            "Clear",
                                            variant="secondary",
                                        )
                                with gr.Column():
                                    with gr.Group(
                                        visible=False
                                    ) as compact_output_box:
                                        compact_output = gr.Audio(
                                            label="Compacted Audio",
                                            interactive=False,
                                            buttons=["download"],
                                        )
                                        compact_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_preview_split:
                    gr.Markdown("## Preview & Split")
                    with gr.Tabs():
                        with gr.TabItem("Smart Preview"):
                            with gr.Row():
                                with gr.Column():
                                    preview_input = gr.Audio(
                                        label="Upload Audio", type="filepath"
                                    )
                                    preview_format = gr.Radio(
                                        format_choices,
                                        label="Output Format",
                                        value="MP3",
                                    )
                                    preview_duration = gr.Slider(
                                        5,
                                        60,
                                        30,
                                        step=1,
                                        label="Preview Length (seconds)",
                                    )
                                    with gr.Row():
                                        preview_btn = gr.Button(
                                            "Create Preview",
                                            variant="primary",
                                        )
                                        clear_preview_btn = gr.Button(
                                            "Clear",
                                            variant="secondary",
                                        )
                                with gr.Column():
                                    with gr.Group(
                                        visible=False
                                    ) as preview_output_box:
                                        preview_output = gr.Audio(
                                            label="Preview Clip",
                                            interactive=False,
                                            buttons=["download"],
                                        )
                                        preview_summary = gr.Markdown()
                                        preview_share_links = gr.Markdown()
                        with gr.TabItem("Split Audio"):
                            with gr.Row():
                                with gr.Column():
                                    split_input = gr.Audio(
                                        label="Upload Audio", type="filepath"
                                    )
                                    split_format = gr.Radio(
                                        format_choices,
                                        label="Chunk Format",
                                        value="MP3",
                                    )
                                    split_duration = gr.Slider(
                                        5,
                                        300,
                                        30,
                                        step=1,
                                        label="Chunk Length (seconds)",
                                    )
                                    split_skip_time = gr.Number(
                                        label="Skip Time Before First Chunk (seconds)",
                                        value=0,
                                        precision=2,
                                    )
                                    split_chunks_limit = gr.Number(
                                        label="Maximum Number of Chunks (0 = all)",
                                        value=0,
                                        precision=0,
                                    )
                                    split_sample_rate = gr.Number(
                                        label="Target Sample Rate (0 = keep original)",
                                        value=0,
                                        precision=0,
                                    )
                                    with gr.Row():
                                        split_btn = gr.Button(
                                            "Split Audio",
                                            variant="primary",
                                        )
                                        clear_split_btn = gr.Button(
                                            "Clear",
                                            variant="secondary",
                                        )
                                with gr.Column():
                                    with gr.Group(
                                        visible=False
                                    ) as split_output_box:
                                        split_preview_output = gr.Audio(
                                            label="First Chunk Preview",
                                            interactive=False,
                                            buttons=["download"],
                                        )
                                        split_files_output = gr.File(
                                            label="Chunk Files",
                                            interactive=False,
                                            file_count="multiple",
                                        )
                                        split_summary_output = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_midi_tools:
                    gr.Markdown("## MIDI Tools")
                    with gr.Tabs():
                        with gr.TabItem("Audio to MIDI"):
                            with gr.Row():
                                with gr.Column():
                                    a2m_input = gr.Audio(
                                        label="Upload Audio", type="filepath"
                                    )
                                    with gr.Row():
                                        a2m_btn = gr.Button(
                                            "Convert to MIDI", variant="primary"
                                        )
                                        clear_a2m_btn = gr.Button(
                                            "Clear", variant="secondary"
                                        )
                                with gr.Column():
                                    with gr.Group(
                                        visible=False
                                    ) as a2m_output_box:
                                        a2m_output = gr.File(
                                            label="Output MIDI",
                                            interactive=False,
                                        )
                                        a2m_share_links = gr.Markdown()
                        with gr.TabItem("MIDI to Audio"):
                            with gr.Row():
                                with gr.Column():
                                    m2a_input = gr.File(
                                        label="Upload MIDI",
                                        file_types=[".mid", ".midi"],
                                    )
                                    m2a_format = gr.Radio(
                                        format_choices,
                                        label="Output Format",
                                        value=format_choices[0],
                                    )
                                    with gr.Row():
                                        m2a_btn = gr.Button(
                                            "Convert to Audio",
                                            variant="primary",
                                        )
                                        clear_m2a_btn = gr.Button(
                                            "Clear", variant="secondary"
                                        )
                                with gr.Column():
                                    with gr.Group(
                                        visible=False
                                    ) as m2a_output_box:
                                        m2a_output = gr.Audio(
                                            label="Output Audio",
                                            interactive=False,
                                            buttons=["download"],
                                        )
                                        m2a_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_audio_extender:
                    gr.Markdown("## Audio Extender")
                    with gr.Row():
                        with gr.Column():
                            extender_input = gr.Audio(
                                label="Upload Audio to Extend", type="filepath"
                            )
                            extender_duration = gr.Slider(
                                5,
                                60,
                                15,
                                step=1,
                                label="Extend Duration (seconds)",
                            )
                            extender_format = gr.Radio(
                                format_choices,
                                label="Output Format",
                                value=format_choices[0],
                            )
                            with gr.Row():
                                extender_btn = gr.Button(
                                    "Extend Audio", variant="primary"
                                )
                                clear_extender_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(visible=False) as extender_output_box:
                                extender_output = gr.Audio(
                                    label="Extended Audio",
                                    interactive=False,
                                    buttons=["download"],
                                )
                                extender_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_stem_mixer:
                    gr.Markdown("## Stem Mixer")
                    with gr.Row():
                        with gr.Column():
                            stem_mixer_files = gr.File(
                                label="Upload Stems (Drums, Bass, Vocals, etc.)",
                                file_count="multiple",
                                type="filepath",
                            )
                            stem_mixer_format = gr.Radio(
                                format_choices,
                                label="Output Format",
                                value=format_choices[0],
                            )
                            with gr.Row():
                                stem_mixer_btn = gr.Button(
                                    "Mix Stems", variant="primary"
                                )
                                clear_stem_mixer_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(
                                visible=False
                            ) as stem_mixer_output_box:
                                stem_mixer_output = gr.Audio(
                                    label="Mixed Track",
                                    interactive=False,
                                    buttons=["download"],
                                )
                                stem_mixer_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_feedback:
                    gr.Markdown("## AI Track Feedback")
                    with gr.Row():
                        with gr.Column():
                            feedback_input = gr.Audio(
                                label="Upload Your Track", type="filepath"
                            )
                            with gr.Row():
                                feedback_btn = gr.Button(
                                    "Get Feedback", variant="primary"
                                )
                                clear_feedback_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            feedback_output = gr.Markdown(label="Feedback")
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_instrument_id:
                    gr.Markdown("## Instrument Identification")
                    with gr.Row():
                        with gr.Column():
                            instrument_id_input = gr.Audio(
                                label="Upload Audio", type="filepath"
                            )
                            with gr.Row():
                                instrument_id_btn = gr.Button(
                                    "Identify Instruments", variant="primary"
                                )
                                clear_instrument_id_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            instrument_id_output = gr.Markdown(
                                label="Detected Instruments"
                            )
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_video_gen:
                    gr.Markdown("## AI Music Clip Generation")
                    with gr.Row():
                        with gr.Column():
                            video_gen_audio = gr.Audio(
                                label="Upload Audio", type="filepath"
                            )
                            with gr.Row():
                                video_gen_btn = gr.Button(
                                    "Generate Video", variant="primary"
                                )
                                clear_video_gen_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(
                                visible=False
                            ) as video_gen_output_box:
                                video_gen_output = gr.Video(
                                    label="Generated Clip",
                                    interactive=False,
                                    buttons=["download"],
                                )
                                video_gen_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_speed:
                    gr.Markdown("## Speed & Pitch")
                    with gr.Row():
                        with gr.Column():
                            speed_input = gr.Audio(
                                label="Upload Track", type="filepath"
                            )
                            speed_factor = gr.Slider(
                                minimum=0.5,
                                maximum=2.0,
                                value=1.0,
                                step=0.01,
                                label="Speed Factor",
                            )
                            preserve_pitch = gr.Checkbox(
                                label="Preserve Pitch (higher quality)",
                                value=True,
                            )
                            speed_format = gr.Radio(
                                format_choices,
                                label="Output Format",
                                value=format_choices[0],
                            )
                            with gr.Row():
                                speed_btn = gr.Button(
                                    "Change Speed", variant="primary"
                                )
                                clear_speed_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(visible=False) as speed_output_box:
                                speed_output = gr.Audio(
                                    label="Modified Audio",
                                    interactive=False,
                                    buttons=["download"],
                                )
                                speed_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_stem:
                    gr.Markdown("## Stem Separation")
                    with gr.Row():
                        with gr.Column():
                            stem_input = gr.Audio(
                                label="Upload Full Mix", type="filepath"
                            )
                            stem_mode = gr.Radio(
                                [
                                    "Acapella (Vocals Only)",
                                    "Karaoke (Instrumental Only)",
                                    "Vocals + Karaoke",
                                    "Mastering Layers (Vocals / Drums / Bass / Other)",
                                ],
                                label="Separation Mode",
                                value="Acapella (Vocals Only)",
                            )
                            stem_format = gr.Radio(
                                format_choices,
                                label="Output Format",
                                value="MP3",
                            )
                            stem_mode_note = gr.Markdown(
                                value="**Layer Controls:** Layer strategy settings appear only in Mastering Layers mode. Use Vocals + Karaoke when you only need the vocal and instrumental pair."
                            )
                            with gr.Accordion("Layer Controls", open=False):
                                with gr.Group(
                                    visible=False
                                ) as stem_layer_settings:
                                    stem_model_name = gr.Dropdown(
                                        STEM_MODEL_STRATEGY_CHOICES,
                                        label="Layer Separation Strategy",
                                        value=STEM_MODEL_STRATEGY_CHOICES[0],
                                    )
                                    stem_custom_model_name = gr.Textbox(
                                        label="Custom layer checkpoint",
                                        placeholder="Example: htdemucs_6s or custom_model.yaml",
                                        visible=False,
                                    )
                                    stem_layer_strategy_note = gr.Markdown(
                                        value=initial_stem_strategy_note
                                    )
                                    stem_shifts = gr.Slider(
                                        1,
                                        8,
                                        2,
                                        step=1,
                                        label="Layer Separation Shifts",
                                    )
                            with gr.Row():
                                stem_btn = gr.Button(
                                    "Separate Stems", variant="primary"
                                )
                                clear_stem_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(visible=False) as stem_output_box:
                                stem_output = gr.Audio(
                                    label="Primary Stem Preview",
                                    interactive=False,
                                    buttons=["download"],
                                )
                                stem_files_output = gr.File(
                                    label="Stem Files",
                                    interactive=False,
                                    file_count="multiple",
                                )
                                stem_summary_output = gr.Markdown()
                                stem_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_vps:
                    gr.Markdown("## Vocal Pitch Shifter")
                    with gr.Row():
                        with gr.Column():
                            vps_input = gr.Audio(
                                label="Upload Full Song", type="filepath"
                            )
                            vps_pitch = gr.Slider(
                                -12,
                                12,
                                0,
                                step=1,
                                label="Vocal Pitch Shift (Semitones)",
                            )
                            vps_format = gr.Radio(
                                format_choices,
                                label="Output Format",
                                value=format_choices[0],
                            )
                            with gr.Row():
                                vps_btn = gr.Button(
                                    "Shift Vocal Pitch", variant="primary"
                                )
                                clear_vps_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(visible=False) as vps_output_box:
                                vps_output = gr.Audio(
                                    label="Pitch Shifted Song",
                                    interactive=False,
                                    buttons=["download"],
                                )
                                vps_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_voice_lab:
                    gr.Markdown("## ðŸ”¬ Voice Lab")
                    with gr.Row(visible=False):
                        experiment = gr.Textbox(value=random_string())
                    with gr.Row():
                        inp = gr.File(label="Input", type="filepath")
                        outp = gr.File(
                            label="Output",
                            type="filepath",
                            file_count="multiple",
                        )
                    with gr.Row(visible=False):
                        lvl = gr.Number(
                            label="(re-)training step",
                            value=1,
                            minimum=1,
                            step=1,
                        )
                    with gr.Row():
                        but1 = gr.Button("Train", variant="primary")
                        bind_audio_action(
                            but1,
                            train_voice_lab_model,
                            inputs=[experiment, inp, lvl],
                            outputs=[outp, lvl],
                            action_label="Train Voice Lab",
                            running_detail="Training the voice lab model.",
                            success_detail="Voice lab training finished.",
                        )
                        but2 = gr.Button("Convert", variant="primary")
                        bind_audio_action(
                            but2,
                            convert_vocal_rvc,
                            inputs=[experiment, inp],
                            outputs=[outp],
                            action_label="Convert Voice",
                            running_detail="Running the voice conversion model.",
                            success_detail="Converted voice output is ready.",
                        )
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_dj:
                    gr.Markdown("## DJ AutoMix")
                    with gr.Row():
                        with gr.Column():
                            dj_files = gr.File(
                                label="Upload Audio Tracks",
                                file_count="multiple",
                                type="filepath",
                                allow_reordering=True,
                            )
                            dj_mix_type = gr.Radio(
                                ["Simple Crossfade", "Beatmatched Crossfade"],
                                label="Mix Type",
                                value="Beatmatched Crossfade",
                            )
                            dj_target_bpm = gr.Number(
                                label="Target BPM (Optional)"
                            )
                            dj_transition = gr.Slider(
                                1,
                                15,
                                5,
                                step=1,
                                label="Transition Duration (seconds)",
                            )
                            dj_format = gr.Radio(
                                format_choices,
                                label="Output Format",
                                value=format_choices[0],
                            )
                            with gr.Row():
                                dj_btn = gr.Button(
                                    "Create DJ Mix", variant="primary"
                                )
                                clear_dj_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(visible=False) as dj_output_box:
                                dj_output = gr.Audio(
                                    label="Final DJ Mix",
                                    interactive=False,
                                    buttons=["download"],
                                )
                                dj_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_music_gen:
                    gr.Markdown("## AI Music Generation")
                    if device() == "cpu":
                        gr.Markdown(
                            "<p style='color:orange;text-align:center;'>Running on a CPU. Music generation will be very slow.</p>"
                        )
                    with gr.Row():
                        with gr.Column():
                            gen_prompt = gr.Textbox(
                                lines=4,
                                label="Music Prompt",
                                placeholder="e.g., '80s synthwave, retro, upbeat'",
                            )
                            gen_duration = gr.Slider(
                                5, 30, 10, step=1, label="Duration (seconds)"
                            )
                            gen_format = gr.Radio(
                                format_choices,
                                label="Output Format",
                                value=format_choices[0],
                            )
                            with gr.Row():
                                gen_btn = gr.Button(
                                    "Generate Music",
                                    variant="primary",
                                    interactive=True,
                                )
                                clear_gen_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(visible=False) as gen_output_box:
                                gen_output = gr.Audio(
                                    label="Generated Music",
                                    interactive=False,
                                    buttons=["download"],
                                )
                                gen_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_voice_gen:
                    gr.Markdown("## AI Voice Generation")
                    with gr.Row():
                        with gr.Column():
                            vg_ref = gr.Audio(
                                label="Reference Audio (Optional tone guide)",
                                type="filepath",
                            )
                            vg_text = gr.Textbox(
                                lines=4,
                                label="Text to Speak",
                                placeholder="Enter the text you want the generated voice to say...",
                            )
                            vg_format = gr.Radio(
                                format_choices,
                                label="Output Format",
                                value=format_choices[0],
                            )
                            with gr.Row():
                                vg_btn = gr.Button(
                                    "Generate Voice",
                                    variant="primary",
                                    interactive=True,
                                )
                                clear_vg_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(visible=False) as vg_output_box:
                                vg_output = gr.Audio(
                                    label="Generated Voice Audio",
                                    interactive=False,
                                    buttons=["download"],
                                )
                                vg_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_analysis:
                    gr.Markdown("## Analysis & Diagnostics")
                    with gr.Row():
                        with gr.Column(scale=1):
                            analysis_input = gr.Audio(
                                label="Upload Audio", type="filepath"
                            )
                            with gr.Accordion("Advanced Analysis", open=False):
                                analysis_hop_length = gr.Number(
                                    label="Hop Length",
                                    value=1024,
                                    precision=0,
                                )
                                analysis_duration = gr.Number(
                                    label="Analysis Window (seconds, 0 = full track)",
                                    value=0,
                                    precision=2,
                                )
                                analysis_offset = gr.Number(
                                    label="Start Offset (seconds)",
                                    value=0,
                                    precision=2,
                                )
                            with gr.Row():
                                analysis_btn = gr.Button(
                                    "Analyze Audio", variant="primary"
                                )
                                clear_analysis_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column(scale=1):
                            with gr.Group(visible=False) as analysis_output_box:
                                analysis_bpm_key_output = gr.Textbox(
                                    label="Detected Key & BPM",
                                    interactive=False,
                                )
                                analysis_diagnostics_output = gr.Markdown()
                                analysis_json_output = gr.File(
                                    label="Analysis Report",
                                    interactive=False,
                                )
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_stt:
                    gr.Markdown("## Speech-to-Text")
                    with gr.Row():
                        with gr.Column():
                            stt_input = gr.Audio(
                                label="Upload Speech Audio", type="filepath"
                            )
                            stt_language = gr.Dropdown(
                                language_choices,
                                label="Language",
                                value="english",
                            )
                            with gr.Row():
                                stt_btn = gr.Button(
                                    "Transcribe Audio",
                                    variant="primary",
                                    interactive=True,
                                )
                                clear_stt_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            stt_output = gr.Textbox(
                                label="Transcription Result",
                                interactive=False,
                                lines=10,
                            )
                            stt_file_output = gr.File(
                                label="Download Transcript",
                                interactive=False,
                                visible=False,
                            )
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_spectrum:
                    gr.Markdown("## Spectrum Analyzer")
                    spec_input = gr.Audio(label="Upload Audio", type="filepath")
                    with gr.Row():
                        spec_btn = gr.Button(
                            "Generate Spectrum", variant="primary"
                        )
                        clear_spec_btn = gr.Button("Clear", variant="secondary")
                    spec_output = gr.Image(
                        label="Spectrum Plot", interactive=False
                    )
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_beat_vis:
                    gr.Markdown("## Beat Visualizer")
                    with gr.Row():
                        with gr.Column():
                            vis_image_input = gr.Image(
                                label="Upload Image", type="filepath"
                            )
                            vis_audio_input = gr.Audio(
                                label="Upload Audio", type="filepath"
                            )
                        with gr.Column():
                            vis_effect = gr.Radio(
                                [
                                    "None",
                                    "Blur",
                                    "Sharpen",
                                    "Contour",
                                    "Emboss",
                                ],
                                label="Image Effect",
                                value="None",
                            )
                            vis_animation = gr.Radio(
                                ["None", "Zoom In", "Zoom Out"],
                                label="Animation Style",
                                value="None",
                            )
                            vis_intensity = gr.Slider(
                                1.05,
                                1.5,
                                1.15,
                                step=0.01,
                                label="Beat Intensity",
                            )
                            with gr.Row():
                                vis_btn = gr.Button(
                                    "Create Beat Visualizer", variant="primary"
                                )
                                clear_vis_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                    with gr.Group(visible=False) as vis_output_box:
                        vis_output = gr.Video(
                            label="Visualizer Output",
                            buttons=["download"],
                        )
                        vis_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_lyric_vid:
                    gr.Markdown("## Lyric Video Creator")
                    with gr.Row():
                        with gr.Column():
                            lyric_audio = gr.Audio(
                                label="Upload Song", type="filepath"
                            )
                            lyric_bg = gr.File(
                                label="Upload Background (Image or Video)",
                                type="filepath",
                            )
                            lyric_position = gr.Radio(
                                ["center", "bottom"],
                                label="Text Position",
                                value="bottom",
                            )
                        with gr.Column():
                            lyric_text = gr.Textbox(
                                label="Lyrics",
                                lines=15,
                                placeholder="Enter lyrics here, one line per phrase...",
                            )
                            load_transcript_btn = gr.Button(
                                "Get Lyrics from Audio (via Speech-to-Text)"
                            )
                            lyric_language = gr.Dropdown(
                                language_choices,
                                label="Lyrics language (for Speech-to-Text)",
                                value="english",
                            )
                    with gr.Row():
                        lyric_btn = gr.Button(
                            "Create Lyric Video", variant="primary"
                        )
                        clear_lyric_btn = gr.Button(
                            "Clear", variant="secondary"
                        )
                    with gr.Group(visible=False) as lyric_output_box:
                        lyric_output = gr.Video(
                            label="Lyric Video Output",
                            buttons=["download"],
                        )
                        lyric_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_chatbot:
                    init_chat("Definers Audio support")

        views = {
            "enhancer": view_enhancer,
            "vocal_finish": view_vocal_finish,
            "cleanup": view_cleanup,
            "preview_split": view_preview_split,
            "midi_tools": view_midi_tools,
            "audio_extender": view_audio_extender,
            "stem_mixer": view_stem_mixer,
            "feedback": view_feedback,
            "instrument_id": view_instrument_id,
            "video_gen": view_video_gen,
            "speed": view_speed,
            "stem": view_stem,
            "vps": view_vps,
            "voice_lab": view_voice_lab,
            "dj": view_dj,
            "music_gen": view_music_gen,
            "voice_gen": view_voice_gen,
            "analysis": view_analysis,
            "stt": view_stt,
            "spectrum": view_spectrum,
            "beat_vis": view_beat_vis,
            "lyric_vid": view_lyric_vid,
            "chatbot": view_chatbot,
        }

        def switch_view(selected_tool_name):
            selected_view_key = tool_map[selected_tool_name]
            return {
                view: gr.update(visible=(key == selected_view_key))
                for key, view in views.items()
            }

        nav_dropdown.change(
            fn=switch_view,
            inputs=nav_dropdown,
            outputs=list(views.values()),
        )

        app.load(
            lambda: switch_view(default_tool_name),
            outputs=list(views.values()),
        )

        def build_share_markup(result_path):
            if not result_path:
                return ""
            return create_share_links(
                "definers",
                "audio",
                result_path,
                "Check out this creation from Definers Audio! ðŸŽ¶",
            )

        def empty_mastering_stem_preview_updates():
            return {
                enhancer_stem_preview_box: gr.update(visible=False),
                enhancer_vocals_stem: gr.update(value=None, visible=False),
                enhancer_drums_stem: gr.update(value=None, visible=False),
                enhancer_bass_stem: gr.update(value=None, visible=False),
                enhancer_other_stem: gr.update(value=None, visible=False),
                enhancer_guitar_stem: gr.update(value=None, visible=False),
                enhancer_piano_stem: gr.update(value=None, visible=False),
            }

        def mastering_stem_preview_updates(stem_files):
            stem_previews, _extra_stem_files = resolve_mastering_stem_previews(
                list(stem_files or ())
            )
            return {
                enhancer_stem_preview_box: gr.update(
                    visible=any(stem_previews.values())
                ),
                enhancer_vocals_stem: gr.update(
                    value=stem_previews["vocals"],
                    visible=stem_previews["vocals"] is not None,
                ),
                enhancer_drums_stem: gr.update(
                    value=stem_previews["drums"],
                    visible=stem_previews["drums"] is not None,
                ),
                enhancer_bass_stem: gr.update(
                    value=stem_previews["bass"],
                    visible=stem_previews["bass"] is not None,
                ),
                enhancer_other_stem: gr.update(
                    value=stem_previews["other"],
                    visible=stem_previews["other"] is not None,
                ),
                enhancer_guitar_stem: gr.update(
                    value=stem_previews["guitar"],
                    visible=stem_previews["guitar"] is not None,
                ),
                enhancer_piano_stem: gr.update(
                    value=stem_previews["piano"],
                    visible=stem_previews["piano"] is not None,
                ),
            }

        def resolve_activity_detail(
            fallback_detail,
            activity_snapshot,
        ):
            if activity_snapshot is None:
                return fallback_detail
            activity_message = str(
                getattr(activity_snapshot, "message", "")
            ).strip()
            return activity_message or fallback_detail

        def poll_activity_updates(
            activity_task,
            base_updates,
            *,
            action_label,
            action_steps,
            detail,
            active_step,
        ):
            last_sequence = -1
            while True:
                task_done = wait_for_download_activity_task(
                    activity_task,
                    0.12,
                )
                activity_snapshot = get_download_activity_snapshot(
                    activity_task.scope_id
                )
                if (
                    activity_snapshot is not None
                    and activity_snapshot.sequence != last_sequence
                ):
                    last_sequence = activity_snapshot.sequence
                    yield {
                        **base_updates,
                        audio_progress: progress_update(
                            action_label,
                            "running",
                            resolve_activity_detail(
                                detail,
                                activity_snapshot,
                            ),
                            steps=action_steps,
                            active_step=active_step,
                            activity_completed=getattr(
                                activity_snapshot,
                                "completed",
                                None,
                            ),
                            activity_total=getattr(
                                activity_snapshot,
                                "total",
                                None,
                            ),
                            bytes_downloaded=getattr(
                                activity_snapshot,
                                "bytes_downloaded",
                                None,
                            ),
                            bytes_total=getattr(
                                activity_snapshot,
                                "bytes_total",
                                None,
                            ),
                        ),
                    }
                if task_done:
                    break

        def create_ui_handler(
            btn, out_el, out_box, out_share, logic_func, *inputs
        ):
            action_label = str(
                getattr(btn, "value", None) or "Processing audio"
            )
            action_steps = (
                "Validate request",
                "Run audio workflow",
                "Publish output",
            )

            def ui_handler_generator(*args):
                yield {
                    btn: gr.update(
                        value=f"{action_label}...", interactive=False
                    ),
                    out_box: gr.update(visible=False),
                    out_el: gr.update(value=None),
                    out_share: gr.update(value=""),
                    audio_progress: progress_update(
                        action_label,
                        "running",
                        "Checking the request.",
                        steps=action_steps,
                        active_step=1,
                    ),
                }
                running_updates = {
                    btn: gr.update(
                        value=f"{action_label}...", interactive=False
                    ),
                    out_box: gr.update(visible=False),
                    out_el: gr.update(value=None),
                    out_share: gr.update(value=""),
                    audio_progress: progress_update(
                        action_label,
                        "running",
                        "Running the workflow.",
                        steps=action_steps,
                        active_step=2,
                    ),
                }
                yield running_updates
                try:
                    activity_task = create_download_activity_task(
                        logic_func,
                        *args,
                    )
                    yield from poll_activity_updates(
                        activity_task,
                        {
                            btn: gr.update(
                                value=f"{action_label}...",
                                interactive=False,
                            ),
                            out_box: gr.update(visible=False),
                            out_el: gr.update(value=None),
                            out_share: gr.update(value=""),
                        },
                        action_label=action_label,
                        action_steps=action_steps,
                        detail="Running the workflow.",
                        active_step=2,
                    )
                    result, _ = resolve_download_activity_task(activity_task)
                    share_html = build_share_markup(result)
                    yield {
                        btn: gr.update(value=action_label, interactive=True),
                        out_box: gr.update(visible=True),
                        out_el: gr.update(value=result),
                        out_share: gr.update(value=share_html),
                        audio_progress: progress_update(
                            action_label,
                            "success",
                            "Result is ready.",
                            steps=action_steps,
                            active_step=len(action_steps),
                        ),
                    }
                except Exception as error:
                    yield {
                        btn: gr.update(value=action_label, interactive=True),
                        out_box: gr.update(visible=False),
                        out_el: gr.update(value=None),
                        out_share: gr.update(value=""),
                        audio_progress: progress_update(
                            action_label,
                            "error",
                            resolve_activity_detail(
                                str(error),
                                getattr(
                                    error,
                                    "download_activity_snapshot",
                                    None,
                                ),
                            ),
                            steps=action_steps,
                            active_step=2,
                        ),
                    }
                    raise gr.Error(str(error))

            btn.click(
                ui_handler_generator,
                inputs=inputs,
                outputs=[btn, out_box, out_el, out_share, audio_progress],
                show_progress="minimal",
            )

        def update_mastering_profile_ui(
            profile_name,
            bass,
            volume,
            effects,
        ):
            state = get_mastering_profile_ui_state(
                profile_name,
                bass,
                volume,
                effects,
            )
            return (
                gr.update(
                    value=float(state["bass"]),
                    interactive=bool(state["controls_enabled"]),
                ),
                gr.update(
                    value=float(state["volume"]),
                    interactive=bool(state["controls_enabled"]),
                ),
                gr.update(
                    value=float(state["effects"]),
                    interactive=bool(state["controls_enabled"]),
                ),
                gr.update(value=str(state["description"])),
                gr.update(value=str(state["macro_note"])),
            )

        def update_mastering_stem_ui(
            stem_mastering_enabled,
            model_selection,
            model_override,
        ):
            enabled = bool(stem_mastering_enabled)
            strategy_note = (
                describe_stem_model_choice(model_selection, model_override)
                if enabled
                else "**Stem Strategy:** Stem-aware mastering is off. The track will be processed as a single stereo master."
            )
            return (
                gr.update(visible=enabled),
                gr.update(
                    visible=enabled
                    and is_custom_stem_model_strategy(model_selection)
                ),
                gr.update(value=str(strategy_note)),
            )

        def update_stem_layer_ui(
            separation_mode,
            model_selection,
            model_override,
        ):
            uses_layer_controls = (
                separation_mode
                == "Mastering Layers (Vocals / Drums / Bass / Other)"
            )
            mode_note = (
                "**Layer Controls:** Mastering Layers exports vocals, drums, bass, and other as separate files. Choose the separator strategy that best matches the source."
                if uses_layer_controls
                else "**Layer Controls:** These settings are used only in Mastering Layers mode. Use Vocals + Karaoke when you only need the vocal and instrumental pair."
            )
            return (
                gr.update(value=mode_note),
                gr.update(visible=uses_layer_controls),
                gr.update(
                    visible=uses_layer_controls
                    and is_custom_stem_model_strategy(model_selection)
                ),
                gr.update(
                    value=describe_stem_model_choice(
                        model_selection,
                        model_override,
                    )
                ),
            )

        enhancer_preset.change(
            update_mastering_profile_ui,
            [
                enhancer_preset,
                enhancer_bass,
                enhancer_volume,
                enhancer_effects,
            ],
            [
                enhancer_bass,
                enhancer_volume,
                enhancer_effects,
                enhancer_profile_note,
                enhancer_macro_note,
            ],
        )

        enhancer_stem_mastering.change(
            update_mastering_stem_ui,
            [
                enhancer_stem_mastering,
                enhancer_stem_strategy,
                enhancer_custom_stem_model,
            ],
            [
                enhancer_stem_settings,
                enhancer_custom_stem_model,
                enhancer_stem_strategy_note,
            ],
        )

        enhancer_stem_strategy.change(
            update_mastering_stem_ui,
            [
                enhancer_stem_mastering,
                enhancer_stem_strategy,
                enhancer_custom_stem_model,
            ],
            [
                enhancer_stem_settings,
                enhancer_custom_stem_model,
                enhancer_stem_strategy_note,
            ],
        )

        enhancer_custom_stem_model.change(
            update_mastering_stem_ui,
            [
                enhancer_stem_mastering,
                enhancer_stem_strategy,
                enhancer_custom_stem_model,
            ],
            [
                enhancer_stem_settings,
                enhancer_custom_stem_model,
                enhancer_stem_strategy_note,
            ],
        )

        stem_mode.change(
            update_stem_layer_ui,
            [stem_mode, stem_model_name, stem_custom_model_name],
            [
                stem_mode_note,
                stem_layer_settings,
                stem_custom_model_name,
                stem_layer_strategy_note,
            ],
        )

        stem_model_name.change(
            update_stem_layer_ui,
            [stem_mode, stem_model_name, stem_custom_model_name],
            [
                stem_mode_note,
                stem_layer_settings,
                stem_custom_model_name,
                stem_layer_strategy_note,
            ],
        )

        stem_custom_model_name.change(
            update_stem_layer_ui,
            [stem_mode, stem_model_name, stem_custom_model_name],
            [
                stem_mode_note,
                stem_layer_settings,
                stem_custom_model_name,
                stem_layer_strategy_note,
            ],
        )

        def mastering_ui(
            audio_path,
            output_format,
            profile_name,
            bass,
            volume,
            effects,
            stem_mastering,
            stem_model_name,
            stem_shifts_value,
            stem_mix_headroom_value,
            save_mastered_stems_value,
            stem_model_override,
            stem_glue_reverb_amount_value,
            stem_drum_edge_amount_value,
            stem_vocal_pullback_db_value,
        ):
            mastering_steps = (
                "Validate source",
                "Configure mastering",
                "Run mastering engine",
                "Load reports and stems",
                "Publish output",
            )
            yield {
                enhancer_btn: gr.update(
                    value="Mastering...", interactive=False
                ),
                enhancer_output_column: gr.update(visible=False),
                enhancer_output: None,
                enhancer_report: gr.update(value=None, visible=False),
                enhancer_stems_output: gr.update(value=None, visible=False),
                **empty_mastering_stem_preview_updates(),
                enhancer_diagnostics: "",
                enhancer_share_links: "",
                audio_progress: progress_update(
                    "Master Audio",
                    "running",
                    "Checking the mastering request.",
                    steps=mastering_steps,
                    active_step=1,
                ),
            }
            yield {
                enhancer_btn: gr.update(
                    value="Mastering...", interactive=False
                ),
                enhancer_output_column: gr.update(visible=False),
                enhancer_output: None,
                enhancer_report: gr.update(value=None, visible=False),
                enhancer_stems_output: gr.update(value=None, visible=False),
                **empty_mastering_stem_preview_updates(),
                enhancer_diagnostics: "",
                enhancer_share_links: "",
                audio_progress: progress_update(
                    "Master Audio",
                    "running",
                    "Preparing the mastering route.",
                    steps=mastering_steps,
                    active_step=2,
                ),
            }
            yield {
                enhancer_btn: gr.update(
                    value="Mastering...", interactive=False
                ),
                enhancer_output_column: gr.update(visible=False),
                enhancer_output: None,
                enhancer_report: gr.update(value=None, visible=False),
                enhancer_stems_output: gr.update(value=None, visible=False),
                **empty_mastering_stem_preview_updates(),
                enhancer_diagnostics: "",
                enhancer_share_links: "",
                audio_progress: progress_update(
                    "Master Audio",
                    "running",
                    "Running the mastering engine.",
                    steps=mastering_steps,
                    active_step=3,
                ),
            }
            try:
                activity_task = create_download_activity_task(
                    run_mastering_tool,
                    audio_path,
                    output_format,
                    profile_name,
                    bass,
                    volume,
                    effects,
                    stem_mastering,
                    stem_model_name,
                    stem_shifts_value,
                    stem_mix_headroom_value,
                    save_mastered_stems_value,
                    stem_model_override,
                    stem_glue_reverb_amount_value,
                    stem_drum_edge_amount_value,
                    stem_vocal_pullback_db_value,
                )
                yield from poll_activity_updates(
                    activity_task,
                    {
                        enhancer_btn: gr.update(
                            value="Mastering...", interactive=False
                        ),
                        enhancer_output_column: gr.update(visible=False),
                        enhancer_output: None,
                        enhancer_report: gr.update(
                            value=None,
                            visible=False,
                        ),
                        enhancer_stems_output: gr.update(
                            value=None,
                            visible=False,
                        ),
                        **empty_mastering_stem_preview_updates(),
                        enhancer_diagnostics: "",
                        enhancer_share_links: "",
                    },
                    action_label="Master Audio",
                    action_steps=mastering_steps,
                    detail="Running the mastering engine.",
                    active_step=3,
                )
                (
                    (
                        mastered_path,
                        report_path,
                        diagnostics_text,
                        stem_files,
                    ),
                    _,
                ) = resolve_download_activity_task(
                    activity_task,
                )
                result_updates = {
                    enhancer_btn: gr.update(
                        value="Master Audio", interactive=True
                    ),
                    enhancer_output_column: gr.update(visible=True),
                    enhancer_output: mastered_path,
                    enhancer_report: gr.update(
                        value=report_path,
                        visible=report_path is not None,
                    ),
                    enhancer_stems_output: gr.update(
                        value=stem_files or None,
                        visible=bool(save_mastered_stems_value and stem_files),
                    ),
                    **mastering_stem_preview_updates(stem_files),
                    enhancer_diagnostics: diagnostics_text,
                    enhancer_share_links: build_share_markup(mastered_path),
                }
                yield {
                    **result_updates,
                    audio_progress: progress_update(
                        "Master Audio",
                        "running",
                        "Loading diagnostics and stem previews.",
                        steps=mastering_steps,
                        active_step=4,
                    ),
                }
                yield {
                    **result_updates,
                    audio_progress: progress_update(
                        "Master Audio",
                        "success",
                        "Mastered audio and stem previews are ready.",
                        steps=mastering_steps,
                        active_step=len(mastering_steps),
                    ),
                }
            except Exception as error:
                yield {
                    enhancer_btn: gr.update(
                        value="Master Audio", interactive=True
                    ),
                    enhancer_output_column: gr.update(visible=False),
                    enhancer_output: None,
                    enhancer_report: gr.update(value=None, visible=False),
                    enhancer_stems_output: gr.update(value=None, visible=False),
                    **empty_mastering_stem_preview_updates(),
                    enhancer_diagnostics: "",
                    enhancer_share_links: "",
                    audio_progress: progress_update(
                        "Master Audio",
                        "error",
                        resolve_activity_detail(
                            str(error),
                            getattr(
                                error,
                                "download_activity_snapshot",
                                None,
                            ),
                        ),
                        steps=mastering_steps,
                        active_step=3,
                    ),
                }
                raise gr.Error(str(error))

        def prepare_mastering_job_view(
            audio_path,
            output_format,
            profile_name,
            bass,
            volume,
            effects,
            stem_mastering,
            stem_model_name,
            stem_shifts_value,
            stem_mix_headroom_value,
            save_mastered_stems_value,
            stem_model_override,
            stem_glue_reverb_amount_value,
            stem_drum_edge_amount_value,
            stem_vocal_pullback_db_value,
        ):
            manifest = prepare_mastering_job(
                audio_path,
                output_format,
                profile_name,
                bass,
                volume,
                effects,
                stem_mastering,
                stem_model_name,
                stem_shifts_value,
                stem_mix_headroom_value,
                save_mastered_stems_value,
                stem_model_override=stem_model_override,
                stem_glue_reverb_amount=stem_glue_reverb_amount_value,
                stem_drum_edge_amount=stem_drum_edge_amount_value,
                stem_vocal_pullback_db=stem_vocal_pullback_db_value,
            )
            return render_mastering_job_view(
                str(manifest["job_dir"]),
                title="Job prepared",
            )

        def run_full_mastering_job_view(
            audio_path,
            output_format,
            profile_name,
            bass,
            volume,
            effects,
            stem_mastering,
            stem_model_name,
            stem_shifts_value,
            stem_mix_headroom_value,
            save_mastered_stems_value,
            stem_model_override,
            stem_glue_reverb_amount_value,
            stem_drum_edge_amount_value,
            stem_vocal_pullback_db_value,
        ):
            manifest = run_full_mastering_job(
                audio_path,
                output_format,
                profile_name,
                bass,
                volume,
                effects,
                stem_mastering,
                stem_model_name,
                stem_shifts_value,
                stem_mix_headroom_value,
                save_mastered_stems_value,
                stem_model_override=stem_model_override,
                stem_glue_reverb_amount=stem_glue_reverb_amount_value,
                stem_drum_edge_amount=stem_drum_edge_amount_value,
                stem_vocal_pullback_db=stem_vocal_pullback_db_value,
            )
            return render_mastering_job_view(
                str(manifest["job_dir"]),
                title="Job completed",
                detail="The full staged mastering flow finished and published the artifacts.",
            )

        def separate_mastering_job_view(current_job_dir):
            manifest = separate_mastering_job_stems(current_job_dir)
            if not bool(
                dict(manifest.get("settings", {})).get("stem_mastering")
            ):
                return render_mastering_job_view(
                    current_job_dir,
                    title="Stereo-only job",
                    detail="Stem separation is disabled for this job. Go directly to Finalize Master.",
                )
            return render_mastering_job_view(current_job_dir)

        def mix_mastering_job_view(current_job_dir):
            manifest = build_mastering_job_mix(current_job_dir)
            if not bool(
                dict(manifest.get("settings", {})).get("stem_mastering")
            ):
                return render_mastering_job_view(
                    current_job_dir,
                    title="Stereo-only job",
                    detail="No stem mix is needed for this job. Go directly to Finalize Master.",
                )
            return render_mastering_job_view(current_job_dir)

        def finalize_mastering_job_view(current_job_dir):
            finalize_mastering_job(current_job_dir)
            return render_mastering_job_view(current_job_dir)

        def refresh_mastering_job_view(current_job_dir):
            refresh_mastering_job(current_job_dir)
            return render_mastering_job_view(
                current_job_dir,
                title="Job loaded",
                detail="Resume from the next unfinished step.",
            )

        enhancer_btn.click(
            mastering_ui,
            [
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
            [
                enhancer_btn,
                enhancer_output_column,
                enhancer_output,
                enhancer_report,
                enhancer_stems_output,
                enhancer_stem_preview_box,
                enhancer_vocals_stem,
                enhancer_drums_stem,
                enhancer_bass_stem,
                enhancer_other_stem,
                enhancer_guitar_stem,
                enhancer_piano_stem,
                enhancer_diagnostics,
                enhancer_share_links,
                audio_progress,
            ],
            show_progress="minimal",
        )

        enhancer_job_outputs = [
            enhancer_job_dir,
            enhancer_job_status,
            enhancer_job_raw_stems,
            enhancer_job_processed_stems,
            enhancer_job_mixed_audio,
            enhancer_job_mastered_audio,
            enhancer_job_report,
            enhancer_job_summary,
            enhancer_job_manifest,
        ]

        bind_progress_click(
            enhancer_job_prepare_btn,
            prepare_mastering_job_view,
            progress_output=audio_progress,
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
            outputs=enhancer_job_outputs,
            action_label="Prepare Staged Job",
            steps=(
                "Validate source",
                "Analyze mastering input",
                "Write job manifest",
                "Publish job",
            ),
            running_detail="Preparing the resumable mastering job.",
            success_detail="Staged mastering job is ready.",
        )
        bind_progress_click(
            enhancer_job_full_btn,
            run_full_mastering_job_view,
            progress_output=audio_progress,
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
            outputs=enhancer_job_outputs,
            action_label="Run Full Job",
            steps=(
                "Prepare job",
                "Run heavy stages",
                "Finalize master",
                "Publish artifacts",
            ),
            running_detail="Running the full staged mastering flow.",
            success_detail="Full staged mastering flow is complete.",
        )
        bind_progress_click(
            enhancer_job_separate_btn,
            separate_mastering_job_view,
            progress_output=audio_progress,
            inputs=[enhancer_job_dir],
            outputs=enhancer_job_outputs,
            action_label="Separate Stems",
            steps=(
                "Load job",
                "Separate stems",
                "Publish raw stems",
            ),
            running_detail="Running the stem-separation stage.",
            success_detail="Stem separation stage is complete.",
        )
        bind_progress_click(
            enhancer_job_mix_btn,
            mix_mastering_job_view,
            progress_output=audio_progress,
            inputs=[enhancer_job_dir],
            outputs=enhancer_job_outputs,
            action_label="Build Stem Mix",
            steps=(
                "Load job",
                "Process separated stems",
                "Publish mix artifacts",
            ),
            running_detail="Building the staged stem mix.",
            success_detail="Stem mix stage is complete.",
        )
        bind_progress_click(
            enhancer_job_finalize_btn,
            finalize_mastering_job_view,
            progress_output=audio_progress,
            inputs=[enhancer_job_dir],
            outputs=enhancer_job_outputs,
            action_label="Finalize Master",
            steps=(
                "Load job",
                "Render final delivery",
                "Publish master",
            ),
            running_detail="Rendering the staged final master.",
            success_detail="Staged final master is ready.",
        )
        bind_progress_click(
            enhancer_job_refresh_btn,
            refresh_mastering_job_view,
            progress_output=audio_progress,
            inputs=[enhancer_job_dir],
            outputs=enhancer_job_outputs,
            action_label="Refresh Job",
            steps=(
                "Load job",
                "Refresh artifacts",
                "Publish status",
            ),
            running_detail="Refreshing the staged mastering job.",
            success_detail="Staged job state is refreshed.",
        )

        create_ui_handler(
            a2m_btn,
            a2m_output,
            a2m_output_box,
            a2m_share_links,
            audio_to_midi,
            a2m_input,
        )
        create_ui_handler(
            m2a_btn,
            m2a_output,
            m2a_output_box,
            m2a_share_links,
            midi_to_audio,
            m2a_input,
            m2a_format,
        )
        create_ui_handler(
            extender_btn,
            extender_output,
            extender_output_box,
            extender_share_links,
            extend_audio,
            extender_input,
            extender_duration,
            extender_format,
        )
        create_ui_handler(
            autotune_btn,
            autotune_output,
            autotune_output_box,
            autotune_share_links,
            run_autotune_song_tool,
            autotune_input,
            autotune_format,
            autotune_strength,
            autotune_correct_timing,
            autotune_quantize,
            autotune_tolerance,
            autotune_attack,
        )
        create_ui_handler(
            humanize_btn,
            humanize_output,
            humanize_output_box,
            humanize_share_links,
            run_humanize_vocals_tool,
            humanize_input,
            humanize_amount,
            humanize_format,
        )
        create_ui_handler(
            silence_btn,
            silence_output,
            silence_output_box,
            silence_share_links,
            run_remove_silence_tool,
            silence_input,
            silence_format,
        )
        create_ui_handler(
            compact_btn,
            compact_output,
            compact_output_box,
            compact_share_links,
            run_compact_audio_tool,
            compact_input,
            compact_format,
        )
        create_ui_handler(
            stem_mixer_btn,
            stem_mixer_output,
            stem_mixer_output_box,
            stem_mixer_share_links,
            stem_mixer,
            stem_mixer_files,
            stem_mixer_format,
        )

        def preview_ui(audio_path, max_duration, output_format):
            preview_steps = (
                "Validate source",
                "Build preview clip",
                "Publish output",
            )
            yield {
                preview_btn: gr.update(
                    value="Building Preview...", interactive=False
                ),
                preview_output_box: gr.update(visible=False),
                preview_output: None,
                preview_summary: "",
                preview_share_links: "",
                audio_progress: progress_update(
                    "Create Preview",
                    "running",
                    "Checking the preview request.",
                    steps=preview_steps,
                    active_step=1,
                ),
            }
            yield {
                preview_btn: gr.update(
                    value="Building Preview...", interactive=False
                ),
                preview_output_box: gr.update(visible=False),
                preview_output: None,
                preview_summary: "",
                preview_share_links: "",
                audio_progress: progress_update(
                    "Create Preview",
                    "running",
                    "Building the preview clip.",
                    steps=preview_steps,
                    active_step=2,
                ),
            }
            try:
                activity_task = create_download_activity_task(
                    run_audio_preview_tool,
                    audio_path,
                    max_duration,
                    output_format,
                )
                yield from poll_activity_updates(
                    activity_task,
                    {
                        preview_btn: gr.update(
                            value="Building Preview...",
                            interactive=False,
                        ),
                        preview_output_box: gr.update(visible=False),
                        preview_output: None,
                        preview_summary: "",
                        preview_share_links: "",
                    },
                    action_label="Create Preview",
                    action_steps=preview_steps,
                    detail="Building the preview clip.",
                    active_step=2,
                )
                (
                    (
                        preview_path,
                        preview_text,
                    ),
                    _,
                ) = resolve_download_activity_task(
                    activity_task,
                )
                yield {
                    preview_btn: gr.update(
                        value="Create Preview", interactive=True
                    ),
                    preview_output_box: gr.update(visible=True),
                    preview_output: preview_path,
                    preview_summary: preview_text,
                    preview_share_links: build_share_markup(preview_path),
                    audio_progress: progress_update(
                        "Create Preview",
                        "success",
                        "Preview clip is ready.",
                        steps=preview_steps,
                        active_step=len(preview_steps),
                    ),
                }
            except Exception as error:
                yield {
                    preview_btn: gr.update(
                        value="Create Preview", interactive=True
                    ),
                    preview_output_box: gr.update(visible=False),
                    preview_output: None,
                    preview_summary: "",
                    preview_share_links: "",
                    audio_progress: progress_update(
                        "Create Preview",
                        "error",
                        resolve_activity_detail(
                            str(error),
                            getattr(
                                error,
                                "download_activity_snapshot",
                                None,
                            ),
                        ),
                        steps=preview_steps,
                        active_step=2,
                    ),
                }
                raise gr.Error(str(error))

        preview_btn.click(
            preview_ui,
            [preview_input, preview_duration, preview_format],
            [
                preview_btn,
                preview_output_box,
                preview_output,
                preview_summary,
                preview_share_links,
                audio_progress,
            ],
            show_progress="minimal",
        )

        def split_ui(
            audio_path,
            chunk_duration,
            output_format,
            chunks_limit,
            skip_time,
            target_sample_rate,
        ):
            split_steps = (
                "Validate source",
                "Split into chunks",
                "Publish output",
            )
            yield {
                split_btn: gr.update(value="Splitting...", interactive=False),
                split_output_box: gr.update(visible=False),
                split_preview_output: None,
                split_files_output: None,
                split_summary_output: "",
                audio_progress: progress_update(
                    "Split Audio",
                    "running",
                    "Checking the split request.",
                    steps=split_steps,
                    active_step=1,
                ),
            }
            yield {
                split_btn: gr.update(value="Splitting...", interactive=False),
                split_output_box: gr.update(visible=False),
                split_preview_output: None,
                split_files_output: None,
                split_summary_output: "",
                audio_progress: progress_update(
                    "Split Audio",
                    "running",
                    "Splitting the audio into chunks.",
                    steps=split_steps,
                    active_step=2,
                ),
            }
            try:
                activity_task = create_download_activity_task(
                    run_split_audio_tool,
                    audio_path,
                    chunk_duration,
                    output_format,
                    chunks_limit,
                    skip_time,
                    target_sample_rate,
                )
                yield from poll_activity_updates(
                    activity_task,
                    {
                        split_btn: gr.update(
                            value="Splitting...",
                            interactive=False,
                        ),
                        split_output_box: gr.update(visible=False),
                        split_preview_output: None,
                        split_files_output: None,
                        split_summary_output: "",
                    },
                    action_label="Split Audio",
                    action_steps=split_steps,
                    detail="Splitting the audio into chunks.",
                    active_step=2,
                )
                (
                    (
                        preview_path,
                        split_files,
                        summary_text,
                    ),
                    _,
                ) = resolve_download_activity_task(
                    activity_task,
                )
                yield {
                    split_btn: gr.update(value="Split Audio", interactive=True),
                    split_output_box: gr.update(visible=True),
                    split_preview_output: preview_path,
                    split_files_output: split_files,
                    split_summary_output: summary_text,
                    audio_progress: progress_update(
                        "Split Audio",
                        "success",
                        "Split files are ready.",
                        steps=split_steps,
                        active_step=len(split_steps),
                    ),
                }
            except Exception as error:
                yield {
                    split_btn: gr.update(value="Split Audio", interactive=True),
                    split_output_box: gr.update(visible=False),
                    split_preview_output: None,
                    split_files_output: None,
                    split_summary_output: "",
                    audio_progress: progress_update(
                        "Split Audio",
                        "error",
                        resolve_activity_detail(
                            str(error),
                            getattr(
                                error,
                                "download_activity_snapshot",
                                None,
                            ),
                        ),
                        steps=split_steps,
                        active_step=2,
                    ),
                }
                raise gr.Error(str(error))

        split_btn.click(
            split_ui,
            [
                split_input,
                split_duration,
                split_format,
                split_chunks_limit,
                split_skip_time,
                split_sample_rate,
            ],
            [
                split_btn,
                split_output_box,
                split_preview_output,
                split_files_output,
                split_summary_output,
                audio_progress,
            ],
            show_progress="minimal",
        )

        create_ui_handler(
            video_gen_btn,
            video_gen_output,
            video_gen_output_box,
            video_gen_share_links,
            music_video,
            video_gen_audio,
        )
        create_ui_handler(
            speed_btn,
            speed_output,
            speed_output_box,
            speed_share_links,
            change_audio_speed,
            speed_input,
            speed_factor,
            preserve_pitch,
            speed_format,
        )

        def stem_ui(
            audio_path,
            separation_mode,
            output_format,
            model_name,
            shifts_value,
            model_override,
        ):
            stem_steps = (
                "Validate source",
                "Resolve separation route",
                "Separate layers",
                "Publish output",
            )
            mode_map = {
                "Acapella (Vocals Only)": "acapella",
                "Karaoke (Instrumental Only)": "karaoke",
                "Vocals + Karaoke": "vocals_karaoke",
                "Mastering Layers (Vocals / Drums / Bass / Other)": "mastering_layers",
            }
            resolved_mode = mode_map.get(separation_mode, "acapella")
            yield {
                stem_btn: gr.update(value="Separating...", interactive=False),
                stem_output_box: gr.update(visible=False),
                stem_output: None,
                stem_files_output: None,
                stem_summary_output: "",
                stem_share_links: "",
                audio_progress: progress_update(
                    "Separate Stems",
                    "running",
                    "Checking the stem request.",
                    steps=stem_steps,
                    active_step=1,
                ),
            }
            yield {
                stem_btn: gr.update(value="Separating...", interactive=False),
                stem_output_box: gr.update(visible=False),
                stem_output: None,
                stem_files_output: None,
                stem_summary_output: "",
                stem_share_links: "",
                audio_progress: progress_update(
                    "Separate Stems",
                    "running",
                    "Selecting the stem separation route.",
                    steps=stem_steps,
                    active_step=2,
                ),
            }
            yield {
                stem_btn: gr.update(value="Separating...", interactive=False),
                stem_output_box: gr.update(visible=False),
                stem_output: None,
                stem_files_output: None,
                stem_summary_output: "",
                stem_share_links: "",
                audio_progress: progress_update(
                    "Separate Stems",
                    "running",
                    "Separating the selected stem layers.",
                    steps=stem_steps,
                    active_step=3,
                ),
            }
            try:
                activity_task = create_download_activity_task(
                    run_stem_separation_tool,
                    audio_path,
                    resolved_mode,
                    output_format,
                    model_name,
                    shifts_value,
                    model_override,
                )
                yield from poll_activity_updates(
                    activity_task,
                    {
                        stem_btn: gr.update(
                            value="Separating...",
                            interactive=False,
                        ),
                        stem_output_box: gr.update(visible=False),
                        stem_output: None,
                        stem_files_output: None,
                        stem_summary_output: "",
                        stem_share_links: "",
                    },
                    action_label="Separate Stems",
                    action_steps=stem_steps,
                    detail="Separating the selected stem layers.",
                    active_step=3,
                )
                (
                    (
                        primary_output,
                        stem_files,
                        summary_text,
                    ),
                    _,
                ) = resolve_download_activity_task(activity_task)
                yield {
                    stem_btn: gr.update(
                        value="Separate Stems", interactive=True
                    ),
                    stem_output_box: gr.update(visible=True),
                    stem_output: primary_output,
                    stem_files_output: stem_files,
                    stem_summary_output: summary_text,
                    stem_share_links: build_share_markup(primary_output),
                    audio_progress: progress_update(
                        "Separate Stems",
                        "success",
                        "Stem outputs are ready.",
                        steps=stem_steps,
                        active_step=len(stem_steps),
                    ),
                }
            except Exception as error:
                yield {
                    stem_btn: gr.update(
                        value="Separate Stems", interactive=True
                    ),
                    stem_output_box: gr.update(visible=False),
                    stem_output: None,
                    stem_files_output: None,
                    stem_summary_output: "",
                    stem_share_links: "",
                    audio_progress: progress_update(
                        "Separate Stems",
                        "error",
                        resolve_activity_detail(
                            str(error),
                            getattr(
                                error,
                                "download_activity_snapshot",
                                None,
                            ),
                        ),
                        steps=stem_steps,
                        active_step=3,
                    ),
                }
                raise gr.Error(str(error))

        stem_btn.click(
            stem_ui,
            [
                stem_input,
                stem_mode,
                stem_format,
                stem_model_name,
                stem_shifts,
                stem_custom_model_name,
            ],
            [
                stem_btn,
                stem_output_box,
                stem_output,
                stem_files_output,
                stem_summary_output,
                stem_share_links,
                audio_progress,
            ],
            show_progress="minimal",
        )

        create_ui_handler(
            vps_btn,
            vps_output,
            vps_output_box,
            vps_share_links,
            pitch_shift_vocals,
            vps_input,
            vps_pitch,
            vps_format,
        )
        create_ui_handler(
            dj_btn,
            dj_output,
            dj_output_box,
            dj_share_links,
            dj_mix,
            dj_files,
            dj_mix_type,
            dj_target_bpm,
            dj_transition,
            dj_format,
        )
        create_ui_handler(
            gen_btn,
            gen_output,
            gen_output_box,
            gen_share_links,
            generate_music,
            gen_prompt,
            gen_duration,
            gen_format,
        )
        create_ui_handler(
            vg_btn,
            vg_output,
            vg_output_box,
            vg_share_links,
            generate_voice,
            vg_text,
            vg_ref,
            vg_format,
        )
        create_ui_handler(
            vis_btn,
            vis_output,
            vis_output_box,
            vis_share_links,
            beat_visualizer,
            vis_image_input,
            vis_audio_input,
            vis_effect,
            vis_animation,
            vis_intensity,
        )
        create_ui_handler(
            lyric_btn,
            lyric_output,
            lyric_output_box,
            lyric_share_links,
            lyric_video,
            lyric_audio,
            lyric_bg,
            lyric_text,
            lyric_position,
        )

        def analysis_ui(audio_path, hop_length, duration_value, offset_value):
            analysis_steps = (
                "Validate source",
                "Analyze tempo and key",
                "Publish report",
            )
            yield {
                analysis_btn: gr.update(
                    value="Analyzing...", interactive=False
                ),
                analysis_output_box: gr.update(visible=False),
                analysis_bpm_key_output: "",
                analysis_diagnostics_output: "",
                analysis_json_output: None,
                audio_progress: progress_update(
                    "Analyze Audio",
                    "running",
                    "Checking the analysis request.",
                    steps=analysis_steps,
                    active_step=1,
                ),
            }
            yield {
                analysis_btn: gr.update(
                    value="Analyzing...", interactive=False
                ),
                analysis_output_box: gr.update(visible=False),
                analysis_bpm_key_output: "",
                analysis_diagnostics_output: "",
                analysis_json_output: None,
                audio_progress: progress_update(
                    "Analyze Audio",
                    "running",
                    "Inspecting tempo, key, and diagnostics.",
                    steps=analysis_steps,
                    active_step=2,
                ),
            }
            try:
                activity_task = create_download_activity_task(
                    run_audio_analysis_tool,
                    audio_path,
                    hop_length,
                    duration_value,
                    offset_value,
                )
                yield from poll_activity_updates(
                    activity_task,
                    {
                        analysis_btn: gr.update(
                            value="Analyzing...",
                            interactive=False,
                        ),
                        analysis_output_box: gr.update(visible=False),
                        analysis_bpm_key_output: "",
                        analysis_diagnostics_output: "",
                        analysis_json_output: None,
                    },
                    action_label="Analyze Audio",
                    action_steps=analysis_steps,
                    detail="Inspecting tempo, key, and diagnostics.",
                    active_step=2,
                )
                (
                    (
                        bpm_key_text,
                        diagnostics_text,
                        report_path,
                    ),
                    _,
                ) = resolve_download_activity_task(
                    activity_task,
                )
                yield {
                    analysis_btn: gr.update(
                        value="Analyze Audio", interactive=True
                    ),
                    analysis_output_box: gr.update(visible=True),
                    analysis_bpm_key_output: bpm_key_text,
                    analysis_diagnostics_output: diagnostics_text,
                    analysis_json_output: report_path,
                    audio_progress: progress_update(
                        "Analyze Audio",
                        "success",
                        "Audio analysis is ready.",
                        steps=analysis_steps,
                        active_step=len(analysis_steps),
                    ),
                }
            except Exception as error:
                yield {
                    analysis_btn: gr.update(
                        value="Analyze Audio", interactive=True
                    ),
                    analysis_output_box: gr.update(visible=False),
                    analysis_bpm_key_output: "",
                    analysis_diagnostics_output: "",
                    analysis_json_output: None,
                    audio_progress: progress_update(
                        "Analyze Audio",
                        "error",
                        resolve_activity_detail(
                            str(error),
                            getattr(
                                error,
                                "download_activity_snapshot",
                                None,
                            ),
                        ),
                        steps=analysis_steps,
                        active_step=2,
                    ),
                }
                raise gr.Error(str(error))

        analysis_btn.click(
            analysis_ui,
            [
                analysis_input,
                analysis_hop_length,
                analysis_duration,
                analysis_offset,
            ],
            [
                analysis_btn,
                analysis_output_box,
                analysis_bpm_key_output,
                analysis_diagnostics_output,
                analysis_json_output,
                audio_progress,
            ],
            show_progress="minimal",
        )

        def feedback_ui(audio_path):
            feedback_steps = (
                "Validate source",
                "Review track",
                "Publish feedback",
            )
            yield {
                feedback_btn: gr.update(
                    value="Analyzing...", interactive=False
                ),
                feedback_output: "",
                audio_progress: progress_update(
                    "Get Feedback",
                    "running",
                    "Checking the feedback request.",
                    steps=feedback_steps,
                    active_step=1,
                ),
            }
            yield {
                feedback_btn: gr.update(
                    value="Analyzing...", interactive=False
                ),
                feedback_output: "",
                audio_progress: progress_update(
                    "Get Feedback",
                    "running",
                    "Reviewing the track for feedback.",
                    steps=feedback_steps,
                    active_step=2,
                ),
            }
            try:
                activity_task = create_download_activity_task(
                    get_audio_feedback,
                    audio_path,
                )
                yield from poll_activity_updates(
                    activity_task,
                    {
                        feedback_btn: gr.update(
                            value="Analyzing...",
                            interactive=False,
                        ),
                        feedback_output: "",
                    },
                    action_label="Get Feedback",
                    action_steps=feedback_steps,
                    detail="Reviewing the track for feedback.",
                    active_step=2,
                )
                feedback_text, _ = resolve_download_activity_task(activity_task)
                yield {
                    feedback_btn: gr.update(
                        value="Get Feedback", interactive=True
                    ),
                    feedback_output: feedback_text,
                    audio_progress: progress_update(
                        "Get Feedback",
                        "success",
                        "Feedback is ready.",
                        steps=feedback_steps,
                        active_step=len(feedback_steps),
                    ),
                }
            except Exception as error:
                yield {
                    feedback_btn: gr.update(
                        value="Get Feedback", interactive=True
                    ),
                    audio_progress: progress_update(
                        "Get Feedback",
                        "error",
                        resolve_activity_detail(
                            str(error),
                            getattr(
                                error,
                                "download_activity_snapshot",
                                None,
                            ),
                        ),
                        steps=feedback_steps,
                        active_step=2,
                    ),
                }
                raise gr.Error(str(error))

        feedback_btn.click(
            feedback_ui,
            [feedback_input],
            [feedback_btn, feedback_output, audio_progress],
            show_progress="minimal",
        )

        def instrument_id_ui(audio_path):
            instrument_steps = (
                "Validate source",
                "Identify instruments",
                "Publish report",
            )
            yield {
                instrument_id_btn: gr.update(
                    value="Identifying...", interactive=False
                ),
                instrument_id_output: "",
                audio_progress: progress_update(
                    "Identify Instruments",
                    "running",
                    "Checking the identification request.",
                    steps=instrument_steps,
                    active_step=1,
                ),
            }
            yield {
                instrument_id_btn: gr.update(
                    value="Identifying...", interactive=False
                ),
                instrument_id_output: "",
                audio_progress: progress_update(
                    "Identify Instruments",
                    "running",
                    "Identifying the instrument mix.",
                    steps=instrument_steps,
                    active_step=2,
                ),
            }
            try:
                activity_task = create_download_activity_task(
                    identify_instruments,
                    audio_path,
                )
                yield from poll_activity_updates(
                    activity_task,
                    {
                        instrument_id_btn: gr.update(
                            value="Identifying...",
                            interactive=False,
                        ),
                        instrument_id_output: "",
                    },
                    action_label="Identify Instruments",
                    action_steps=instrument_steps,
                    detail="Identifying the instrument mix.",
                    active_step=2,
                )
                instrument_text, _ = resolve_download_activity_task(
                    activity_task
                )
                yield {
                    instrument_id_btn: gr.update(
                        value="Identify Instruments",
                        interactive=True,
                    ),
                    instrument_id_output: instrument_text,
                    audio_progress: progress_update(
                        "Identify Instruments",
                        "success",
                        "Instrument identification is ready.",
                        steps=instrument_steps,
                        active_step=len(instrument_steps),
                    ),
                }
            except Exception as error:
                yield {
                    instrument_id_btn: gr.update(
                        value="Identify Instruments",
                        interactive=True,
                    ),
                    audio_progress: progress_update(
                        "Identify Instruments",
                        "error",
                        resolve_activity_detail(
                            str(error),
                            getattr(
                                error,
                                "download_activity_snapshot",
                                None,
                            ),
                        ),
                        steps=instrument_steps,
                        active_step=2,
                    ),
                }
                raise gr.Error(str(error))

        instrument_id_btn.click(
            instrument_id_ui,
            [instrument_id_input],
            [instrument_id_btn, instrument_id_output, audio_progress],
            show_progress="minimal",
        )

        def stt_ui(audio_path, language):
            transcription_steps = (
                "Validate source",
                "Transcribe audio",
                "Save transcript",
                "Publish output",
            )
            yield {
                stt_btn: gr.update(value="Transcribing...", interactive=False),
                stt_output: "",
                stt_file_output: gr.update(visible=False),
                audio_progress: progress_update(
                    "Transcribe Audio",
                    "running",
                    "Checking the transcription request.",
                    steps=transcription_steps,
                    active_step=1,
                ),
            }
            yield {
                stt_btn: gr.update(value="Transcribing...", interactive=False),
                stt_output: "",
                stt_file_output: gr.update(visible=False),
                audio_progress: progress_update(
                    "Transcribe Audio",
                    "running",
                    "Transcribing the selected audio.",
                    steps=transcription_steps,
                    active_step=2,
                ),
            }
            try:
                activity_task = create_download_activity_task(
                    transcribe_audio,
                    audio_path,
                    language,
                )
                yield from poll_activity_updates(
                    activity_task,
                    {
                        stt_btn: gr.update(
                            value="Transcribing...",
                            interactive=False,
                        ),
                        stt_output: "",
                        stt_file_output: gr.update(visible=False),
                    },
                    action_label="Transcribe Audio",
                    action_steps=transcription_steps,
                    detail="Transcribing the selected audio.",
                    active_step=2,
                )
                transcript, _ = resolve_download_activity_task(activity_task)
                yield {
                    stt_btn: gr.update(
                        value="Transcribing...", interactive=False
                    ),
                    stt_output: transcript,
                    stt_file_output: gr.update(visible=False),
                    audio_progress: progress_update(
                        "Transcribe Audio",
                        "running",
                        "Saving the transcript file.",
                        steps=transcription_steps,
                        active_step=3,
                    ),
                }
                file_path = save_text_to_file(transcript)
                yield {
                    stt_btn: gr.update(
                        value="Transcribe Audio", interactive=True
                    ),
                    stt_output: transcript,
                    stt_file_output: gr.update(visible=True, value=file_path),
                    audio_progress: progress_update(
                        "Transcribe Audio",
                        "success",
                        "Transcript is ready.",
                        steps=transcription_steps,
                        active_step=len(transcription_steps),
                    ),
                }
            except Exception as error:
                yield {
                    stt_btn: gr.update(
                        value="Transcribe Audio", interactive=True
                    ),
                    audio_progress: progress_update(
                        "Transcribe Audio",
                        "error",
                        resolve_activity_detail(
                            str(error),
                            getattr(
                                error,
                                "download_activity_snapshot",
                                None,
                            ),
                        ),
                        steps=transcription_steps,
                        active_step=3,
                    ),
                }
                raise gr.Error(str(error))

        stt_btn.click(
            stt_ui,
            [stt_input, stt_language],
            [stt_btn, stt_output, stt_file_output, audio_progress],
            show_progress="minimal",
        )

        def spec_ui(audio_path):
            spectrum_steps = (
                "Validate source",
                "Render spectrum",
                "Publish output",
            )
            yield {
                spec_btn: gr.update(value="Generating...", interactive=False),
                spec_output: None,
                audio_progress: progress_update(
                    "Generate Spectrum",
                    "running",
                    "Checking the spectrum request.",
                    steps=spectrum_steps,
                    active_step=1,
                ),
            }
            yield {
                spec_btn: gr.update(value="Generating...", interactive=False),
                spec_output: None,
                audio_progress: progress_update(
                    "Generate Spectrum",
                    "running",
                    "Rendering the spectrum visualization.",
                    steps=spectrum_steps,
                    active_step=2,
                ),
            }
            try:
                spec_image = create_spectrum_visualization(audio_path)
                yield {
                    spec_btn: gr.update(
                        value="Generate Spectrum", interactive=True
                    ),
                    spec_output: spec_image,
                    audio_progress: progress_update(
                        "Generate Spectrum",
                        "success",
                        "Spectrum visualization is ready.",
                        steps=spectrum_steps,
                        active_step=len(spectrum_steps),
                    ),
                }
            except Exception as error:
                yield {
                    spec_btn: gr.update(
                        value="Generate Spectrum", interactive=True
                    ),
                    audio_progress: progress_update(
                        "Generate Spectrum",
                        "error",
                        str(error),
                        steps=spectrum_steps,
                        active_step=2,
                    ),
                }
                raise gr.Error(str(error))

        spec_btn.click(
            spec_ui,
            [spec_input],
            [spec_btn, spec_output, audio_progress],
            show_progress="minimal",
        )

        def clear_ui(*components):
            updates = {}
            for comp in components:
                if isinstance(
                    comp,
                    (
                        gr.Audio,
                        gr.Video,
                        gr.Image,
                        gr.File,
                        gr.Textbox,
                        gr.Markdown,
                    ),
                ):
                    updates[comp] = None
                if isinstance(comp, gr.Group):
                    updates[comp] = gr.update(visible=False)
            return updates

        clear_enhancer_btn.click(
            lambda: {
                **clear_ui(
                    enhancer_input,
                    enhancer_output,
                    enhancer_report,
                    enhancer_stems_output,
                    enhancer_job_dir,
                    enhancer_job_raw_stems,
                    enhancer_job_processed_stems,
                    enhancer_job_mixed_audio,
                    enhancer_job_mastered_audio,
                    enhancer_job_report,
                    enhancer_job_summary,
                    enhancer_job_manifest,
                    enhancer_diagnostics,
                    enhancer_share_links,
                ),
                **empty_mastering_stem_preview_updates(),
                enhancer_output_column: gr.update(visible=False),
                enhancer_report: gr.update(value=None, visible=False),
                enhancer_stems_output: gr.update(value=None, visible=False),
                enhancer_job_status: gr.update(
                    value="## Staged mastering ready\nPrepare a job to enable resumable mastering stages."
                ),
            },
            [],
            [
                enhancer_input,
                enhancer_output,
                enhancer_report,
                enhancer_stems_output,
                enhancer_job_dir,
                enhancer_job_status,
                enhancer_job_raw_stems,
                enhancer_job_processed_stems,
                enhancer_job_mixed_audio,
                enhancer_job_mastered_audio,
                enhancer_job_report,
                enhancer_job_summary,
                enhancer_job_manifest,
                enhancer_diagnostics,
                enhancer_share_links,
                enhancer_output_column,
                enhancer_stem_preview_box,
                enhancer_vocals_stem,
                enhancer_drums_stem,
                enhancer_bass_stem,
                enhancer_other_stem,
                enhancer_guitar_stem,
                enhancer_piano_stem,
            ],
        )
        clear_autotune_btn.click(
            lambda: clear_ui(
                autotune_input,
                autotune_output,
                autotune_share_links,
                autotune_output_box,
            ),
            [],
            [
                autotune_input,
                autotune_output,
                autotune_share_links,
                autotune_output_box,
            ],
        )
        clear_humanize_btn.click(
            lambda: clear_ui(
                humanize_input,
                humanize_output,
                humanize_share_links,
                humanize_output_box,
            ),
            [],
            [
                humanize_input,
                humanize_output,
                humanize_share_links,
                humanize_output_box,
            ],
        )
        clear_silence_btn.click(
            lambda: clear_ui(
                silence_input,
                silence_output,
                silence_share_links,
                silence_output_box,
            ),
            [],
            [
                silence_input,
                silence_output,
                silence_share_links,
                silence_output_box,
            ],
        )
        clear_compact_btn.click(
            lambda: clear_ui(
                compact_input,
                compact_output,
                compact_share_links,
                compact_output_box,
            ),
            [],
            [
                compact_input,
                compact_output,
                compact_share_links,
                compact_output_box,
            ],
        )
        clear_preview_btn.click(
            lambda: clear_ui(
                preview_input,
                preview_output,
                preview_summary,
                preview_share_links,
                preview_output_box,
            ),
            [],
            [
                preview_input,
                preview_output,
                preview_summary,
                preview_share_links,
                preview_output_box,
            ],
        )
        clear_split_btn.click(
            lambda: clear_ui(
                split_input,
                split_preview_output,
                split_files_output,
                split_summary_output,
                split_output_box,
            ),
            [],
            [
                split_input,
                split_preview_output,
                split_files_output,
                split_summary_output,
                split_output_box,
            ],
        )
        clear_a2m_btn.click(
            lambda: clear_ui(a2m_input, a2m_output, a2m_output_box),
            [],
            [a2m_input, a2m_output, a2m_output_box],
        )
        clear_m2a_btn.click(
            lambda: clear_ui(m2a_input, m2a_output, m2a_output_box),
            [],
            [m2a_input, m2a_output, m2a_output_box],
        )
        clear_extender_btn.click(
            lambda: clear_ui(
                extender_input, extender_output, extender_output_box
            ),
            [],
            [extender_input, extender_output, extender_output_box],
        )
        clear_stem_mixer_btn.click(
            lambda: clear_ui(
                stem_mixer_files,
                stem_mixer_output,
                stem_mixer_output_box,
            ),
            [],
            [stem_mixer_files, stem_mixer_output, stem_mixer_output_box],
        )
        clear_feedback_btn.click(
            lambda: clear_ui(feedback_input, feedback_output),
            [],
            [feedback_input, feedback_output],
        )
        clear_instrument_id_btn.click(
            lambda: clear_ui(instrument_id_input, instrument_id_output),
            [],
            [instrument_id_input, instrument_id_output],
        )
        clear_video_gen_btn.click(
            lambda: clear_ui(
                video_gen_audio, video_gen_output, video_gen_output_box
            ),
            [],
            [video_gen_audio, video_gen_output, video_gen_output_box],
        )
        clear_speed_btn.click(
            lambda: clear_ui(speed_input, speed_output, speed_output_box),
            [],
            [speed_input, speed_output, speed_output_box],
        )
        clear_stem_btn.click(
            lambda: clear_ui(
                stem_input,
                stem_output,
                stem_files_output,
                stem_summary_output,
                stem_share_links,
                stem_output_box,
            ),
            [],
            [
                stem_input,
                stem_output,
                stem_files_output,
                stem_summary_output,
                stem_share_links,
                stem_output_box,
            ],
        )
        clear_vps_btn.click(
            lambda: clear_ui(vps_input, vps_output, vps_output_box),
            [],
            [vps_input, vps_output, vps_output_box],
        )
        clear_dj_btn.click(
            lambda: clear_ui(dj_files, dj_output, dj_output_box),
            [],
            [dj_files, dj_output, dj_output_box],
        )
        clear_gen_btn.click(
            lambda: {
                **clear_ui(gen_output, gen_output_box),
                **{gen_prompt: ""},
            },
            [],
            [gen_output, gen_output_box, gen_prompt],
        )
        clear_vg_btn.click(
            lambda: {
                **clear_ui(vg_ref, vg_output, vg_output_box),
                **{vg_text: ""},
            },
            [],
            [vg_ref, vg_output, vg_output_box, vg_text],
        )
        clear_analysis_btn.click(
            lambda: {
                **clear_ui(
                    analysis_input,
                    analysis_diagnostics_output,
                    analysis_json_output,
                    analysis_output_box,
                ),
                **{analysis_bpm_key_output: ""},
            },
            [],
            [
                analysis_input,
                analysis_bpm_key_output,
                analysis_diagnostics_output,
                analysis_json_output,
                analysis_output_box,
            ],
        )
        clear_stt_btn.click(
            lambda: clear_ui(stt_input, stt_output, stt_file_output),
            [],
            [stt_input, stt_output, stt_file_output],
        )
        clear_spec_btn.click(
            lambda: clear_ui(spec_input, spec_output),
            [],
            [spec_input, spec_output],
        )
        clear_vis_btn.click(
            lambda: clear_ui(
                vis_image_input, vis_audio_input, vis_output, vis_output_box
            ),
            [],
            [vis_image_input, vis_audio_input, vis_output, vis_output_box],
        )
        clear_lyric_btn.click(
            lambda: {
                **clear_ui(
                    lyric_audio, lyric_bg, lyric_output, lyric_output_box
                ),
                **{lyric_text: ""},
            },
            [],
            [lyric_audio, lyric_bg, lyric_output, lyric_output_box, lyric_text],
        )
        bind_audio_action(
            load_transcript_btn,
            transcribe_audio,
            inputs=[lyric_audio, lyric_language],
            outputs=[lyric_text],
            action_label="Load Transcript",
            running_detail="Transcribing the lyric reference audio.",
            success_detail="Transcript has been loaded into the editor.",
        )

    launch_blocks(app)
