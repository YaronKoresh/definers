from __future__ import annotations

import inspect
import json
from pathlib import Path

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

    output_metrics = getattr(report, "output_metrics", None)
    input_metrics = getattr(report, "input_metrics", None)
    delivery_issues = tuple(getattr(report, "delivery_issues", ()) or ())
    summary_lines = []
    if control_mode:
        summary_lines.append(f"**Control Mode:** {control_mode}")
    if stem_mastering and stem_strategy:
        summary_lines.append(f"**Stem Strategy:** {stem_strategy}")
    summary_lines.extend(
        [
            f"**Preset:** {getattr(report, 'preset_name', 'n/a')}",
            f"**Delivery Profile:** {getattr(report, 'delivery_profile_name', 'n/a')}",
            f"**Stem-aware Path:** {'On' if stem_mastering else 'Off'}",
            f"**Input LUFS:** {_format_metric_value(_metric_attr(input_metrics, 'integrated_lufs'), ' LUFS')}",
            f"**Output LUFS:** {_format_metric_value(_metric_attr(output_metrics, 'integrated_lufs'), ' LUFS')}",
            f"**True Peak:** {_format_metric_value(_metric_attr(output_metrics, 'true_peak_dbfs'), ' dBFS')}",
            f"**Crest Factor:** {_format_metric_value(_metric_attr(output_metrics, 'crest_factor_db'), ' dB')}",
            f"**Stereo Width:** {_format_metric_value(_metric_attr(output_metrics, 'stereo_width_ratio'))}",
            f"**Headroom Recovery:** {getattr(report, 'headroom_recovery_mode', 'n/a')}",
            f"**Recovered Gain:** {_format_metric_value(getattr(report, 'headroom_recovery_gain_db', None), ' dB')}",
            f"**Post-spatial Motion:** {_format_metric_value(getattr(report, 'post_spatial_stereo_motion', None))}",
            f"**Post-clamp Peak:** {_format_metric_value(getattr(report, 'post_clamp_true_peak_dbfs', None), ' dBFS')}",
        ]
    )
    if delivery_issues:
        summary_lines.append(
            "**Additional Information:** "
            + ", ".join(str(issue) for issue in delivery_issues)
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
) -> tuple[str, str | None, str, list[str]]:
    from definers.audio import master

    _report_audio_activity(
        "Resolve mastering request",
        detail="Normalizing the mastering profile and output paths.",
        completed=1,
        total=4,
    )
    output_path = _temp_audio_output_path(format_choice)
    report_path = _write_json_payload({"status": "pending"})
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
    mastering_kwargs: dict[str, object] = {
        "report_path": report_path,
        "stem_mastering": bool(stem_mastering),
        "stem_model_name": resolved_stem_model_name,
        "stem_shifts": max(int(stem_shifts), 1),
        "stem_mix_headroom_db": float(stem_mix_headroom_db),
        "save_mastered_stems": should_collect_mastered_stems,
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
    import numpy as np

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


__all__ = (
    "MASTERING_STEM_PREVIEW_ORDER",
    "MASTERING_PROFILE_CHOICES",
    "STEM_MODEL_STRATEGY_CHOICES",
    "describe_stem_model_choice",
    "format_mastering_summary",
    "get_mastering_profile_ui_state",
    "is_custom_stem_model_strategy",
    "normalize_audio_format_choice",
    "normalize_mastering_profile_selection",
    "resolve_mastering_stem_previews",
    "resolve_mastering_request",
    "resolve_stem_model_name",
    "run_audio_analysis_tool",
    "run_audio_preview_tool",
    "run_autotune_song_tool",
    "run_compact_audio_tool",
    "run_humanize_vocals_tool",
    "run_mastering_tool",
    "run_remove_silence_tool",
    "run_split_audio_tool",
    "run_stem_separation_tool",
)
