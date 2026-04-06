from __future__ import annotations

import json
from pathlib import Path


def normalize_audio_format_choice(format_choice: str) -> str:
    return str(format_choice).strip().lower().lstrip(".") or "wav"


def _temp_audio_output_path(format_choice: str) -> str:
    from definers.system import tmp

    return tmp(normalize_audio_format_choice(format_choice), keep=False)


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


def _write_json_payload(payload: dict[str, object]) -> str:
    from definers.system import tmp

    destination_path = Path(tmp("json", keep=False))
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
    from definers.system import tmp

    output_dir = Path(tmp(dir=True))
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
    stem_mastering: bool,
    stem_files: list[str],
) -> str:
    if report is None:
        return "Mastering completed without diagnostics."

    output_metrics = getattr(report, "output_metrics", None)
    input_metrics = getattr(report, "input_metrics", None)
    delivery_issues = tuple(getattr(report, "delivery_issues", ()) or ())
    summary_lines = [
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
    if delivery_issues:
        summary_lines.append(
            "**Delivery Issues:** "
            + ", ".join(str(issue) for issue in delivery_issues)
        )
    if stem_files:
        summary_lines.append(f"**Mastered Stems:** {len(stem_files)} files")
    return "\n\n".join(summary_lines)


def run_mastering_tool(
    audio_path: str,
    format_choice: str,
    preset_name: str,
    bass: float,
    volume: float,
    effects: float,
    stem_mastering: bool,
    stem_model_name: str,
    stem_shifts: int,
    stem_mix_headroom_db: float,
    save_mastered_stems: bool,
) -> tuple[str, str | None, str, list[str]]:
    from definers.audio import master

    output_path = _temp_audio_output_path(format_choice)
    report_path = _write_json_payload({"status": "pending"})
    mastering_kwargs: dict[str, object] = {
        "report_path": report_path,
        "bass": float(bass),
        "volume": float(volume),
        "effects": float(effects),
        "stem_mastering": bool(stem_mastering),
        "stem_model_name": str(stem_model_name).strip() or "mastering",
        "stem_shifts": max(int(stem_shifts), 1),
        "stem_mix_headroom_db": float(stem_mix_headroom_db),
        "save_mastered_stems": bool(save_mastered_stems),
    }
    normalized_preset = str(preset_name).strip().lower()
    if normalized_preset and normalized_preset != "auto":
        mastering_kwargs["preset"] = normalized_preset

    mastered_path, report = master(
        audio_path,
        output_path=output_path,
        **mastering_kwargs,
    )
    if mastered_path is None:
        raise RuntimeError("Mastering failed")

    stem_files: list[str] = []
    stem_output_dir = (
        Path(mastered_path).parent / f"{Path(mastered_path).stem}_stems"
    )
    if save_mastered_stems and stem_output_dir.exists():
        stem_files = [
            str(path)
            for path in sorted(stem_output_dir.iterdir())
            if path.is_file()
        ]

    resolved_report_path = report_path if Path(report_path).exists() else None
    return (
        mastered_path,
        resolved_report_path,
        format_mastering_summary(
            report,
            stem_mastering=bool(stem_mastering),
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
) -> tuple[str | None, list[str], str]:
    from definers.audio import separate_stem_layers, separate_stems

    normalized_format = normalize_audio_format_choice(format_choice)
    normalized_mode = str(separation_mode).strip().lower()

    if normalized_mode == "mastering_layers":
        stem_paths, _output_dir = separate_stem_layers(
            audio_path,
            model_name=str(model_name).strip() or "mastering",
            shifts=max(int(shifts), 1),
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
        summary = "Separated mastering layers: " + ", ".join(
            Path(output_path).stem for output_path in outputs
        )
        return (outputs[0] if outputs else None, outputs, summary)

    if normalized_mode == "vocals_karaoke":
        outputs = list(
            separate_stems(audio_path, format_choice=normalized_format) or ()
        )
        if not outputs:
            raise RuntimeError("Stem separation failed")
        return outputs[0], outputs, "Created vocals and instrumental stems"

    separation_type = "acapella" if normalized_mode == "acapella" else "karaoke"
    output_path = separate_stems(
        audio_path,
        separation_type=separation_type,
        format_choice=normalized_format,
    )
    if output_path is None:
        raise RuntimeError("Stem separation failed")
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
    return tuned_path


def run_humanize_vocals_tool(
    audio_path: str,
    amount: float,
    format_choice: str,
) -> str:
    from definers.audio import humanize_vocals

    humanized_path = humanize_vocals(audio_path, amount=float(amount))
    if humanized_path is None:
        raise RuntimeError("Vocal humanization failed")
    converted_paths = _convert_audio_outputs(
        [humanized_path],
        normalize_audio_format_choice(format_choice),
    )
    return converted_paths[0]


def run_remove_silence_tool(audio_path: str, format_choice: str) -> str:
    from definers.audio import remove_silence

    output_path = _temp_audio_output_path(format_choice)
    cleaned_path = remove_silence(audio_path, output_path)
    if cleaned_path is None:
        raise RuntimeError("Silence removal failed")
    return cleaned_path


def run_compact_audio_tool(audio_path: str, format_choice: str) -> str:
    from definers.audio import compact_audio

    output_path = _temp_audio_output_path(format_choice)
    compacted_path = compact_audio(audio_path, output_path)
    if compacted_path is None:
        raise RuntimeError("Audio compaction failed")
    return compacted_path


def run_audio_preview_tool(
    audio_path: str,
    max_duration: float,
    format_choice: str,
) -> tuple[str, str]:
    from definers.audio import audio_preview, get_audio_duration

    preview_path = audio_preview(audio_path, max_duration=float(max_duration))
    if preview_path is None:
        raise RuntimeError("Preview generation failed")
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
    from definers.system import tmp

    output_dir = tmp(dir=True)
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
    return summary_text, summary_markdown, _write_json_payload(analysis_payload)


__all__ = (
    "format_mastering_summary",
    "normalize_audio_format_choice",
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
