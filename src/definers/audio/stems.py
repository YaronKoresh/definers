from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from definers.constants import MODELS
from definers.logger import init_logger
from definers.system import catch, delete, tmp
from definers.text import random_string

from .dependencies import librosa_module
from .io import read_audio, save_audio
from .utils import normalize_audio_to_peak

_logger = init_logger()

_LEGACY_VOCAL_PAIR_MODEL_NAMES = {
    "auto",
    "hdemucs_mmi",
    "hdemucs_mmi.yaml",
    "mastering",
}

_MASTERING_RESERVED_MODEL_NAMES = {
    "auto",
    "mastering",
}

_STEM_NAME_ALIASES = {
    "vocals": "vocals",
    "vocal": "vocals",
    "instrumental": "instrumental",
    "inst": "instrumental",
    "no vocals": "instrumental",
    "no_vocals": "instrumental",
    "karaoke": "instrumental",
    "other": "other",
    "drums": "drums",
    "bass": "bass",
    "guitar": "guitar",
    "piano": "piano",
    "noreverb": "noreverb",
    "no reverb": "noreverb",
    "reverb": "reverb",
    "dry": "dry",
    "no noise": "no_noise",
    "noise": "noise",
    "no bleed": "no_bleed",
    "bleed": "bleed",
}


@dataclass(frozen=True, slots=True)
class SeparatorModelStage:
    model_candidates: tuple[str, ...]
    preferred_stems: tuple[str, ...]
    required: bool = True


@dataclass(frozen=True, slots=True)
class MasteringSeparatorPlan:
    target_sample_rate: int
    quality_flags: tuple[str, ...]
    preprocess_stages: tuple[SeparatorModelStage, ...]
    vocal_pair_stage: SeparatorModelStage | None
    reference_split_stage: SeparatorModelStage
    four_stem_stage: SeparatorModelStage
    vocal_stage: SeparatorModelStage
    vocal_restoration_stage: SeparatorModelStage
    instrumental_cleanup_stage: SeparatorModelStage


def _normalize_model_name(value: str, *, option_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"Missing {option_name}")
    if any(
        not (character.isalnum() or character in {"_", "-", ".", "/", "+"})
        for character in normalized
    ):
        raise ValueError(f"Invalid {option_name}: {value}")
    if normalized.lower().startswith(("htdemucs", "hdemucs")):
        suffix = Path(normalized).suffix.lower()
        if suffix not in {".yaml", ".yml"}:
            normalized = f"{normalized}.yaml"
    return normalized


def _normalize_optional_model_name(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return _normalize_model_name(
        normalized,
        option_name="separator model name",
    )


def _normalize_stage_candidates(candidates: Sequence[str]) -> tuple[str, ...]:
    normalized_candidates: list[str] = []
    for candidate in candidates:
        normalized_candidate = _normalize_model_name(
            candidate,
            option_name="separator model candidate",
        )
        if normalized_candidate not in normalized_candidates:
            normalized_candidates.append(normalized_candidate)
    return tuple(normalized_candidates)


def _resolve_mastering_sample_rate(input_sample_rate: int) -> int:
    if int(input_sample_rate) >= 48000:
        return 48000
    return 44100


def build_mastering_separator_plan(
    input_sample_rate: int,
    *,
    quality_flags: Sequence[str] = (),
    model_name: str = "mastering",
) -> MasteringSeparatorPlan:
    resolved_quality_flags = tuple(
        dict.fromkeys(
            str(flag).strip() for flag in quality_flags if str(flag).strip()
        )
    )
    override_model_name = _normalize_optional_model_name(model_name)
    use_vocal_pair_stage = (
        override_model_name is None
        or override_model_name.lower() in _MASTERING_RESERVED_MODEL_NAMES
    )
    four_stem_candidates = ["htdemucs_ft.yaml", "hdemucs_mmi.yaml"]
    if (
        override_model_name is not None
        and override_model_name.lower() not in _MASTERING_RESERVED_MODEL_NAMES
        and override_model_name.lower() not in _LEGACY_VOCAL_PAIR_MODEL_NAMES
    ):
        four_stem_candidates.insert(0, override_model_name)
    elif (
        override_model_name is not None
        and override_model_name.lower() not in _MASTERING_RESERVED_MODEL_NAMES
    ):
        four_stem_candidates.insert(0, override_model_name)

    preprocess_stages: tuple[SeparatorModelStage, ...] = ()
    if any(
        flag in {"Low-Quality", "Old-Recording"}
        for flag in resolved_quality_flags
    ):
        preprocess_stages = (
            SeparatorModelStage(
                model_candidates=_normalize_stage_candidates(
                    (
                        "deverb_bs_roformer_8_384dim_10depth.ckpt",
                        "deverb_bs_roformer_8_256dim_8depth.ckpt",
                        "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
                    )
                ),
                preferred_stems=("noreverb", "no reverb", "dry"),
                required=False,
            ),
            SeparatorModelStage(
                model_candidates=_normalize_stage_candidates(
                    (
                        "denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt",
                        "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
                    )
                ),
                preferred_stems=("no noise", "dry"),
                required=False,
            ),
        )

    reference_split_candidates = [
        "MDX23C-8KFFT-InstVoc_HQ.ckpt",
        "bs_roformer_instrumental_resurrection_unwa.ckpt",
    ]
    if use_vocal_pair_stage:
        reference_split_candidates = [
            "bs_roformer_instrumental_resurrection_unwa.ckpt",
            "MDX23C-8KFFT-InstVoc_HQ.ckpt",
        ]

    return MasteringSeparatorPlan(
        target_sample_rate=_resolve_mastering_sample_rate(input_sample_rate),
        quality_flags=resolved_quality_flags,
        preprocess_stages=preprocess_stages,
        vocal_pair_stage=(
            _build_vocal_pair_stage("mastering")
            if use_vocal_pair_stage
            else None
        ),
        reference_split_stage=SeparatorModelStage(
            model_candidates=_normalize_stage_candidates(
                tuple(reference_split_candidates)
            ),
            preferred_stems=("other", "instrumental"),
            required=False,
        ),
        four_stem_stage=SeparatorModelStage(
            model_candidates=_normalize_stage_candidates(
                tuple(four_stem_candidates)
            ),
            preferred_stems=("vocals", "drums", "bass", "other"),
            required=True,
        ),
        vocal_stage=SeparatorModelStage(
            model_candidates=_normalize_stage_candidates(
                (
                    "MelBandRoformerSYHFTV2.5.ckpt",
                    "MelBandRoformerSYHFT.ckpt",
                    "vocals_mel_band_roformer.ckpt",
                )
            ),
            preferred_stems=("vocals",),
            required=True,
        ),
        vocal_restoration_stage=SeparatorModelStage(
            model_candidates=_normalize_stage_candidates(
                ("bs_roformer_vocals_resurrection_unwa.ckpt",)
            ),
            preferred_stems=("vocals",),
            required=False,
        ),
        instrumental_cleanup_stage=SeparatorModelStage(
            model_candidates=_normalize_stage_candidates(
                (
                    "mel_band_roformer_bleed_suppressor_v1.ckpt",
                    "mel_band_roformer_instrumental_fv7z_gabox.ckpt",
                )
            ),
            preferred_stems=("no bleed", "other", "instrumental"),
            required=False,
        ),
    )


def _build_vocal_pair_stage(model_name: str) -> SeparatorModelStage:
    model_candidates: list[str] = []
    override_model_name = _normalize_optional_model_name(model_name)
    if (
        override_model_name is not None
        and override_model_name.lower() not in _LEGACY_VOCAL_PAIR_MODEL_NAMES
    ):
        model_candidates.append(override_model_name)
    model_candidates.extend(
        (
            "bs_roformer_vocals_resurrection_unwa.ckpt",
            "MelBandRoformerSYHFTV2.5.ckpt",
            "MelBandRoformerSYHFT.ckpt",
        )
    )
    return SeparatorModelStage(
        model_candidates=_normalize_stage_candidates(tuple(model_candidates)),
        preferred_stems=("vocals", "other", "instrumental"),
        required=True,
    )


def _load_audio_separator_class():
    try:
        from definers.model_installation import (
            install_audio_separator_runtime_hooks,
        )

        install_audio_separator_runtime_hooks()
        from audio_separator.separator import Separator
    except Exception as error:
        raise RuntimeError(
            "audio-separator is required for stem separation"
        ) from error
    return Separator


def _build_separator_kwargs(
    output_dir: str,
    target_sample_rate: int,
    *,
    shifts: int = 2,
) -> dict[str, object]:
    resolved_shifts = max(int(shifts), 1)
    kwargs: dict[str, object] = {
        "output_dir": output_dir,
        "output_format": "WAV",
        "sample_rate": int(target_sample_rate),
        "use_soundfile": True,
        "log_level": 40,
        "demucs_params": {
            "shifts": resolved_shifts,
            "overlap": 0.25,
            "segments_enabled": True,
        },
        "mdxc_params": {
            "segment_size": 256,
            "overlap": 4,
        },
    }
    from definers.model_installation import stem_model_dir

    kwargs["model_file_dir"] = stem_model_dir()
    return kwargs


def _has_local_stem_model(model_name: str) -> bool:
    from definers.model_installation import stem_model_artifacts_ready

    return stem_model_artifacts_ready(str(model_name))


def _runtime_stage_model_candidates(
    stage: SeparatorModelStage,
) -> tuple[str, ...]:
    cached_candidates = tuple(
        model_candidate
        for model_candidate in stage.model_candidates
        if _has_local_stem_model(model_candidate)
    )
    if cached_candidates:
        return cached_candidates
    return stage.model_candidates[:1]


def _download_runtime_stage_models(
    model_candidates: Sequence[str],
) -> tuple[str, ...]:
    from definers.model_installation import resolve_stem_model_filename

    resolved_candidates = tuple(
        dict.fromkeys(
            resolve_stem_model_filename(model_candidate)
            for model_candidate in model_candidates
        )
    )
    if not resolved_candidates:
        return ()
    missing_candidates = tuple(
        model_candidate
        for model_candidate in resolved_candidates
        if not _has_local_stem_model(model_candidate)
    )
    if missing_candidates:
        from definers.model_installation import download_stem_models

        download_stem_models(missing_candidates)
    return resolved_candidates


def _prefetch_stage_model_candidates(
    stage: SeparatorModelStage | None,
) -> tuple[str, ...]:
    if stage is None:
        return ()
    if not stage.model_candidates:
        return ()
    cached_candidates = tuple(
        model_candidate
        for model_candidate in stage.model_candidates
        if _has_local_stem_model(model_candidate)
    )
    if cached_candidates:
        return cached_candidates[:1]
    return stage.model_candidates[:1]


def _prefetch_mastering_plan_models(
    plan: MasteringSeparatorPlan,
) -> tuple[str, ...]:
    prefetch_candidates: list[str] = []
    for stage in plan.preprocess_stages:
        prefetch_candidates.extend(_prefetch_stage_model_candidates(stage))
    for stage in (
        plan.vocal_pair_stage,
        plan.reference_split_stage,
        plan.four_stem_stage,
        plan.vocal_stage,
        plan.vocal_restoration_stage,
        plan.instrumental_cleanup_stage,
    ):
        prefetch_candidates.extend(_prefetch_stage_model_candidates(stage))
    return _download_runtime_stage_models(
        tuple(dict.fromkeys(prefetch_candidates))
    )


def _prepare_stage_directory(output_dir: str) -> str:
    delete(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


def _separator_output_error(
    output_files: Sequence[object],
    output_dir: str,
) -> FileNotFoundError:
    flattened_output_files = _flatten_output_files(output_files)
    if not flattened_output_files:
        return FileNotFoundError("Audio separator returned no output files")
    return FileNotFoundError(
        "Audio separator returned output files that were not created under "
        + str(Path(output_dir))
        + ": "
        + ", ".join(flattened_output_files)
    )


def _resolve_output_paths(
    output_files: Sequence[str],
    output_dir: str,
) -> tuple[str, ...]:
    resolved_output_paths: list[str] = []
    for output_file in _flatten_output_files(output_files):
        candidate_path = Path(output_file)
        if candidate_path.is_absolute():
            if candidate_path.exists():
                resolved_output_paths.append(str(candidate_path))
            continue
        output_path = Path(output_dir) / output_file
        if not output_path.exists():
            output_path = Path(output_dir) / candidate_path.name
        if output_path.exists():
            resolved_output_paths.append(str(output_path))
    return tuple(resolved_output_paths)


def _flatten_output_files(output_files: Sequence[object]) -> tuple[str, ...]:
    flattened_output_files: list[str] = []
    for output_file in output_files:
        if isinstance(output_file, (list, tuple)):
            flattened_output_files.extend(_flatten_output_files(output_file))
            continue
        flattened_output_files.append(str(output_file))
    return tuple(flattened_output_files)


def _canonicalize_stem_name(stem_name: str) -> str:
    normalized_name = re.sub(r"[\s_-]+", " ", str(stem_name)).strip().lower()
    return _STEM_NAME_ALIASES.get(normalized_name, normalized_name)


def _extract_stem_name(output_path: str) -> str | None:
    stem_matches = re.findall(r"_\(([^)]+)\)", Path(output_path).name)
    if not stem_matches:
        return None
    return _canonicalize_stem_name(stem_matches[-1])


def _select_stage_output(
    output_files: Sequence[str],
    preferred_stems: Sequence[str],
) -> str | None:
    resolved_preferred_stems = {
        _canonicalize_stem_name(stem_name) for stem_name in preferred_stems
    }
    for output_file in output_files:
        stem_name = _extract_stem_name(output_file)
        if stem_name in resolved_preferred_stems:
            return output_file
    if len(output_files) == 1:
        return output_files[0]
    return None


def _output_matches_source_path(output_path: str, source_path: str) -> bool:
    output_token = re.sub(r"[^a-z0-9]+", "", Path(output_path).stem.lower())
    source_token = re.sub(r"[^a-z0-9]+", "", Path(source_path).stem.lower())
    return bool(source_token) and output_token.startswith(source_token)


def _select_stage_outputs_for_inputs(
    output_files: Sequence[str],
    input_paths: Mapping[str, str],
    preferred_stems: Sequence[str],
) -> dict[str, str]:
    selected_outputs: dict[str, str] = {}
    for stem_name, source_path in input_paths.items():
        matching_outputs = [
            output_path
            for output_path in output_files
            if _output_matches_source_path(output_path, source_path)
        ]
        selected_output = _select_stage_output(
            matching_outputs,
            preferred_stems,
        )
        if selected_output is not None:
            selected_outputs[stem_name] = selected_output
    return selected_outputs


def _run_separator_stage(
    input_path: str,
    stage: SeparatorModelStage,
    output_dir: str,
    target_sample_rate: int,
    *,
    shifts: int = 2,
) -> tuple[str, tuple[str, ...]]:
    Separator = _load_audio_separator_class()
    last_error: Exception | None = None
    runtime_model_candidates = _runtime_stage_model_candidates(stage)
    try:
        runtime_model_candidates = _download_runtime_stage_models(
            runtime_model_candidates
        )
    except Exception as error:
        if stage.required:
            raise RuntimeError(
                "Audio separator stage failed while preparing models: "
                + str(error)
            ) from error
        _logger.warning("Skipping optional separator stage: %s", error)
        return "", ()
    for model_candidate in runtime_model_candidates:
        _prepare_stage_directory(output_dir)
        try:
            separator = Separator(
                **_build_separator_kwargs(
                    output_dir,
                    target_sample_rate,
                    shifts=shifts,
                )
            )
            separator.load_model(model_filename=model_candidate)
            output_files = separator.separate(input_path)
            resolved_output_files = _resolve_output_paths(
                output_files, output_dir
            )
            if resolved_output_files:
                return model_candidate, resolved_output_files
            last_error = _separator_output_error(output_files, output_dir)
        except Exception as error:
            last_error = error
    if stage.required:
        raise RuntimeError(
            "Audio separator stage failed for model candidates "
            + ", ".join(runtime_model_candidates)
            + ": "
            + str(last_error)
        ) from last_error
    if last_error is not None:
        _logger.warning("Skipping optional separator stage: %s", last_error)
    return "", ()


def _run_separator_stage_batch(
    input_paths: Mapping[str, str],
    stage: SeparatorModelStage,
    output_dir: str,
    target_sample_rate: int,
    *,
    shifts: int = 2,
) -> dict[str, str]:
    if not input_paths:
        return {}

    Separator = _load_audio_separator_class()
    last_error: Exception | None = None
    batched_input_paths = tuple(input_paths.values())
    runtime_model_candidates = _runtime_stage_model_candidates(stage)
    try:
        runtime_model_candidates = _download_runtime_stage_models(
            runtime_model_candidates
        )
    except Exception as error:
        if stage.required:
            raise RuntimeError(
                "Audio separator stage failed while preparing models: "
                + str(error)
            ) from error
        _logger.warning("Skipping optional separator stage: %s", error)
        return dict(input_paths)
    for model_candidate in runtime_model_candidates:
        _prepare_stage_directory(output_dir)
        try:
            separator = Separator(
                **_build_separator_kwargs(
                    output_dir,
                    target_sample_rate,
                    shifts=shifts,
                )
            )
            separator.load_model(model_filename=model_candidate)
            output_files = separator.separate(list(batched_input_paths))
            resolved_output_files = _resolve_output_paths(
                output_files, output_dir
            )
            selected_outputs = _select_stage_outputs_for_inputs(
                resolved_output_files,
                input_paths,
                stage.preferred_stems,
            )
            if stage.required and len(selected_outputs) == len(input_paths):
                return selected_outputs
            if not stage.required and selected_outputs:
                return {
                    stem_name: selected_outputs.get(stem_name, source_path)
                    for stem_name, source_path in input_paths.items()
                }
            last_error = _separator_output_error(output_files, output_dir)
        except Exception as error:
            last_error = error
    if stage.required:
        raise RuntimeError(
            "Audio separator stage failed for model candidates "
            + ", ".join(runtime_model_candidates)
            + ": "
            + str(last_error)
        ) from last_error
    if last_error is not None:
        _logger.warning("Skipping optional separator stage: %s", last_error)
    return dict(input_paths)


def _as_audio_array(audio_signal: np.ndarray) -> np.ndarray:
    audio_array = np.asarray(audio_signal, dtype=np.float32)
    if audio_array.ndim == 1:
        return audio_array.reshape(1, -1)
    if audio_array.ndim == 2 and audio_array.shape[0] <= audio_array.shape[-1]:
        return audio_array.astype(np.float32, copy=False)
    if audio_array.ndim == 2:
        return audio_array.T.astype(np.float32, copy=False)
    raise ValueError("Unsupported audio shape")


def _resample_audio_array(
    audio_signal: np.ndarray,
    original_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    librosa = librosa_module()
    audio_array = _as_audio_array(audio_signal)
    if int(original_sample_rate) == int(target_sample_rate):
        return audio_array.astype(np.float32, copy=False)
    resampled_channels = [
        librosa.resample(
            np.asarray(channel, dtype=np.float32),
            orig_sr=int(original_sample_rate),
            target_sr=int(target_sample_rate),
        ).astype(np.float32, copy=False)
        for channel in audio_array
    ]
    return np.vstack(resampled_channels).astype(np.float32, copy=False)


def _align_audio_length(
    audio_signal: np.ndarray, target_length: int
) -> np.ndarray:
    audio_array = _as_audio_array(audio_signal)
    current_length = int(audio_array.shape[-1])
    if current_length == target_length:
        return audio_array.astype(np.float32, copy=False)
    if current_length > target_length:
        return audio_array[:, :target_length].astype(np.float32, copy=False)
    return np.pad(
        audio_array,
        ((0, 0), (0, target_length - current_length)),
    ).astype(np.float32, copy=False)


def _read_stage_signal(audio_path: str, target_sample_rate: int) -> np.ndarray:
    sample_rate, audio_signal = read_audio(audio_path)
    return _resample_audio_array(audio_signal, sample_rate, target_sample_rate)


def _prepare_separator_input_audio(
    audio_path: str,
    output_dir: str,
    target_sample_rate: int,
) -> str:
    sample_rate, audio_signal = read_audio(audio_path)
    prepared_signal = _resample_audio_array(
        audio_signal,
        sample_rate,
        target_sample_rate,
    )
    prepared_path = str(Path(output_dir) / "prepared_input.wav")
    save_audio(
        destination_path=prepared_path,
        audio_signal=prepared_signal,
        sample_rate=target_sample_rate,
        bit_depth=32,
    )
    return prepared_path


def _apply_stage_to_single_output(
    source_path: str,
    stage: SeparatorModelStage,
    stage_name: str,
    output_root: str,
    target_sample_rate: int,
    *,
    shifts: int = 2,
) -> str:
    _model_name, output_files = _run_separator_stage(
        source_path,
        stage,
        str(Path(output_root) / stage_name),
        target_sample_rate,
        shifts=shifts,
    )
    selected_output = _select_stage_output(output_files, stage.preferred_stems)
    if selected_output is not None:
        return selected_output
    if stage.required:
        raise FileNotFoundError("Stem separation failed")
    return source_path


def _write_selected_stage_outputs(
    selected_stage_paths: dict[str, str],
    output_root: str,
    target_sample_rate: int,
) -> dict[str, str]:
    aligned_signals: dict[str, np.ndarray] = {}
    target_length = 0
    for stem_name, stem_path in selected_stage_paths.items():
        stem_signal = _read_stage_signal(stem_path, target_sample_rate)
        aligned_signals[stem_name] = stem_signal
        target_length = max(target_length, int(stem_signal.shape[-1]))

    peak = 0.0
    for stem_name, stem_signal in tuple(aligned_signals.items()):
        aligned_signal = _align_audio_length(stem_signal, target_length)
        aligned_signals[stem_name] = aligned_signal
        if aligned_signal.size:
            peak = max(peak, float(np.max(np.abs(aligned_signal))))

    if peak > 0.98:
        scale = 0.98 / peak
        aligned_signals = {
            stem_name: stem_signal * scale
            for stem_name, stem_signal in aligned_signals.items()
        }

    final_dir = Path(output_root) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    written_paths: dict[str, str] = {}
    for stem_name, stem_signal in aligned_signals.items():
        destination_path = str(final_dir / f"{stem_name}.wav")
        written_paths[stem_name] = save_audio(
            destination_path=destination_path,
            audio_signal=stem_signal,
            sample_rate=target_sample_rate,
            bit_depth=32,
        )
    return written_paths


def _run_mastering_separator_pipeline(
    audio_path: str,
    output_root: str,
    plan: MasteringSeparatorPlan,
    *,
    shifts: int = 2,
) -> dict[str, str]:
    working_mix_path = _prepare_separator_input_audio(
        audio_path,
        output_root,
        plan.target_sample_rate,
    )
    for stage_index, stage in enumerate(plan.preprocess_stages):
        working_mix_path = _apply_stage_to_single_output(
            working_mix_path,
            stage,
            f"preprocess_{stage_index}",
            output_root,
            plan.target_sample_rate,
            shifts=shifts,
        )

    isolated_vocal_path: str | None = None
    instrumental_reference_path: str | None = None
    if plan.vocal_pair_stage is not None:
        _model_name, vocal_pair_outputs = _run_separator_stage(
            working_mix_path,
            plan.vocal_pair_stage,
            str(Path(output_root) / "vocal_pair"),
            plan.target_sample_rate,
            shifts=shifts,
        )
        isolated_vocal_path = _select_stage_output(
            vocal_pair_outputs,
            ("vocals",),
        )
        instrumental_reference_path = _select_stage_output(
            vocal_pair_outputs,
            ("instrumental", "other"),
        )
        if isolated_vocal_path is None or instrumental_reference_path is None:
            raise FileNotFoundError("Mastering stem separation failed")
    else:
        instrumental_reference_path = _apply_stage_to_single_output(
            working_mix_path,
            plan.reference_split_stage,
            "reference_split",
            output_root,
            plan.target_sample_rate,
            shifts=shifts,
        )

    four_stem_input_path = instrumental_reference_path or working_mix_path
    _four_stem_model_name, four_stem_outputs = _run_separator_stage(
        four_stem_input_path,
        plan.four_stem_stage,
        str(Path(output_root) / "four_stem"),
        plan.target_sample_rate,
        shifts=shifts,
    )
    drums_path = _select_stage_output(four_stem_outputs, ("drums",))
    bass_path = _select_stage_output(four_stem_outputs, ("bass",))
    other_path = _select_stage_output(four_stem_outputs, ("other",))
    if drums_path is None or bass_path is None or other_path is None:
        raise FileNotFoundError("Mastering stem separation failed")

    if isolated_vocal_path is None:
        isolated_vocal_path = _apply_stage_to_single_output(
            working_mix_path,
            plan.vocal_stage,
            "vocal_isolation",
            output_root,
            plan.target_sample_rate,
            shifts=shifts,
        )
    restored_vocal_path = _apply_stage_to_single_output(
        isolated_vocal_path,
        plan.vocal_restoration_stage,
        "vocal_restoration",
        output_root,
        plan.target_sample_rate,
        shifts=shifts,
    )
    cleaned_stage_paths = _run_separator_stage_batch(
        {
            "drums": drums_path,
            "bass": bass_path,
            "other": other_path,
        },
        plan.instrumental_cleanup_stage,
        str(Path(output_root) / "instrumental_cleanup"),
        plan.target_sample_rate,
        shifts=shifts,
    )
    cleaned_drums_path = cleaned_stage_paths["drums"]
    cleaned_bass_path = cleaned_stage_paths["bass"]
    cleaned_other_path = cleaned_stage_paths["other"]

    return _write_selected_stage_outputs(
        {
            "vocals": restored_vocal_path,
            "drums": cleaned_drums_path,
            "bass": cleaned_bass_path,
            "other": cleaned_other_path,
        },
        output_root,
        plan.target_sample_rate,
    )


def _run_vocal_pair_separator_pipeline(
    audio_path: str,
    output_root: str,
    *,
    model_name: str,
    shifts: int = 2,
) -> dict[str, str]:
    input_sample_rate, _input_signal = read_audio(audio_path)
    target_sample_rate = _resolve_mastering_sample_rate(input_sample_rate)
    prepared_mix_path = _prepare_separator_input_audio(
        audio_path,
        output_root,
        target_sample_rate,
    )
    stage = _build_vocal_pair_stage(model_name)
    _model_name, output_files = _run_separator_stage(
        prepared_mix_path,
        stage,
        str(Path(output_root) / "vocal_pair"),
        target_sample_rate,
        shifts=shifts,
    )
    vocals_path = _select_stage_output(output_files, ("vocals",))
    instrumental_path = _select_stage_output(
        output_files,
        ("instrumental", "other"),
    )
    if vocals_path is None or instrumental_path is None:
        raise FileNotFoundError("Stem separation failed")
    return _write_selected_stage_outputs(
        {
            "vocals": vocals_path,
            "no_vocals": instrumental_path,
        },
        output_root,
        target_sample_rate,
    )


def separate_stem_layers(
    audio_path: str,
    *,
    model_name: str = "mastering",
    shifts: int = 2,
    two_stems: str | None = None,
    output_dir: str | None = None,
    quality_flags: Sequence[str] = (),
) -> tuple[dict[str, str], str]:
    from definers.system.output_paths import managed_output_session_dir

    resolved_two_stems = None
    if two_stems is not None:
        resolved_two_stems = _normalize_model_name(
            two_stems,
            option_name="separator two-stems value",
        ).lower()
    resolved_output_dir = output_dir or managed_output_session_dir(
        "audio/stems",
        stem=Path(audio_path).stem,
    )
    owns_output_dir = output_dir is None

    try:
        if resolved_two_stems is not None:
            if resolved_two_stems != "vocals":
                raise ValueError(f"Unsupported two-stems value: {two_stems}")
            stem_paths = _run_vocal_pair_separator_pipeline(
                audio_path,
                resolved_output_dir,
                model_name=model_name,
                shifts=shifts,
            )
            return stem_paths, str(resolved_output_dir)

        input_sample_rate, _input_signal = read_audio(audio_path)
        plan = build_mastering_separator_plan(
            input_sample_rate,
            quality_flags=quality_flags,
            model_name=model_name,
        )
        _prefetch_mastering_plan_models(plan)
        stem_paths = _run_mastering_separator_pipeline(
            audio_path,
            resolved_output_dir,
            plan,
            shifts=shifts,
        )
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
    from definers.system.output_paths import managed_output_session_dir

    output_dir = managed_output_session_dir(
        "audio/stems",
        stem=Path(audio_path).stem,
    )
    try:
        stem_paths, _resolved_output_dir = separate_stem_layers(
            audio_path,
            model_name="mastering",
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
    final_output_dir = Path(output_dir) / "exports"
    final_output_dir.mkdir(parents=True, exist_ok=True)

    def export_stem(chosen_stem_path, suffix):
        sr, sound = read_audio(chosen_stem_path)
        normalized_format = (
            str(format_choice).strip().lower().lstrip(".") or "wav"
        )
        output_stem = str(
            final_output_dir
            / f"{Path(audio_path).stem}{suffix}.{normalized_format}"
        )
        return save_audio(
            audio_signal=sound,
            destination_path=output_stem,
            sample_rate=sr,
        )

    if separation_type == "acapella":
        voice = export_stem(vocals_path, "_acapella")
        return normalize_audio_to_peak(voice, output_path=voice)
    if separation_type == "karaoke":
        music = export_stem(accompaniment_path, "_karaoke")
        return normalize_audio_to_peak(music, output_path=music)

    voice = normalize_audio_to_peak(
        export_stem(vocals_path, "_acapella"),
        output_path=str(
            final_output_dir
            / f"{Path(audio_path).stem}_acapella.{str(format_choice).strip().lower().lstrip('.') or 'wav'}"
        ),
    )
    music = normalize_audio_to_peak(
        export_stem(accompaniment_path, "_karaoke"),
        output_path=str(
            final_output_dir
            / f"{Path(audio_path).stem}_karaoke.{str(format_choice).strip().lower().lstrip('.') or 'wav'}"
        ),
    )
    return voice, music


def stem_mixer(files, format_choice):
    import pydub
    from scipy.io.wavfile import write as write_wav

    librosa = librosa_module()
    from definers.system.output_paths import managed_output_path

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
    normalized_format = str(format_choice).strip().lower().lstrip(".") or "wav"
    output_stem = managed_output_path(
        normalized_format,
        section="audio",
        stem=f"stem_mix_{random_string()}",
    )
    output_path = save_audio(
        audio_signal=sound,
        destination_path=output_stem,
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
