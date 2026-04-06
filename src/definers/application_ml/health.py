from __future__ import annotations

import importlib
from dataclasses import dataclass

_UNSET = object()


@dataclass(frozen=True, slots=True)
class MlHealthSnapshot:
    training_ready: bool
    data_preparation_ready: bool
    answer_runtime_ready: bool
    available_prediction_targets: tuple[str, ...]
    missing_capabilities: tuple[str, ...]
    recommended_extras: tuple[str, ...]


class MlHealthService:
    @staticmethod
    def _import_symbol(module_name: str, symbol_name: str):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            return None
        return getattr(module, symbol_name, None)

    @classmethod
    def collect_ml_health_snapshot(
        cls,
        *,
        prepare_data_fn=_UNSET,
        create_vectorizer_fn=_UNSET,
        numpy_to_cupy_fn=_UNSET,
        cupy_to_numpy_fn=_UNSET,
        reshape_numpy_fn=_UNSET,
        features_to_audio_fn=_UNSET,
        features_to_image_fn=_UNSET,
        features_to_video_fn=_UNSET,
        models=_UNSET,
        processors=_UNSET,
    ) -> MlHealthSnapshot:
        if prepare_data_fn is _UNSET:
            prepare_data_fn = cls._import_symbol(
                "definers.application_data.preparation",
                "prepare_data",
            )
        if create_vectorizer_fn is _UNSET:
            create_vectorizer_fn = cls._import_symbol(
                "definers.application_data.vectorizers",
                "create_vectorizer",
            )
        if numpy_to_cupy_fn is _UNSET:
            numpy_to_cupy_fn = cls._import_symbol(
                "definers.application_data.arrays",
                "numpy_to_cupy",
            )
        if cupy_to_numpy_fn is _UNSET:
            cupy_to_numpy_fn = cls._import_symbol(
                "definers.application_data.arrays",
                "cupy_to_numpy",
            )
        if reshape_numpy_fn is _UNSET:
            reshape_numpy_fn = cls._import_symbol(
                "definers.application_data.arrays",
                "reshape_numpy",
            )
        if features_to_audio_fn is _UNSET:
            features_to_audio_fn = cls._import_symbol(
                "definers.audio",
                "features_to_audio",
            )
        if features_to_image_fn is _UNSET:
            features_to_image_fn = cls._import_symbol(
                "definers.image",
                "features_to_image",
            )
        if features_to_video_fn is _UNSET:
            features_to_video_fn = cls._import_symbol(
                "definers.media.video_helpers",
                "features_to_video",
            )
        if models is _UNSET:
            models = cls._import_symbol("definers.constants", "MODELS")
        if processors is _UNSET:
            processors = cls._import_symbol("definers.constants", "PROCESSORS")

        training_ready = all(
            (
                create_vectorizer_fn,
                numpy_to_cupy_fn,
                cupy_to_numpy_fn,
                reshape_numpy_fn,
            )
        )
        data_preparation_ready = all((prepare_data_fn, create_vectorizer_fn))
        answer_runtime_ready = models is not None and processors is not None

        prediction_targets = ["text"]
        if features_to_audio_fn is not None:
            prediction_targets.append("audio")
        if features_to_image_fn is not None:
            prediction_targets.append("image")
        if features_to_video_fn is not None:
            prediction_targets.append("video")

        missing_capabilities = []
        recommended_extras = []
        if not training_ready:
            missing_capabilities.append("training-array-pipeline")
            recommended_extras.append("dev")
        if not data_preparation_ready:
            missing_capabilities.append("data-preparation")
            recommended_extras.append("ml")
        if not answer_runtime_ready:
            missing_capabilities.append("answer-runtime")
            recommended_extras.append("ml")
        if features_to_audio_fn is None:
            recommended_extras.append("audio")
        if features_to_image_fn is None:
            recommended_extras.append("image")
        if features_to_video_fn is None:
            recommended_extras.append("video")

        return MlHealthSnapshot(
            training_ready=training_ready,
            data_preparation_ready=data_preparation_ready,
            answer_runtime_ready=answer_runtime_ready,
            available_prediction_targets=tuple(prediction_targets),
            missing_capabilities=tuple(dict.fromkeys(missing_capabilities)),
            recommended_extras=tuple(dict.fromkeys(recommended_extras)),
        )

    @staticmethod
    def validate_ml_health_snapshot(snapshot: MlHealthSnapshot):
        missing = []
        if not snapshot.training_ready:
            missing.append("training-array-pipeline")
        if not snapshot.data_preparation_ready:
            missing.append("data-preparation")
        if missing:
            raise LookupError("Missing ML capabilities: " + ", ".join(missing))
        return snapshot

    @staticmethod
    def render_ml_health_markdown(snapshot: MlHealthSnapshot) -> str:
        lines = [
            "## ML Health",
            f"- Training Ready: {snapshot.training_ready}",
            f"- Data Preparation Ready: {snapshot.data_preparation_ready}",
            f"- Answer Runtime Ready: {snapshot.answer_runtime_ready}",
            "- Prediction Targets: "
            + ", ".join(snapshot.available_prediction_targets),
            "- Missing Capabilities: "
            + (", ".join(snapshot.missing_capabilities) or "none"),
            "- Recommended Extras: "
            + (", ".join(snapshot.recommended_extras) or "none"),
        ]
        return "\n".join(lines)

    @classmethod
    def collect_live_ml_health_snapshot(cls):
        return cls.collect_ml_health_snapshot()

    @classmethod
    def run_ml_health_check(cls):
        return cls.validate_ml_health_snapshot(
            cls.collect_live_ml_health_snapshot()
        )


collect_ml_health_snapshot = MlHealthService.collect_ml_health_snapshot
validate_ml_health_snapshot = MlHealthService.validate_ml_health_snapshot
render_ml_health_markdown = MlHealthService.render_ml_health_markdown
collect_live_ml_health_snapshot = (
    MlHealthService.collect_live_ml_health_snapshot
)
run_ml_health_check = MlHealthService.run_ml_health_check
