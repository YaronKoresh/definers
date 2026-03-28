from definers.application_ml.health import (
    collect_ml_health_snapshot,
    render_ml_health_markdown,
    validate_ml_health_snapshot,
)


def test_collect_ml_health_snapshot_reports_ready_stack():
    snapshot = collect_ml_health_snapshot(
        prepare_data_fn=object(),
        create_vectorizer_fn=object(),
        numpy_to_cupy_fn=object(),
        cupy_to_numpy_fn=object(),
        reshape_numpy_fn=object(),
        features_to_audio_fn=object(),
        features_to_image_fn=object(),
        features_to_video_fn=object(),
        models={},
        processors={},
    )

    assert snapshot.training_ready is True
    assert snapshot.data_preparation_ready is True
    assert snapshot.answer_runtime_ready is True
    assert snapshot.available_prediction_targets == (
        "text",
        "audio",
        "image",
        "video",
    )
    assert snapshot.missing_capabilities == ()


def test_validate_ml_health_snapshot_rejects_missing_training_pipeline():
    snapshot = collect_ml_health_snapshot(
        prepare_data_fn=object(),
        create_vectorizer_fn=None,
        numpy_to_cupy_fn=None,
        cupy_to_numpy_fn=None,
        reshape_numpy_fn=None,
        models={},
        processors={},
    )

    try:
        validate_ml_health_snapshot(snapshot)
    except LookupError as exc:
        assert str(exc) == (
            "Missing ML capabilities: training-array-pipeline, data-preparation"
        )
    else:
        raise AssertionError("expected LookupError")


def test_render_ml_health_markdown_lists_recommendations():
    snapshot = collect_ml_health_snapshot(
        prepare_data_fn=object(),
        create_vectorizer_fn=object(),
        numpy_to_cupy_fn=object(),
        cupy_to_numpy_fn=object(),
        reshape_numpy_fn=object(),
        features_to_audio_fn=None,
        features_to_image_fn=None,
        features_to_video_fn=None,
        models={},
        processors={},
    )

    markdown = render_ml_health_markdown(snapshot)

    assert "## ML Health" in markdown
    assert "Prediction Targets: text" in markdown
    assert "Recommended Extras: audio, image, video" in markdown