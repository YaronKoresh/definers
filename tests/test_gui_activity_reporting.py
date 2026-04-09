import sys
import types


def test_image_generate_image_reports_granular_activity(monkeypatch):
    import definers.text.validation as validation_module
    from definers.ui.apps.image import ImageApp

    activity = []

    monkeypatch.setattr(
        "definers.system.download_activity.report_download_activity",
        lambda item_label=None, **kwargs: activity.append(item_label),
    )
    monkeypatch.setattr(
        validation_module.TextInputValidator,
        "default",
        staticmethod(
            lambda: types.SimpleNamespace(
                validate=lambda value: f"clean:{value}"
            )
        ),
    )
    monkeypatch.setattr(
        "definers.image.get_max_resolution",
        lambda width, height, mega_pixels=0.25, factor=16: (512, 768),
    )
    monkeypatch.setattr(
        "definers.ml.optimize_prompt_realism",
        lambda text: f"opt:{text}",
    )
    monkeypatch.setattr(
        "definers.ml.pipe",
        lambda *args, **kwargs: "generated.png",
    )

    result = ImageApp.generate_image("prompt", 1, 1)

    assert result == "generated.png"
    assert activity == [
        "Validate prompt",
        "Resolve canvas",
        "Prepare prompt",
        "Generate image",
    ]


def test_translate_app_reports_target_language_activity(monkeypatch):
    import definers.text as text_module
    from definers.ui.apps.translate import TranslateApp

    activity = []

    monkeypatch.setattr(
        "definers.system.download_activity.report_download_activity",
        lambda item_label=None, **kwargs: activity.append(
            (item_label, kwargs.get("detail"))
        ),
    )
    monkeypatch.setattr(
        text_module,
        "ai_translate",
        lambda text, lang: f"{text}:{lang}",
    )

    result = TranslateApp.translate_text("shalom", "english")

    assert result == "shalom:en"
    assert activity == [
        ("Validate text", "Checking the translation input."),
        ("Resolve target language", "Using target language 'en'."),
        ("Translate paragraphs", "Running the translation workflow."),
    ]


def test_animation_reset_state_reports_granular_activity(monkeypatch):
    from definers.ui.apps.animation import AnimationApp

    activity = []
    removed_paths = []

    monkeypatch.setattr(
        "definers.system.download_activity.report_download_activity",
        lambda item_label=None, **kwargs: activity.append(item_label),
    )
    monkeypatch.setitem(
        sys.modules,
        "gradio",
        types.SimpleNamespace(update=lambda **kwargs: kwargs),
    )
    monkeypatch.setattr(
        "definers.system.output_paths.managed_output_session_dir",
        lambda section, stem=None: f"/tmp/{section}/{stem}",
    )
    monkeypatch.setattr(
        "definers.system.output_paths.cleanup_managed_output_path",
        lambda path: removed_paths.append(path) or True,
    )

    chunk_state = {
        "current_chunk": 4,
        "chunk_paths": ["chunk_1.gif"],
        "chunks_path": "old-path",
    }

    updated_state, latest_chunk, combine_button, generate_button = (
        AnimationApp.reset_state(chunk_state)
    )

    assert updated_state["current_chunk"] == 1
    assert updated_state["chunk_paths"] == []
    assert updated_state["chunks_path"] == "/tmp/animation/chunks"
    assert removed_paths == ["old-path"]
    assert latest_chunk is None
    assert combine_button == {"visible": False}
    assert generate_button == {"interactive": True}
    assert activity == [
        "Clear chunk session",
        "Create chunk workspace",
        "Reset animation controls",
    ]


def test_build_faiss_reports_clone_stage(monkeypatch):
    import pytest

    import definers.ml as ml_module

    activity = []

    monkeypatch.setattr(
        "definers.system.download_activity.report_download_activity",
        lambda item_label=None, **kwargs: activity.append(item_label),
    )
    monkeypatch.setattr(
        "definers.system.output_paths.managed_output_session_dir",
        lambda section, stem=None: f"/tmp/{stem or section}",
    )
    monkeypatch.setattr(ml_module, "catch", lambda *args, **kwargs: None)

    def fail_clone(*_args, **_kwargs):
        raise RuntimeError("stop after clone stage")

    monkeypatch.setattr(ml_module, "git", fail_clone)

    with pytest.raises(RuntimeError, match="stop after clone stage"):
        ml_module.build_faiss()
    assert activity[0] == "Clone FAISS source"
