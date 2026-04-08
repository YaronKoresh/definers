from contextlib import nullcontext

from definers.ui.apps.audio_workspace import (
    AUDIO_TOOL_MAP,
    get_audio_language_choices,
    prepare_audio_workspace,
    train_voice_lab_model,
)


def test_get_audio_language_choices_sorts_and_deduplicates():
    result = get_audio_language_choices(
        {"en": "English", "en-US": "English", "he": "Hebrew"}
    )

    assert result == ["English", "Hebrew"]


def test_train_voice_lab_model_returns_incremented_level(monkeypatch):
    monkeypatch.setattr("definers.system.cwd", lambda: nullcontext())
    monkeypatch.setattr(
        "definers.ml.train_model_rvc",
        lambda experiment, inp, lvl: f"{experiment}:{inp}:{lvl}",
    )

    assert train_voice_lab_model("exp", "input.wav", 3) == (
        "exp:input.wav:3",
        4,
    )


def test_prepare_audio_workspace_defers_runtime_bootstrap(monkeypatch):
    initialized_models = []
    stable_whisper_calls = []
    ffmpeg_calls = []
    audio_effect_calls = []
    system_message = {}

    monkeypatch.setattr(
        "definers.system.install_audio_effects",
        lambda: audio_effect_calls.append("audio-effects"),
    )
    monkeypatch.setattr(
        "definers.system.install_ffmpeg",
        lambda: ffmpeg_calls.append("ffmpeg"),
    )
    monkeypatch.setattr("definers.system.cwd", lambda: nullcontext())
    monkeypatch.setattr("definers.system.exist", lambda path: False)
    monkeypatch.setattr(
        "definers.ui.lyric_video_service.init_stable_whisper",
        lambda: stable_whisper_calls.append("stable-whisper"),
    )
    monkeypatch.setattr(
        "definers.ml.init_pretrained_model",
        lambda model_name: initialized_models.append(model_name),
    )
    monkeypatch.setattr(
        "definers.text.set_system_message",
        lambda **kwargs: system_message.update(kwargs),
    )

    result = prepare_audio_workspace()

    assert result == {"svc_installed": False}
    assert initialized_models == []
    assert stable_whisper_calls == []
    assert ffmpeg_calls == []
    assert audio_effect_calls == []
    assert system_message["name"] == "Definers Audio Assistant"
    assert "Support Chat" in AUDIO_TOOL_MAP
    assert "Mastering Studio" in AUDIO_TOOL_MAP
    assert "Vocal Finishing" in AUDIO_TOOL_MAP
    assert "Audio Cleanup" in AUDIO_TOOL_MAP
    assert "Preview & Split" in AUDIO_TOOL_MAP
