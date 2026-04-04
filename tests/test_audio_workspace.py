from contextlib import nullcontext

from definers.presentation.apps.audio_workspace import (
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


def test_prepare_audio_workspace_initializes_required_models(monkeypatch):
    initialized_models = []
    system_message = {}

    monkeypatch.setattr("definers.system.install_audio_effects", lambda: None)
    monkeypatch.setattr("definers.system.install_ffmpeg", lambda: None)
    monkeypatch.setattr("definers.system.cwd", lambda: nullcontext())
    monkeypatch.setattr("definers.system.exist", lambda path: False)
    monkeypatch.setattr("definers.chat.init_stable_whisper", lambda: None)
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
    assert initialized_models == [
        "tts",
        "svc",
        "speech-recognition",
        "audio-classification",
        "music",
        "summary",
        "answer",
        "translate",
    ]
    assert system_message["name"] == "Definers Audio Assistant"
    assert "Support Chat" in AUDIO_TOOL_MAP
