import gradio.blocks as _gr_blocks
import pytest

_orig_bc_init = _gr_blocks.BlockContext.__init__


def _patched_bc_init(self, *args, **kwargs):
    kwargs.pop("css", None)
    kwargs.pop("theme", None)
    return _orig_bc_init(self, *args, **kwargs)


_gr_blocks.BlockContext.__init__ = _patched_bc_init


def test_start_audio(monkeypatch):

    import definers._system as _sys

    monkeypatch.setattr(_sys, "install_ffmpeg", lambda *a, **k: None)
    monkeypatch.setattr(_sys, "install_audio_effects", lambda *a, **k: None)
    monkeypatch.setattr("gradio.Blocks.launch", lambda *args, **kwargs: None)

    called = {"audio": False}

    def fake_helper():
        called["audio"] = True

    monkeypatch.setattr("definers._chat._gui_audio", fake_helper)

    from definers._chat import start

    start("audio")
    assert called["audio"]


def test_start_train(monkeypatch):

    import definers._system as _sys

    monkeypatch.setattr(_sys, "install_ffmpeg", lambda *a, **k: None)
    monkeypatch.setattr(_sys, "install_audio_effects", lambda *a, **k: None)
    monkeypatch.setattr("gradio.Blocks.launch", lambda *args, **kwargs: None)
    called = {"train": False}

    from definers._chat import _gui_train

    def fake_helper():
        called["train"] = True

    monkeypatch.setattr("definers._chat._gui_train", fake_helper)

    from definers._chat import start

    start("train")
    assert called["train"]


def test_start_invalid(monkeypatch):

    import definers._system as _sys

    monkeypatch.setattr(_sys, "install_ffmpeg", lambda *a, **k: None)
    monkeypatch.setattr(_sys, "install_audio_effects", lambda *a, **k: None)
    from definers import _system

    def fake_catch(e):
        raise ValueError(e)

    import definers._chat as _chat

    monkeypatch.setattr(_chat, "catch", fake_catch)
    from definers._chat import start

    with pytest.raises(ValueError) as exc:
        start("not-a-real-proj")
    assert "No project called" in str(exc.value)


def test_gui_audio_builds(monkeypatch):
    import definers._system as _sys

    monkeypatch.setattr(_sys, "install_ffmpeg", lambda *a, **k: None)
    monkeypatch.setattr(_sys, "install_audio_effects", lambda *a, **k: None)
    monkeypatch.setattr("gradio.Blocks.launch", lambda *args, **kwargs: None)

    import definers

    monkeypatch.setattr(definers, "init_stable_whisper", lambda *a, **k: None)
    monkeypatch.setattr(definers, "init_pretrained_model", lambda *a, **k: None)
    monkeypatch.setattr(definers, "set_system_message", lambda *a, **k: None)
    monkeypatch.setattr(definers, "train_model_rvc", lambda *a, **k: None)
    monkeypatch.setattr(definers, "init_chat", lambda *a, **k: None)

    monkeypatch.setattr(definers, "language_codes", {})
    monkeypatch.setattr(definers, "theme", lambda *a, **k: None)
    monkeypatch.setattr(definers, "css", lambda *a, **k: "")

    called = {"audio": False}

    from definers._chat import _gui_audio

    def fake_helper():
        called["audio"] = True

    monkeypatch.setattr("definers._chat._gui_audio", fake_helper)

    from definers._chat import start

    start("audio")
    assert called["audio"]


def test_start_video(monkeypatch):

    import definers._system as _sys

    monkeypatch.setattr(_sys, "install_ffmpeg", lambda *a, **k: None)
    monkeypatch.setattr(_sys, "install_audio_effects", lambda *a, **k: None)
    monkeypatch.setattr("gradio.Blocks.launch", lambda *args, **kwargs: None)
    called = {"video": False}

    def fake():
        called["video"] = True

    monkeypatch.setattr("definers._chat._gui_video", fake)
    from definers._chat import start

    start("video")
    assert called["video"]


def test_gui_video_builds(monkeypatch):

    import definers._system as _sys

    monkeypatch.setattr(_sys, "install_ffmpeg", lambda *a, **k: None)
    monkeypatch.setattr(_sys, "install_audio_effects", lambda *a, **k: None)
    monkeypatch.setattr("gradio.Blocks.launch", lambda *args, **kwargs: None)

    from definers._chat import _gui_video

    _gui_video()


def test_video_helpers(monkeypatch):

    monkeypatch.setattr(
        "definers._chat.lyric_video", lambda *a, **k: "/fake/video.mp4"
    )
    monkeypatch.setattr(
        "definers._chat.music_video", lambda *a, **k: "/fake/video.mp4"
    )
    from definers._chat import lyric_video, music_video

    assert lyric_video("a", "b", "c", "bottom") == "/fake/video.mp4"
    assert music_video("a", 320, 240, 15) == "/fake/video.mp4"
