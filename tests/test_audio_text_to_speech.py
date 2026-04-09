import sys
import types

import numpy as np

from definers.audio import text_to_speech, voice


class FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)
        self.device = None

    def to(self, device_name):
        self.device = device_name
        return self

    def squeeze(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.value, dtype=np.float32)


def test_local_text_to_speech_generate_uses_tokenizer_model_and_reference(
    monkeypatch,
):
    tokenizer_calls = []
    model_calls = []

    class FakeTokenizer:
        def __call__(self, text, return_tensors=None):
            tokenizer_calls.append((text, return_tensors))
            return {"input_ids": FakeTensor([1.0, 2.0, 3.0])}

    class FakeModel:
        def __call__(self, **kwargs):
            model_calls.append(kwargs)
            return types.SimpleNamespace(waveform=FakeTensor([0.4, -0.2]))

    fake_torch = types.SimpleNamespace(no_grad=FakeNoGrad)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        text_to_speech,
        "_apply_reference_style",
        lambda audio_signal, sample_rate, reference_audio_path: (
            np.asarray(audio_signal, dtype=np.float32) * 0.5
        ),
    )

    backend = text_to_speech.LocalTextToSpeech(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        sample_rate=22050,
        device_name="cpu",
    )

    audio_signal, sample_rate = backend.generate(
        text="  hello   world  ",
        reference_audio_path="reference.wav",
    )

    assert tokenizer_calls == [("hello world", "pt")]
    assert model_calls and model_calls[0]["input_ids"].device == "cpu"
    assert sample_rate == 22050
    assert np.allclose(audio_signal, np.array([0.2, -0.1], dtype=np.float32))


def test_generate_voice_writes_backend_audio(monkeypatch):
    original_model = voice.MODELS.get("tts")

    class FakeTts:
        def __init__(self):
            self.calls = []

        def generate(self, *, text, reference_audio_path=None):
            self.calls.append((text, reference_audio_path))
            return np.array([0.25, -0.5, 0.125], dtype=np.float32), 16000

    fake_tts = FakeTts()
    monkeypatch.setitem(voice.MODELS, "tts", fake_tts)

    temp_paths = iter(["generated.wav"])
    monkeypatch.setattr(
        voice,
        "tmp",
        lambda extension=None, keep=True: next(temp_paths),
    )
    monkeypatch.setattr(
        "definers.system.output_paths.managed_output_path",
        lambda suffix=None, *, section, stem, filename=None, unique=True: (
            "managed/generated_voice.mp3"
        ),
    )

    written = {}
    saved = {}
    fake_sound = object()
    fake_soundfile = types.SimpleNamespace(
        write=lambda path, data, sample_rate: written.update(
            {
                "path": path,
                "data": np.asarray(data, dtype=np.float32),
                "sample_rate": sample_rate,
            }
        )
    )
    fake_pydub = types.SimpleNamespace(
        AudioSegment=types.SimpleNamespace(from_file=lambda path: fake_sound)
    )
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)
    monkeypatch.setitem(sys.modules, "pydub", fake_pydub)
    monkeypatch.setattr(
        voice,
        "save_audio",
        lambda **kwargs: saved.update(kwargs) or "voice_output.mp3",
    )

    try:
        result = voice.generate_voice("hello there", "ref.wav", "mp3")
    finally:
        voice.MODELS["tts"] = original_model

    assert result == "voice_output.mp3"
    assert fake_tts.calls == [("hello there", "ref.wav")]
    assert written["path"] == "generated.wav"
    assert written["sample_rate"] == 16000
    assert np.isclose(np.max(np.abs(written["data"])), 0.9)
    assert saved["destination_path"] == "managed/generated_voice.mp3"
    assert saved["audio_signal"] is fake_sound
    assert saved["sample_rate"] == 16000
    assert "output_format" not in saved


def test_generate_voice_returns_none_when_backend_fails(monkeypatch):
    original_model = voice.MODELS.get("tts")

    class BrokenTts:
        def generate(self, *, text, reference_audio_path=None):
            raise RuntimeError("backend failure")

    caught_messages = []
    monkeypatch.setitem(voice.MODELS, "tts", BrokenTts())
    monkeypatch.setattr(voice, "catch", caught_messages.append)
    monkeypatch.setitem(
        sys.modules,
        "soundfile",
        types.SimpleNamespace(write=lambda *args, **kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "pydub",
        types.SimpleNamespace(
            AudioSegment=types.SimpleNamespace(from_file=lambda path: object())
        ),
    )

    try:
        result = voice.generate_voice("hello", None, "wav")
    finally:
        voice.MODELS["tts"] = original_model

    assert result is None
    assert caught_messages == ["Generation failed: backend failure"]
