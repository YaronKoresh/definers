from types import SimpleNamespace

from definers.application_ml.answer_history_preparer import (
    AnswerHistoryPreparer,
)
from definers.application_ml.answer_service import AnswerService


class TrackingDependencyLoader:
    def __init__(
        self,
        *,
        image_module=None,
        soundfile_module=None,
        librosa_module=None,
    ):
        self.image_module = image_module
        self.soundfile_module = soundfile_module
        self.librosa_module = librosa_module
        self.image_calls = 0
        self.soundfile_calls = 0
        self.librosa_calls = 0

    def load_image_module(self):
        self.image_calls += 1
        return self.image_module

    def load_soundfile_module(self):
        self.soundfile_calls += 1
        return self.soundfile_module

    def load_librosa_module(self):
        self.librosa_calls += 1
        return self.librosa_module


def runtime_stub(*, model=None, processor=None):
    return SimpleNamespace(
        MODELS={"answer": model},
        PROCESSORS={"answer": processor},
        SYSTEM_MESSAGE="system",
        common_audio_formats=["wav", "mp3"],
        iio_formats=["jpg", "png"],
    )


def test_prepare_answer_history_keeps_text_only_path_dependency_free():
    loader = TrackingDependencyLoader(
        image_module=object(),
        soundfile_module=object(),
        librosa_module=object(),
    )

    prepared_history, image_items, audio_items = (
        AnswerHistoryPreparer.prepare_answer_history(
            [{"role": "user", "content": "plain english"}],
            runtime_stub(),
            loader,
        )
    )

    assert prepared_history == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "plain english"},
    ]
    assert image_items == []
    assert audio_items == []
    assert loader.image_calls == 0
    assert loader.soundfile_calls == 0
    assert loader.librosa_calls == 0


def test_prepare_answer_history_reuses_image_dependency(monkeypatch):
    import definers.system as system_module
    from definers.application_ml.answer_image_loader import AnswerImageLoader

    image_module = object()
    loader = TrackingDependencyLoader(image_module=image_module)
    returned_images = [object(), object()]

    monkeypatch.setattr(
        system_module,
        "get_ext",
        lambda path: path.rsplit(".", 1)[-1],
    )
    monkeypatch.setattr(system_module, "read", lambda path: f"read:{path}")
    image_reads = []

    def fake_read_answer_image(path, current_image_module):
        image_reads.append((path, current_image_module))
        return returned_images[len(image_reads) - 1]

    monkeypatch.setattr(
        AnswerImageLoader,
        "read_answer_image",
        staticmethod(fake_read_answer_image),
    )

    prepared_history, image_items, audio_items = (
        AnswerHistoryPreparer.prepare_answer_history(
            [
                {
                    "role": "user",
                    "content": (
                        {"path": "one.jpg"},
                        {"path": "two.png"},
                    ),
                }
            ],
            runtime_stub(),
            loader,
        )
    )

    assert prepared_history == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "<|image_1|> <|image_2|>"},
    ]
    assert image_items == returned_images
    assert audio_items == []
    assert loader.image_calls == 1
    assert loader.soundfile_calls == 0
    assert loader.librosa_calls == 0
    assert image_reads == [
        ("one.jpg", image_module),
        ("two.png", image_module),
    ]


def test_prepare_answer_history_uses_soundfile_without_loading_librosa(
    monkeypatch,
):
    import definers.system as system_module
    from definers.application_ml.answer_audio_loader import AnswerAudioLoader

    soundfile_module = object()
    loader = TrackingDependencyLoader(
        soundfile_module=soundfile_module,
        librosa_module=object(),
    )
    returned_audios = [("a", 16000), ("b", 16000)]

    monkeypatch.setattr(
        system_module,
        "get_ext",
        lambda path: path.rsplit(".", 1)[-1],
    )
    monkeypatch.setattr(system_module, "read", lambda path: f"read:{path}")
    audio_reads = []

    def fake_read_answer_audio(path, current_soundfile_module, current_librosa):
        audio_reads.append((path, current_soundfile_module, current_librosa))
        if current_soundfile_module is soundfile_module:
            return returned_audios[len(audio_reads) - 1]
        return None

    monkeypatch.setattr(
        AnswerAudioLoader,
        "read_answer_audio",
        staticmethod(fake_read_answer_audio),
    )

    prepared_history, image_items, audio_items = (
        AnswerHistoryPreparer.prepare_answer_history(
            [
                {
                    "role": "user",
                    "content": (
                        {"path": "one.wav"},
                        {"path": "two.mp3"},
                    ),
                }
            ],
            runtime_stub(),
            loader,
        )
    )

    assert prepared_history == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "<|audio_1|> <|audio_2|>"},
    ]
    assert image_items == []
    assert audio_items == returned_audios
    assert loader.image_calls == 0
    assert loader.soundfile_calls == 1
    assert loader.librosa_calls == 0
    assert audio_reads == [
        ("one.wav", soundfile_module, None),
        ("two.mp3", soundfile_module, None),
    ]


def test_prepare_answer_history_falls_back_to_librosa_only_after_failure(
    monkeypatch,
):
    import definers.system as system_module
    from definers.application_ml.answer_audio_loader import AnswerAudioLoader

    soundfile_module = object()
    librosa_module = object()
    loader = TrackingDependencyLoader(
        soundfile_module=soundfile_module,
        librosa_module=librosa_module,
    )

    monkeypatch.setattr(
        system_module,
        "get_ext",
        lambda path: path.rsplit(".", 1)[-1],
    )
    monkeypatch.setattr(system_module, "read", lambda path: f"read:{path}")
    audio_reads = []

    def fake_read_answer_audio(path, current_soundfile_module, current_librosa):
        audio_reads.append((path, current_soundfile_module, current_librosa))
        if current_soundfile_module is soundfile_module:
            return None
        if current_librosa is librosa_module:
            return ("fallback", 16000)
        return None

    monkeypatch.setattr(
        AnswerAudioLoader,
        "read_answer_audio",
        staticmethod(fake_read_answer_audio),
    )

    prepared_history, image_items, audio_items = (
        AnswerHistoryPreparer.prepare_answer_history(
            [{"role": "user", "content": {"path": "clip.wav"}}],
            runtime_stub(),
            loader,
        )
    )

    assert prepared_history == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "<|audio_1|>"},
    ]
    assert image_items == []
    assert audio_items == [("fallback", 16000)]
    assert loader.image_calls == 0
    assert loader.soundfile_calls == 1
    assert loader.librosa_calls == 1
    assert audio_reads == [
        ("clip.wav", soundfile_module, None),
        ("clip.wav", None, librosa_module),
    ]


def test_answer_service_text_only_flow_never_touches_optional_dependencies():
    class NoMediaDependencyLoader:
        def load_image_module(self):
            raise AssertionError("image dependency should not load")

        def load_soundfile_module(self):
            raise AssertionError("soundfile dependency should not load")

        def load_librosa_module(self):
            raise AssertionError("librosa dependency should not load")

    model = SimpleNamespace(
        generate=lambda **kwargs: kwargs["prompt"],
    )

    response = AnswerService.answer(
        [{"role": "user", "content": "plain english"}],
        runtime_stub(model=model),
        dependency_loader=NoMediaDependencyLoader(),
    )

    assert response == (
        "<|system|>system<|end|><|user|>plain english<|end|><|assistant|>"
    )


def test_answer_service_returns_none_without_model():
    loader = TrackingDependencyLoader(
        image_module=object(),
        soundfile_module=object(),
        librosa_module=object(),
    )

    assert (
        AnswerService.answer(
            [{"role": "user", "content": {"path": "one.jpg"}}],
            runtime_stub(model=None),
            dependency_loader=loader,
        )
        is None
    )
    assert loader.image_calls == 0
    assert loader.soundfile_calls == 0
    assert loader.librosa_calls == 0


def test_prepare_answer_history_skips_unreadable_text_attachment(monkeypatch):
    import definers.system as system_module

    loader = TrackingDependencyLoader()

    monkeypatch.setattr(
        system_module,
        "get_ext",
        lambda path: path.rsplit(".", 1)[-1],
    )

    def fake_read(path):
        raise OSError(path)

    monkeypatch.setattr(system_module, "read", fake_read)

    prepared_history, image_items, audio_items = (
        AnswerHistoryPreparer.prepare_answer_history(
            [{"role": "user", "content": {"path": "notes.txt"}}],
            runtime_stub(),
            loader,
        )
    )

    assert prepared_history == [{"role": "system", "content": "system"}]
    assert image_items == []
    assert audio_items == []


def test_prepare_answer_history_handles_mixed_media_partial_failures(
    monkeypatch,
):
    import definers.system as system_module
    from definers.application_ml.answer_audio_loader import AnswerAudioLoader
    from definers.application_ml.answer_image_loader import AnswerImageLoader

    soundfile_module = object()
    librosa_module = object()
    image_module = object()
    loader = TrackingDependencyLoader(
        image_module=image_module,
        soundfile_module=soundfile_module,
        librosa_module=librosa_module,
    )

    monkeypatch.setattr(
        system_module,
        "get_ext",
        lambda path: path.rsplit(".", 1)[-1],
    )

    def fake_read(path):
        if path == "context.txt":
            return "context"
        raise OSError(path)

    monkeypatch.setattr(system_module, "read", fake_read)

    audio_reads = []
    image_reads = []

    def fake_read_answer_audio(path, current_soundfile_module, current_librosa):
        audio_reads.append((path, current_soundfile_module, current_librosa))
        if current_soundfile_module is soundfile_module:
            return None
        if current_librosa is librosa_module:
            return ("audio", 16000)
        return None

    def fake_read_answer_image(path, current_image_module):
        image_reads.append((path, current_image_module))
        return None

    monkeypatch.setattr(
        AnswerAudioLoader,
        "read_answer_audio",
        staticmethod(fake_read_answer_audio),
    )
    monkeypatch.setattr(
        AnswerImageLoader,
        "read_answer_image",
        staticmethod(fake_read_answer_image),
    )

    prepared_history, image_items, audio_items = (
        AnswerHistoryPreparer.prepare_answer_history(
            [
                {
                    "role": "user",
                    "content": (
                        {"path": "broken.jpg"},
                        {"path": "clip.wav"},
                        {"path": "context.txt"},
                        {"path": "missing.txt"},
                    ),
                }
            ],
            runtime_stub(),
            loader,
        )
    )

    assert prepared_history == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "<|audio_1|>\n\ncontext"},
    ]
    assert image_items == []
    assert audio_items == [("audio", 16000)]
    assert loader.image_calls == 1
    assert loader.soundfile_calls == 1
    assert loader.librosa_calls == 1
    assert image_reads == [("broken.jpg", image_module)]
    assert audio_reads == [
        ("clip.wav", soundfile_module, None),
        ("clip.wav", None, librosa_module),
    ]
