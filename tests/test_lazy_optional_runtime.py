import sys
import types

from definers.ml.answer_service import AnswerService
from definers.ml.text_generation import TextGenerationService

from definers.constants import MODELS, TOKENIZERS
from definers.text import translation


class FakeTensor:
    def to(self, *_args, **_kwargs):
        return self


class FakeSummaryTokenizer:
    def __call__(self, *_args, **_kwargs):
        return {"input_ids": FakeTensor()}

    def decode(self, *_args, **_kwargs):
        return "summary-output"


class FakeSummaryModel:
    def generate(self, **_kwargs):
        return [[1]]


def test_summarize_bootstraps_summary_runtime(monkeypatch):
    summary_model = FakeSummaryModel()
    summary_tokenizer = FakeSummaryTokenizer()
    original_model = MODELS.get("summary")
    original_tokenizer = TOKENIZERS.get("summary")
    calls = []

    monkeypatch.setitem(MODELS, "summary", None)
    monkeypatch.setitem(TOKENIZERS, "summary", None)

    def fake_init_pretrained_model(task):
        calls.append(task)
        MODELS["summary"] = summary_model
        TOKENIZERS["summary"] = summary_tokenizer

    monkeypatch.setattr(
        "definers.ml.init_pretrained_model",
        fake_init_pretrained_model,
    )

    try:
        assert TextGenerationService.summarize("hello") == "summary-output"
    finally:
        MODELS["summary"] = original_model
        TOKENIZERS["summary"] = original_tokenizer

    assert calls == ["summary"]


def test_ai_translate_bootstraps_translate_runtime(monkeypatch):
    fake_model = object()
    fake_tokenizer = types.SimpleNamespace(tgt_lang=None, src_lang=None)
    original_model = MODELS.get("translate")
    original_tokenizer = TOKENIZERS.get("translate")
    calls = []

    monkeypatch.setitem(MODELS, "translate", None)
    monkeypatch.setitem(TOKENIZERS, "translate", None)
    monkeypatch.setattr(translation, "resolve_target_code", lambda lang: "en")
    monkeypatch.setattr(
        translation,
        "resolve_source_translation_context",
        lambda paragraph: ("he", "he"),
    )
    monkeypatch.setattr(
        translation,
        "get_sentence_splitter",
        lambda _code: lambda value: [value],
    )
    monkeypatch.setattr(
        translation,
        "translate_text_segment",
        lambda *args, **kwargs: "translated-text",
    )

    def fake_init_pretrained_model(task):
        calls.append(task)
        MODELS["translate"] = fake_model
        TOKENIZERS["translate"] = fake_tokenizer

    monkeypatch.setattr(
        "definers.ml.init_pretrained_model",
        fake_init_pretrained_model,
    )

    fake_sacremoses = types.SimpleNamespace(
        MosesPunctNormalizer=lambda lang: types.SimpleNamespace(
            normalize=lambda value: value
        )
    )

    try:
        with monkeypatch.context() as ctx:
            ctx.setitem(sys.modules, "sacremoses", fake_sacremoses)
            assert (
                translation.ai_translate("shalom", lang="english")
                == "translated-text"
            )
    finally:
        MODELS["translate"] = original_model
        TOKENIZERS["translate"] = original_tokenizer

    assert calls == ["translate"]
