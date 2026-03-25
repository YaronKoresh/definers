from types import SimpleNamespace

from definers.application_ml.answer_content_path_resolver import (
    AnswerContentPathResolver,
)
from definers.application_ml.answer_text_service import AnswerTextService


def test_answer_content_paths_ignores_entries_without_paths():
    content = (
        {"path": "one.txt"},
        {"missing": "value"},
        "skip",
        {"path": "two.txt"},
    )

    assert AnswerContentPathResolver.content_paths(content) == [
        "one.txt",
        "two.txt",
    ]


def test_append_history_message_merges_adjacent_roles():
    history = [{"role": "system", "content": "base"}]

    AnswerTextService.append_history_message(history, "user", "alpha")
    AnswerTextService.append_history_message(history, "user", "beta")

    assert history == [
        {"role": "system", "content": "base"},
        {"role": "user", "content": "alpha\n\nbeta"},
    ]


def test_answer_content_paths_returns_dict_path():
    assert AnswerContentPathResolver.content_paths({"path": "asset.wav"}) == [
        "asset.wav"
    ]


def test_normalize_answer_text_skips_language_stack_for_ascii_english():
    assert AnswerTextService.normalize_answer_text("plain english", "en") == (
        "plain english"
    )


def test_append_history_message_appends_new_role():
    history = [{"role": "system", "content": "base"}]

    AnswerTextService.append_history_message(history, "assistant", "reply")

    assert history == [
        {"role": "system", "content": "base"},
        {"role": "assistant", "content": "reply"},
    ]