from definers.application_chat.handlers import (
    MEDIA_INSTRUCTION,
    build_chat_audit_message,
    create_chat_request,
    create_chat_request_handler,
    handle_chat_request,
    normalize_chat_request,
)


def test_normalize_chat_request_translates_and_appends_media_marker():
    context = normalize_chat_request(
        {"text": "shalom", "files": ["clip.wav"]},
        [{"role": "system", "content": "base"}],
        detect_language=lambda value: "he",
        translate_text=lambda value: f"translated:{value}",
        validate_text=lambda value: value.upper(),
    )

    assert context.original_language == "he"
    assert context.metadata.includes_media is True
    assert context.metadata.includes_text is True
    assert context.included_types == ("files", "text")
    assert context.base_history == [{"role": "system", "content": "base"}]
    assert context.media[0].path == "clip.wav"
    assert context.text is not None
    assert context.text.raw == "shalom"
    assert context.text.translated == "translated:shalom"
    assert context.text.validated == "TRANSLATED:SHALOM"
    assert context.history == [
        {"role": "system", "content": "base"},
        {"role": "user", "content": {"path": "clip.wav"}},
        {"role": "user", "content": "TRANSLATED:SHALOM"},
        {"role": "user", "content": MEDIA_INSTRUCTION},
    ]


def test_normalize_chat_request_keeps_history_copy_and_skips_translation_for_english():
    history = [{"role": "system", "content": "seed"}]

    context = normalize_chat_request(
        {"text": "hello", "files": []},
        history,
        detect_language=lambda value: "en",
        translate_text=lambda value: f"translated:{value}",
        validate_text=lambda value: value.strip(),
    )

    history.append({"role": "system", "content": "mutated"})

    assert context.base_history == [{"role": "system", "content": "seed"}]
    assert context.media == ()
    assert context.text is not None
    assert context.text.translated == "hello"
    assert context.history == [
        {"role": "system", "content": "seed"},
        {"role": "user", "content": "hello"},
    ]


def test_normalize_chat_request_snapshots_message_and_ignores_blank_text():
    message = {"text": "   ", "files": [" clip.wav "]}
    history = [{"role": "system", "content": "seed"}]

    context = normalize_chat_request(
        message,
        history,
        detect_language=lambda value: "en",
        translate_text=lambda value: value,
        validate_text=lambda value: value,
    )

    message["files"].append("later.wav")
    history.append({"role": "system", "content": "mutated"})

    assert context.metadata.includes_media is True
    assert context.metadata.includes_text is False
    assert len(context.media) == 1
    assert context.media[0].path == "clip.wav"
    assert context.text is None
    assert context.base_history == [{"role": "system", "content": "seed"}]
    assert context.history == [
        {"role": "system", "content": "seed"},
        {"role": "user", "content": {"path": "clip.wav"}},
    ]


def test_build_chat_audit_message_uses_language_mapping():
    message = build_chat_audit_message(
        ["files", "text"],
        "en",
        {"en": "English"},
    )

    assert "English" in message
    assert "files\ntext" in message


def test_create_chat_request_handler_logs_and_answers_with_context():
    messages: list[tuple[str, object]] = []

    handler = create_chat_request_handler(
        detect_language=lambda value: "en",
        translate_text=lambda value: value,
        validate_text=lambda value: value.strip(),
        answer=lambda history: history[-1]["content"],
        log=lambda subject, data: messages.append((subject, data)),
        language_names={"en": "English"},
    )

    response = handler.handle({"text": "hello"}, [])

    assert response == "hello"
    assert messages[0][0] == "Chat"
    assert "English" in str(messages[0][1])
    assert messages[1] == ("Chatbot response", "hello")


def test_build_chat_audit_message_reports_empty_payload_types():
    message = build_chat_audit_message([], None, {})

    assert message.endswith("none")


def test_create_chat_request_copies_existing_request_payload():
    source_message = {"text": "hello"}
    source_history = [{"role": "system", "content": "seed"}]
    request = create_chat_request(source_message, source_history)

    source_message["text"] = "mutated"
    source_history.append({"role": "user", "content": "later"})

    assert request.message == {"text": "hello"}
    assert list(request.history) == [{"role": "system", "content": "seed"}]


def test_handle_chat_request_prefers_request_snapshot_history():
    request = create_chat_request(
        {"text": "hello"},
        [{"role": "system", "content": "seed"}],
    )

    response = handle_chat_request(
        request,
        [{"role": "system", "content": "ignored"}],
        detect_language=lambda value: "en",
        translate_text=lambda value: value,
        validate_text=lambda value: value,
        answer=lambda history: history[0]["content"],
        log=lambda subject, data: None,
        language_names={"en": "English"},
    )

    assert response == "seed"
