from definers.application_chat.handlers import (
    MEDIA_INSTRUCTION,
    build_chat_audit_message,
    create_chat_request_handler,
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
