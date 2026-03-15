from typing import Any

import definers.text as text
from definers.application_chat.contracts import (
    ChatHistory,
    ChatMessage,
    ChatRequest,
)
from definers.application_chat.handlers import (
    build_chat_audit_message as _build_chat_audit_message,
    create_chat_request,
    create_chat_request_handler,
    handle_chat_request,
)
from definers.constants import (
    MAX_CONSECUTIVE_SPACES,
    MAX_INPUT_LENGTH,
    language_codes,
)
from definers.ml import answer
from definers.system import log


def validate_text_input(value):
    import gradio as gr

    if value is None:
        return ""
    if len(value) > MAX_INPUT_LENGTH:
        log(
            "Validation reject",
            f"input length {len(value)} exceeds {MAX_INPUT_LENGTH}",
        )
        raise gr.Error(f"Input too long ({len(value)} > {MAX_INPUT_LENGTH})")
    if " " * (MAX_CONSECUTIVE_SPACES + 1) in value:
        log("Validation reject", "input has excessive consecutive spaces")
        raise gr.Error("Input contains too many consecutive spaces")
    return text.simple_text(value)


def build_chat_audit_message(
    included_types: list[str], original_language: str | None
) -> str:
    return _build_chat_audit_message(
        included_types,
        original_language,
        language_codes,
    )


def create_default_chat_request_handler():
    return create_chat_request_handler(
        detect_language=text.language,
        translate_text=text.ai_translate,
        validate_text=validate_text_input,
        answer=answer,
        log=log,
        language_names=language_codes,
    )


def to_chat_request(
    message: ChatMessage,
    history: ChatHistory,
) -> ChatRequest:
    return create_chat_request(message, history)


def get_chat_response(
    message: ChatMessage,
    history: ChatHistory,
) -> Any:
    return handle_chat_request(
        message,
        history,
        detect_language=text.language,
        translate_text=text.ai_translate,
        validate_text=validate_text_input,
        answer=answer,
        log=log,
        language_names=language_codes,
    )
