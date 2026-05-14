def create_default_chat_request_handler():
    import definers.text as text
    from definers.chat.handlers import create_chat_request_handler
    from definers.constants import language_codes
    from definers.system import log
    from definers.text.validation import TextInputValidator

    validator = TextInputValidator.default()
    current_answer = answer
    if current_answer is None:
        from definers.ml import answer as current_answer

    return create_chat_request_handler(
        detect_language=text.language,
        translate_text=text.ai_translate,
        validate_text=validator.validate,
        answer=current_answer,
        log=log,
        language_names=language_codes,
    )


def get_chat_response(message, history):
    return create_default_chat_request_handler().handle(message, history)


def get_chat_response_stream(message, history):
    from definers.chat.handlers import (
        coerce_chat_request,
        log_chat_request,
        log_chat_response,
    )

    request_handler = create_default_chat_request_handler()
    yield "Validating chat request..."
    normalized_request = coerce_chat_request(message, history)
    yield "Normalizing chat context..."
    context = request_handler.normalize_request(normalized_request)
    yield "Logging request context..."
    log_chat_request(
        request_handler.log,
        request_handler.build_audit_message,
        context,
    )
    yield "Running answer runtime..."
    response = request_handler.answer(context.history)
    yield "Finalizing response..."
    log_chat_response(request_handler.log, response)
    yield response


answer = None
