class TextValidationError(RuntimeError):
    pass


class TextInputValidator:
    def __init__(
        self,
        *,
        normalize_text,
        log,
        max_input_length,
        max_consecutive_spaces,
        error_factory,
    ):
        self._normalize_text = normalize_text
        self._log = log
        self._max_input_length = max_input_length
        self._max_consecutive_spaces = max_consecutive_spaces
        self._error_factory = error_factory

    @classmethod
    def default(cls):
        import definers.text as text
        from definers.constants import MAX_CONSECUTIVE_SPACES, MAX_INPUT_LENGTH
        from definers.system import log

        try:
            import gradio as gr

            error_factory = gr.Error
        except Exception:
            error_factory = TextValidationError

        return cls(
            normalize_text=text.simple_text,
            log=log,
            max_input_length=MAX_INPUT_LENGTH,
            max_consecutive_spaces=MAX_CONSECUTIVE_SPACES,
            error_factory=error_factory,
        )

    def validate(self, value):
        if value is None:
            return ""
        normalized_value = str(value)
        if len(normalized_value) > self._max_input_length:
            self._log(
                "Validation reject",
                f"input length {len(normalized_value)} exceeds {self._max_input_length}",
            )
            raise self._error_factory(
                f"Input too long ({len(normalized_value)} > {self._max_input_length})"
            )
        if " " * (self._max_consecutive_spaces + 1) in normalized_value:
            self._log(
                "Validation reject", "input has excessive consecutive spaces"
            )
            raise self._error_factory(
                "Input contains too many consecutive spaces"
            )
        return self._normalize_text(normalized_value)
