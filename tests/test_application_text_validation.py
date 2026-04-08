import builtins
import sys

from definers.text.validation import (
    TextInputValidator,
    TextValidationError,
)


class ValidationError(Exception):
    pass


def test_text_input_validator_accepts_none_as_empty_string():
    validator = TextInputValidator(
        normalize_text=lambda value: value,
        log=lambda *_args: None,
        max_input_length=10,
        max_consecutive_spaces=2,
        error_factory=ValidationError,
    )

    assert validator.validate(None) == ""


def test_text_input_validator_rejects_long_values_and_logs_reason():
    messages: list[tuple[str, str]] = []
    validator = TextInputValidator(
        normalize_text=lambda value: value,
        log=lambda subject, message: messages.append((subject, message)),
        max_input_length=3,
        max_consecutive_spaces=2,
        error_factory=ValidationError,
    )

    try:
        validator.validate("abcd")
    except ValidationError as exc:
        assert str(exc) == "Input too long (4 > 3)"
    else:
        raise AssertionError("expected ValidationError")

    assert messages == [("Validation reject", "input length 4 exceeds 3")]


def test_text_input_validator_rejects_excessive_consecutive_spaces():
    messages: list[tuple[str, str]] = []
    validator = TextInputValidator(
        normalize_text=lambda value: value,
        log=lambda subject, message: messages.append((subject, message)),
        max_input_length=20,
        max_consecutive_spaces=2,
        error_factory=ValidationError,
    )

    try:
        validator.validate("a   b")
    except ValidationError as exc:
        assert str(exc) == "Input contains too many consecutive spaces"
    else:
        raise AssertionError("expected ValidationError")

    assert messages == [
        ("Validation reject", "input has excessive consecutive spaces")
    ]


def test_text_input_validator_default_falls_back_without_gradio(monkeypatch):
    monkeypatch.delitem(sys.modules, "gradio", raising=False)

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "gradio":
            raise ModuleNotFoundError("missing gradio", name="gradio")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    validator = TextInputValidator.default()

    assert validator._error_factory is TextValidationError
