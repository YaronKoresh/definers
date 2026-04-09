from collections.abc import Mapping, Sequence


def build_chat_audit_message(
    included_types: Sequence[str],
    original_language: str | None,
    language_names: Mapping[str, str],
) -> str:
    line_break = "\n"
    included = line_break.join(included_types) or "none"
    if original_language is None:
        return (
            f"Got a new message.{line_break}{line_break}"
            f"The message including the following types of data:{line_break}{included}"
        )
    language_name = language_names.get(original_language, original_language)
    return (
        f"Got a new message in {language_name}.{line_break}{line_break}"
        f"The message including the following types of data:{line_break}{included}"
    )
