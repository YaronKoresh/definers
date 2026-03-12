from __future__ import annotations

import re
from re import Pattern

MAX_PATTERN_LENGTH = 1000


_NESTED_QUANTIFIER_RE = re.compile(r"\([^)]*[+*][^)]*\)[^)]*[+*]")


def escape(text: str) -> str:
    return re.escape(text)


def _check_complexity(pattern: str) -> None:
    if len(pattern) > MAX_PATTERN_LENGTH:
        raise ValueError(f"regex pattern too long ({len(pattern)} characters)")

    if _NESTED_QUANTIFIER_RE.search(pattern):
        raise ValueError("regex pattern contains nested quantifiers")


def compile(pattern: str, flags: int = 0) -> Pattern:
    _check_complexity(pattern)
    return re.compile(pattern, flags)


def fullmatch(pattern: str, string: str, flags: int = 0) -> bool:
    regex = compile(pattern, flags)
    return bool(regex.fullmatch(string))


def sub(pattern: str, repl: str, string: str, flags: int = 0) -> str:
    regex = compile(pattern, flags)
    return regex.sub(repl, string)


def escape_and_compile(
    template: str, user_input: str, flags: int = 0
) -> Pattern:
    safe_input = escape(user_input)
    pattern = template.format(safe_input)
    return compile(pattern, flags)
