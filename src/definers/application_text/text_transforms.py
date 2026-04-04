from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from definers.constants import MAX_INPUT_LENGTH
from definers.regex_utils import sub as regex_sub


def detect_language(text: str) -> str:
    try:
        from langdetect import detect

        return detect(text)
    except ModuleNotFoundError:
        if any("\u0590" <= ch <= "\u05ff" for ch in text):
            return "he"
        if any("\u0400" <= ch <= "\u04ff" for ch in text):
            return "ru"
        return "en"


def language(text: str) -> str:
    return detect_language(text).lower()


def strip_nikud(text: str | None) -> str:
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def simple_text(prompt: str | None) -> str:
    if prompt is None:
        return ""
    normalized_prompt = str(prompt)
    if len(normalized_prompt) > MAX_INPUT_LENGTH:
        raise ValueError(
            f"input too long ({len(normalized_prompt)} > {MAX_INPUT_LENGTH})"
        )
    cleaned_lines = []
    for line in normalized_prompt.splitlines():
        cleaned_line = " ".join(line.split())
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    normalized_prompt = "\n".join(cleaned_lines)
    for pattern in [" .", ". ", ".."]:
        while pattern in normalized_prompt:
            normalized_prompt = normalized_prompt.replace(pattern, ".")
    while "--" in normalized_prompt:
        normalized_prompt = normalized_prompt.replace("--", "-")
    normalized_prompt = normalized_prompt.replace("|", " or ")
    normalized_prompt = normalized_prompt.replace("?", " I wonder ")
    normalized_prompt = regex_sub(
        r"(?<=[A-Za-z0-9])\/(?=[A-Za-z0-9])",
        " ",
        normalized_prompt,
    )
    punctuation_characters = "\"'!#$%&()*+,/:;<=>?@[\\]^_`{|}~"
    normalized_prompt = normalized_prompt.translate(
        str.maketrans("", "", punctuation_characters)
    )
    normalized_prompt = normalized_prompt.strip().strip(".")
    normalized_prompt = normalized_prompt.replace(".", " and ")
    lines = [
        line.lower().strip().replace(" -", "-").replace("- ", "-")
        for line in normalized_prompt.splitlines()
    ]
    return "\n".join([" ".join(line.split()) for line in lines if line.strip()])


@lru_cache(maxsize=1024)
def camel_case(txt: str | None) -> str:
    if not txt:
        return ""
    words = re.sub("[^a-zA-Z0-9]+", " ", txt).split()
    if not words:
        return ""
    return words[0].lower() + "".join(word.capitalize() for word in words[1:])
