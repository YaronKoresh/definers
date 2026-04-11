from __future__ import annotations

import hashlib
import importlib
import os
import random
import string
import sys

__all__ = [
    "Database",
    "ai_translate",
    "camel_case",
    "duck_translate",
    "file_to_sha3_512",
    "google_translate",
    "language",
    "number_to_hex",
    "random_number",
    "random_salt",
    "random_string",
    "set_system_message",
    "simple_text",
    "string_to_bytes",
    "string_to_sha3_512",
    "strip_nikud",
    "translate_with_code",
]

_TEXT_EXPORTS = {
    "Database": "definers.database",
    "camel_case": "definers.text.text_transforms",
    "duck_translate": "definers.text.translation",
    "google_translate": "definers.text.translation",
    "language": "definers.text.text_transforms",
    "read": "definers.system",
    "set_system_message": "definers.text.system_messages",
    "simple_text": "definers.text.text_transforms",
    "strip_nikud": "definers.text.text_transforms",
    "translate_with_code_using": "definers.text.translation",
}


def _load_text_export(name: str):
    module_name = _TEXT_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def random_string(min_len: int = 50, max_len: int = 60) -> str:
    characters = string.ascii_letters + string.digits + "_"
    length = random.randint(min_len, max_len)
    return "".join(random.choice(characters) for _ in range(length))


def random_salt(size: int) -> int:
    return int.from_bytes(os.urandom(size), sys.byteorder)


def random_number(min: int = 0, max: int = 100) -> int:
    return int.from_bytes(os.urandom(4), sys.byteorder) % (max - min + 1) + min


def number_to_hex(num: int | str) -> str:
    return hex(int(num))


def string_to_bytes(value: object) -> bytes:
    return bytes(f"{value}", encoding="utf-8")


def file_to_sha3_512(path: str, salt_num: int | None = None) -> str | None:
    reader = globals().get("read")
    if reader is None:
        reader = _load_text_export("read")
    content = reader(path)
    if content is not None:
        return string_to_sha3_512(content, salt_num)
    return None


def string_to_sha3_512(
    value: str | bytes,
    salt_num: int | None = None,
) -> str:
    digest = hashlib.sha3_512()
    if isinstance(value, bytes):
        digest.update(value)
    else:
        digest.update(value.encode("utf-8"))
    if salt_num is not None:
        salt = number_to_hex(salt_num)
        digest.update(salt.encode("utf-8"))
    return digest.hexdigest()


def ai_translate(text: str, lang: str = "en") -> str:
    translation = importlib.import_module("definers.text.translation")
    return translation.ai_translate(text, lang=lang)


def translate_with_code(text_to_translate: str, lang: str) -> str:
    translator = globals().get("translate_with_code_using")
    if translator is None:
        translator = _load_text_export("translate_with_code_using")
    return translator(text_to_translate, lang, ai_translate)


def __getattr__(name: str):
    return _load_text_export(name)


def __dir__() -> list[str]:
    return sorted(set(globals()).union(_TEXT_EXPORTS).union(__all__))
