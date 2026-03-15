import hashlib
import importlib
import os
import random
import string
import sys

from definers.persistence.database import Database
from definers.system import read

application_text_language = importlib.import_module(
    "definers.application_text.language"
)

camel_case = application_text_language.camel_case
duck_translate = application_text_language.duck_translate
google_translate = application_text_language.google_translate
language = application_text_language.language
set_system_message = application_text_language.set_system_message
simple_text = application_text_language.simple_text
strip_nikud = application_text_language.strip_nikud

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
    content = read(path)
    if content is not None:
        return string_to_sha3_512(content, salt_num)


def string_to_sha3_512(value: str | bytes, salt_num: int | None = None) -> str:
    m = hashlib.sha3_512()
    if isinstance(value, bytes):
        m.update(value)
    else:
        m.update(value.encode("utf-8"))
    if salt_num is not None:
        salt = number_to_hex(salt_num)
        m.update(salt.encode("utf-8"))
    return m.hexdigest()


def ai_translate(text: str, lang: str = "en") -> str:
    return application_text_language.ai_translate(text, lang=lang)


def translate_with_code(text_to_translate: str, lang: str) -> str:
    return application_text_language.translate_with_code_using(
        text_to_translate,
        lang,
        ai_translate,
    )
