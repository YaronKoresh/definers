from __future__ import annotations

import hashlib
import os
import random
import string
import sys

from definers.database import Database
from definers.system import read

from .system_messages import set_system_message
from .text_transforms import camel_case, language, simple_text, strip_nikud
from .translation import (
    ai_translate,
    duck_translate,
    google_translate,
    translate_with_code_using,
)


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


def translate_with_code(text_to_translate: str, lang: str) -> str:
    return translate_with_code_using(text_to_translate, lang, ai_translate)


__all__ = [glb for glb in globals() if not glb.startswith("_")]
