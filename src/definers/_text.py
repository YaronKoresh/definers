import hashlib
import os
import random
import re
import shutil
import string
import sys
import unicodedata
from datetime import datetime
from functools import lru_cache
from time import time

from definers._constants import (
    MODELS,
    SYSTEM_MESSAGE,
    TOKENIZERS,
    beam_kwargs,
    higher_beams,
    language_codes,
    unesco_mapping,
)
from definers._cuda import device
from definers._system import read
from definers._web import extract_text


def set_system_message(
    name: str = None,
    role: str = "a helpful AI assistant",
    tone: str = None,
    chattiness: str = None,
    interaction_style: str = None,
    persona_data: dict = None,
    goals: list = None,
    task_rules: list = None,
    output_format: str = None,
    rules: list = None,
    data: list = None,
    verbose: bool = False,
    friendly: bool = True,
    formal: bool = None,
    creative: bool = None,
):
    global SYSTEM_MESSAGE
    _ = (rules, verbose, friendly, formal, creative)
    parts = []
    parts.append(f"You are {role}.")
    if name:
        parts.append(f"Your name is {name}.")
    if tone:
        parts.append(f"Your tone should be {tone}.")
    if chattiness:
        parts.append(f"In terms of verbosity, {chattiness}.")
    if interaction_style:
        parts.append(f"When interacting, {interaction_style}.")
    if persona_data:
        for key, value in persona_data.items():
            parts.append(f"{key} is {value}")
    if goals:
        parts.append("; ".join(goals) + ".")
    if task_rules or output_format:
        parts.append("You must strictly follow these rules:")
        rule_num = 1
        if task_rules:
            for rule in task_rules:
                parts.append(f"{rule_num}. {rule}")
                rule_num += 1
        if output_format:
            parts.append(
                f"{rule_num}. Your final output must be exclusively in the following format: {output_format}."
            )
    SYSTEM_MESSAGE = "\n".join(parts)
    import definers as _d

    _d.SYSTEM_MESSAGE = SYSTEM_MESSAGE


def language(text):
    from langdetect import detect

    return detect(text).lower()


def strip_nikud(text: str) -> str:
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def simple_text(prompt: str) -> str:

    from definers import regex_utils
    from definers._constants import MAX_CONSECUTIVE_SPACES, MAX_INPUT_LENGTH

    if prompt is None:
        return ""
    prompt = str(prompt)

    if len(prompt) > MAX_INPUT_LENGTH:
        raise ValueError(f"input too long ({len(prompt)} > {MAX_INPUT_LENGTH})")
    if " " * (MAX_CONSECUTIVE_SPACES + 1) in prompt:
        raise ValueError("too many consecutive spaces")

    prompt = prompt.replace("	", " ")

    lines = prompt.splitlines()
    collapsed = []
    for line in lines:
        if collapsed and not line and not collapsed[-1]:
            continue
        collapsed.append(line)
    prompt = "\n".join(collapsed)

    while "  " in prompt:
        prompt = prompt.replace("  ", " ")
    for pat in [" .", ". ", ".."]:
        while pat in prompt:
            prompt = prompt.replace(pat, ".")
    while "--" in prompt:
        prompt = prompt.replace("--", "-")

    prompt = prompt.replace("|", " or ")
    prompt = regex_utils.sub(r"\s*\?\s*", " I wonder ", prompt)
    prompt = regex_utils.sub(r"(?<=[A-Za-z0-9])\/(?=[A-Za-z0-9])", " ", prompt)

    punc_chars = "\"'!#$%&()*+,/:;<=>?@[\\]^_`{|}~"
    prompt = prompt.translate(str.maketrans("", "", punc_chars))
    prompt = prompt.strip().strip(".")

    prompt = regex_utils.sub(r"\s*\.\s*", " and ", prompt)
    while "  " in prompt:
        prompt = prompt.replace("  ", " ")
    prompt = regex_utils.sub(r"(\n){2,}", "\n", prompt)

    lines = prompt.split("\n")
    lines = [
        line.lower().strip().replace(" -", "-").replace("- ", "-")
        for line in lines
    ]
    lines = [line for line in lines if line]
    return "\n".join(lines)


@lru_cache(maxsize=1024)
def camel_case(txt: str) -> str:
    if not txt:
        return ""
    words = re.sub("[^a-zA-Z0-9]+", " ", txt).split()
    if not words:
        return ""
    return words[0].lower() + "".join(word.capitalize() for word in words[1:])


def ai_translate(text, lang="en"):
    import torch
    from sacremoses import MosesPunctNormalizer

    try:
        from stopes.pipelines.monolingual.utils.sentence_split import (
            get_split_algo,
        )
    except ImportError:

        def get_split_algo(*_args, **_kwargs):
            return lambda s: [s]

    if not text or not text.strip():
        return ""
    text = strip_nikud(text)
    long_paragraph_threshold = 800
    tgt_code = unesco_mapping[lang]
    if isinstance(tgt_code, list):
        tgt_code = tgt_code[0]
    model = MODELS["translate"]
    tokenizer = TOKENIZERS["translate"]
    tokenizer.tgt_lang = tgt_code
    paragraphs = text.split("\n")
    translated_paragraphs = []
    for paragraph in paragraphs:
        if not paragraph.strip():
            translated_paragraphs.append("")
            continue
        try:
            source_lang_code = language(paragraph)
            src_code = unesco_mapping[source_lang_code]
            if isinstance(src_code, list):
                src_code = src_code[0]
        except (KeyError, Exception) as e:
            from definers._system import catch

            catch(e)
            translated_paragraphs.append(paragraph)
            continue
        if src_code == tgt_code:
            translated_paragraphs.append(paragraph)
            continue
        punct_normalizer = MosesPunctNormalizer(lang=source_lang_code)
        paragraph = punct_normalizer.normalize(paragraph)
        splitter = get_split_algo(src_code[:3], "default")
        tokenizer.src_lang = src_code
        _beam_kwargs = beam_kwargs
        _beam_kwargs["num_beams"] = higher_beams
        if len(paragraph) < long_paragraph_threshold:
            try:
                inputs = tokenizer(
                    paragraph,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                input_ids = inputs.input_ids.to(device())
                forced_token_id = tokenizer.convert_tokens_to_ids(tgt_code)
                translated_ids = model.generate(
                    input_ids=input_ids,
                    forced_bos_token_id=forced_token_id,
                    renormalize_logits=True,
                    max_length=512,
                    **_beam_kwargs,
                )
                translated_paragraph = tokenizer.decode(
                    translated_ids[0], skip_special_tokens=True
                )
                translated_paragraphs.append(translated_paragraph)
            except Exception as e:
                from definers._system import catch

                catch(e)
                translated_paragraphs.append(
                    f"[Translation Error: {paragraph[:30]}...]"
                )
        else:
            try:
                sentences = list(splitter(paragraph))
                translated_sentences = []
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    inputs = tokenizer(
                        sentence,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                    input_ids = inputs.input_ids.to(device())
                    forced_token_id = tokenizer.convert_tokens_to_ids(tgt_code)
                    translated_ids = model.generate(
                        input_ids=input_ids,
                        forced_bos_token_id=forced_token_id,
                        renormalize_logits=True,
                        max_length=512,
                        **_beam_kwargs,
                    )
                    translated_chunk = tokenizer.decode(
                        translated_ids[0], skip_special_tokens=True
                    )
                    translated_sentences.append(translated_chunk)
                translated_paragraph = " ".join(translated_sentences)
                translated_paragraphs.append(translated_paragraph)
            except Exception as e:
                from definers._system import catch

                catch(e)
                translated_paragraphs.append(
                    f"[Translation Error: {paragraph[:30]}...]"
                )
    return "\n".join(translated_paragraphs)


def google_translate(text, lang="en"):
    import requests

    if text is None or lang is None:
        return ""
    if text.strip() == "":
        return ""
    lang = simple_text(lang)
    text = simple_text(text)
    url = f"https://translate.googleapis.com/translate_a/single?client=gtx&dt=t&q={text}&sl={language(text)}&tl={lang}"
    try:
        r = requests.get(url)
        ret = r.text.split('"')[1]
        ret = simple_text(ret)
        from definers import logger

        logger.info(ret)
        return ret
    except Exception as e:
        from definers._system import catch

        catch(e)
        return ""


def duck_translate(text, lang="en"):
    if text is None or lang is None:
        return ""
    if text.strip() == "":
        return ""
    lang = simple_text(lang)
    lang = language_codes[lang]
    text = simple_text(text)
    url = f"https://duckduckgo.com/?q={lang} translate: {text}&ia=web"
    scraped = extract_text(
        url,
        ".module--translations-translatedtext.js-module--translations-translatedtext",
    )
    if scraped is None or scraped == "":
        print("Translation Warning: Failed To Translate!")
    else:
        text = scraped
    text = simple_text(text)
    print(text)
    return text


def random_string(min_len=50, max_len=60):
    characters = string.ascii_letters + string.digits + "_"
    length = random.randint(min_len, max_len)
    return "".join(random.choice(characters) for _ in range(length))


def random_salt(size):
    return int.from_bytes(os.urandom(size), sys.byteorder)


def random_number(min=0, max=100):
    return int.from_bytes(os.urandom(4), sys.byteorder) % (max - min + 1) + min


def number_to_hex(num):
    return hex(int(num))


def string_to_bytes(str):
    return bytes(f"{str}", encoding="utf-8")


def file_to_sha3_512(path, salt_num=None):
    content = read(path)
    if content is not None:
        return string_to_sha3_512(content, salt_num)


def string_to_sha3_512(str, salt_num=None):
    m = hashlib.sha3_512()
    if isinstance(str, bytes):
        m.update(str)
    else:
        m.update(str.encode("utf-8"))
    if salt_num is not None:
        salt = number_to_hex(salt_num)
        m.update(salt.encode("utf-8"))
    return m.hexdigest()


class Database:
    def __init__(self, path):
        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def _get_history(self, db, filters={}, days=None):
        db_path = os.path.join(self.path, db)
        if not os.path.exists(db_path):
            return []
        start_timestamp = 0
        if days is not None and isinstance(days, (int, float)):
            start_timestamp = time() - days * 86400
        try:
            timestamp_dirs = [
                d for d in os.listdir(db_path) if int(d) >= start_timestamp
            ]
        except (ValueError, FileNotFoundError):
            return []
        results = []
        for ts_string in timestamp_dirs:
            record_path = os.path.join(db_path, ts_string)
            if not os.path.isdir(record_path):
                continue
            item_data = {}
            for key_file in os.listdir(record_path):
                with open(os.path.join(record_path, key_file)) as f:
                    item_data[key_file] = f.read()
            all_filters_match = True
            for key, value in filters.items():
                if item_data.get(key) != str(value):
                    all_filters_match = False
                    break
            if all_filters_match:
                ts_int = int(ts_string)
                record = {
                    "timestamp": ts_int,
                    "time": datetime.fromtimestamp(ts_int),
                    "data": item_data,
                }
                results.append(record)
        return sorted(results, key=lambda x: x["timestamp"], reverse=True)

    def history(self, db, filters={}, days=None):
        full_history = self._get_history(db, filters, days)
        return [item["data"] for item in full_history]

    def push(self, db, data, timestamp=None):
        if timestamp is None:
            timestamp = int(time())
        elif not isinstance(timestamp, int):
            try:
                timestamp = int(timestamp)
            except (ValueError, TypeError):
                timestamp = int(time())
        record_path = os.path.join(self.path, db, str(timestamp))
        os.makedirs(record_path, exist_ok=True)
        for key, value in data.items():
            file_path = os.path.join(record_path, key)
            with open(file_path, "w") as f:
                f.write(str(value))

    def latest(self, db="*", filters={}, days=None, identifierKey="id"):
        if db == "*":
            return {
                db_name: self.latest(db_name, filters, days, identifierKey)
                for db_name in os.listdir(self.path)
            }
        if isinstance(db, list):
            return {
                db_name: self.latest(db_name, filters, days, identifierKey)
                for db_name in db
            }
        full_history = self._get_history(db)
        latest_items = {}
        for item in full_history:
            item_id = item["data"].get(identifierKey)
            if item_id is None:
                continue
            if (
                item_id not in latest_items
                or item["timestamp"] > latest_items[item_id]["timestamp"]
            ):
                latest_items[item_id] = item
        filtered_results = list(latest_items.values())
        if days is not None:
            start_timestamp = time() - days * 86400
            filtered_results = [
                item
                for item in filtered_results
                if item["timestamp"] >= start_timestamp
            ]
        if filters:
            final_results = []
            for item in filtered_results:
                all_filters_match = True
                for key, value in filters.items():
                    if item["data"].get(key) != str(value):
                        all_filters_match = False
                        break
                if all_filters_match:
                    final_results.append(item)
            filtered_results = final_results
        sorted_results = sorted(
            filtered_results, key=lambda x: x["timestamp"], reverse=True
        )
        return [item["data"] for item in sorted_results]

    def clean(self, db="*", identifierKey="id"):
        if db == "*":
            dbs = os.listdir(self.path)
            for db_name in dbs:
                self.clean(db_name, identifierKey)
            return
        if isinstance(db, list):
            for db_name in db:
                self.clean(db_name, identifierKey)
            return
        full_history = self._get_history(db)
        latest_items = {}
        for item in full_history:
            item_id = item["data"].get(identifierKey)
            if item_id is None:
                continue
            if (
                item_id not in latest_items
                or item["timestamp"] > latest_items[item_id]["timestamp"]
            ):
                latest_items[item_id] = item
        records_to_keep = list(latest_items.values())
        db_path = os.path.join(self.path, db)
        if os.path.isdir(db_path):
            shutil.rmtree(db_path)
        for item in records_to_keep:
            self.push(db, item["data"], item["timestamp"])


def translate_with_code(text_to_translate, lang):
    if not text_to_translate or not text_to_translate.strip():
        return text_to_translate
    code_block_pattern = "(```.*?```)"
    parts = re.split(code_block_pattern, text_to_translate, flags=re.DOTALL)
    processed_parts = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                translated_text = ai_translate(part, lang=lang)
                processed_parts.append(translated_text)
            else:
                processed_parts.append(part)
        else:
            processed_parts.append(part)
    return "".join(processed_parts)
