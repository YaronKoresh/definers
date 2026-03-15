import re
import sys
import unicodedata
from collections.abc import Callable
from functools import lru_cache

from definers import constants
from definers.constants import (
    MAX_INPUT_LENGTH,
    MODELS,
    TOKENIZERS,
    beam_kwargs,
    higher_beams,
    language_codes,
    unesco_mapping,
)
from definers.cuda import device
from definers.regex_utils import sub as regex_sub
from definers.web import extract_text


def _update_system_message(system_message: str) -> None:
    constants.SYSTEM_MESSAGE = system_message
    definers_package = sys.modules.get("definers")
    if definers_package is not None:
        definers_package.SYSTEM_MESSAGE = system_message


def _build_system_message(
    name: str | None,
    role: str,
    tone: str | None,
    chattiness: str | None,
    interaction_style: str | None,
    persona_data: dict | None,
    goals: list | None,
    task_rules: list | None,
    output_format: str | None,
    rules: list | None,
    data: list | None,
    verbose: bool,
    friendly: bool,
    formal: bool | None,
    creative: bool | None,
) -> str:
    _ = (rules, verbose, friendly, formal, creative, data)
    parts = [f"You are {role}."]
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
    return "\n".join(parts)


def _translate_text_segment(
    model: object,
    tokenizer: object,
    text: str,
    target_code: str,
    generation_kwargs: dict,
) -> str:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = inputs.input_ids.to(device())
    forced_token_id = tokenizer.convert_tokens_to_ids(target_code)
    translated_ids = model.generate(
        input_ids=input_ids,
        forced_bos_token_id=forced_token_id,
        renormalize_logits=True,
        max_length=512,
        **generation_kwargs,
    )
    return tokenizer.decode(translated_ids[0], skip_special_tokens=True)


def _resolve_target_code(lang: str) -> str:
    return _resolve_primary_unesco_code(unesco_mapping[lang])


def _resolve_source_translation_context(paragraph: str) -> tuple[str, str]:
    detected_language = language(paragraph)
    return detected_language, _resolve_primary_unesco_code(
        unesco_mapping[detected_language]
    )


def _resolve_primary_unesco_code(
    code: str | list[str] | tuple[str, ...],
) -> str:
    if isinstance(code, (list, tuple)):
        return code[0]
    return code


def _resolve_target_language(lang: str) -> tuple[str, str]:
    normalized_lang = simple_text(lang)
    if normalized_lang in language_codes:
        return normalized_lang, language_codes[normalized_lang]
    for code, language_name in language_codes.items():
        if language_name == normalized_lang:
            return code, language_name
    return normalized_lang, normalized_lang


def _get_sentence_splitter(source_code: str):
    try:
        import importlib

        get_split_algo = importlib.import_module(
            "stopes.pipelines.monolingual.utils.sentence_split"
        ).get_split_algo
    except ImportError:

        def get_split_algo(*_args, **_kwargs):
            return lambda value: [value]

    return get_split_algo(source_code[:3], "default")


def _translation_error(paragraph: str) -> str:
    return f"[Translation Error: {paragraph[:30]}...]"


def _get_logger():
    active_logger = sys.modules.get("definers.logger")
    if active_logger is not None:
        return active_logger
    from definers import logger

    return logger


def _matches_empty_text(value: str | None) -> bool:
    return value is None or value.strip() == ""


def translate_with_code_using(
    text_to_translate: str,
    lang: str,
    translate_text: Callable[[str, str], str],
) -> str:
    if not text_to_translate or not text_to_translate.strip():
        return text_to_translate
    code_block_pattern = "(```.*?```)"
    parts = re.split(code_block_pattern, text_to_translate, flags=re.DOTALL)
    processed_parts = []
    for index, part in enumerate(parts):
        if index % 2 == 0:
            if part.strip():
                processed_parts.append(translate_text(part, lang=lang))
            else:
                processed_parts.append(part)
            continue
        processed_parts.append(part)
    return "".join(processed_parts)


def set_system_message(
    name: str | None = None,
    role: str = "a helpful AI assistant",
    tone: str | None = None,
    chattiness: str | None = None,
    interaction_style: str | None = None,
    persona_data: dict | None = None,
    goals: list | None = None,
    task_rules: list | None = None,
    output_format: str | None = None,
    rules: list | None = None,
    data: list | None = None,
    verbose: bool = False,
    friendly: bool = True,
    formal: bool | None = None,
    creative: bool | None = None,
) -> None:
    system_message = _build_system_message(
        name,
        role,
        tone,
        chattiness,
        interaction_style,
        persona_data,
        goals,
        task_rules,
        output_format,
        rules,
        data,
        verbose,
        friendly,
        formal,
        creative,
    )
    _update_system_message(system_message)


def language(text: str) -> str:
    from langdetect import detect

    return detect(text).lower()


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


def ai_translate(text: str, lang: str = "en") -> str:
    import torch
    from sacremoses import MosesPunctNormalizer

    from definers.system import catch

    if not text or not text.strip():
        return ""
    normalized_text = strip_nikud(text)
    long_paragraph_threshold = 800
    target_code = _resolve_target_code(lang)
    model = MODELS["translate"]
    tokenizer = TOKENIZERS["translate"]
    tokenizer.tgt_lang = target_code
    translated_paragraphs: list[str] = []
    for paragraph in normalized_text.split("\n"):
        if not paragraph.strip():
            translated_paragraphs.append("")
            continue
        try:
            source_language_code, source_code = (
                _resolve_source_translation_context(paragraph)
            )
        except (KeyError, Exception) as error:
            catch(error)
            translated_paragraphs.append(paragraph)
            continue
        if source_code == target_code:
            translated_paragraphs.append(paragraph)
            continue
        punct_normalizer = MosesPunctNormalizer(lang=source_language_code)
        paragraph = punct_normalizer.normalize(paragraph)
        splitter = _get_sentence_splitter(source_code)
        tokenizer.src_lang = source_code
        generation_kwargs = dict(beam_kwargs)
        generation_kwargs["num_beams"] = higher_beams
        if len(paragraph) < long_paragraph_threshold:
            try:
                translated_paragraphs.append(
                    _translate_text_segment(
                        model,
                        tokenizer,
                        paragraph,
                        target_code,
                        generation_kwargs,
                    )
                )
            except Exception as error:
                catch(error)
                translated_paragraphs.append(_translation_error(paragraph))
            continue
        try:
            translated_sentences = []
            for sentence in list(splitter(paragraph)):
                if not sentence.strip():
                    continue
                translated_sentences.append(
                    _translate_text_segment(
                        model,
                        tokenizer,
                        sentence,
                        target_code,
                        generation_kwargs,
                    )
                )
            translated_paragraphs.append(" ".join(translated_sentences))
        except Exception as error:
            catch(error)
            translated_paragraphs.append(_translation_error(paragraph))
    return "\n".join(translated_paragraphs)


def google_translate(text: str | None, lang: str = "en") -> str:
    import requests

    if lang is None or _matches_empty_text(text):
        return ""
    normalized_lang, _ = _resolve_target_language(lang)
    normalized_text = simple_text(text)
    url = (
        "https://translate.googleapis.com/translate_a/single?client=gtx"
        f"&dt=t&q={normalized_text}&sl={language(normalized_text)}&tl={normalized_lang}"
    )
    try:
        response = requests.get(url)
        translated_text = response.text.split('"')[1]
        translated_text = simple_text(translated_text)
        logger = _get_logger()
        logger.info(translated_text)
        return translated_text
    except Exception as error:
        logger = _get_logger()
        logger.exception(error)
        return ""


def duck_translate(text: str | None, lang: str = "en") -> str:
    if lang is None or _matches_empty_text(text):
        return ""
    _, normalized_lang = _resolve_target_language(lang)
    normalized_text = simple_text(text)
    url = f"https://duckduckgo.com/?q={normalized_lang} translate: {normalized_text}&ia=web"
    scraped = extract_text(
        url,
        ".module--translations-translatedtext.js-module--translations-translatedtext",
    )
    logger = _get_logger()
    if scraped is None or scraped == "":
        logger.warning("Translation Warning: Failed To Translate!")
    else:
        normalized_text = scraped
    normalized_text = simple_text(normalized_text)
    logger.info(normalized_text)
    return normalized_text


def translate_with_code(text_to_translate: str, lang: str) -> str:
    return translate_with_code_using(text_to_translate, lang, ai_translate)
