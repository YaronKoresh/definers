from __future__ import annotations

import re
import sys
from collections.abc import Callable

from definers.constants import (
    MODELS,
    TOKENIZERS,
    beam_kwargs,
    higher_beams,
    language_codes,
    unesco_mapping,
)
from definers.cuda import device
from definers.web import extract_text

from .text_transforms import language, simple_text, strip_nikud


def resolve_primary_unesco_code(code: str | list[str] | tuple[str, ...]) -> str:
    if isinstance(code, (list, tuple)):
        return code[0]
    return code


def resolve_target_code(lang: str) -> str:
    return resolve_primary_unesco_code(unesco_mapping[lang])


def resolve_source_translation_context(paragraph: str) -> tuple[str, str]:
    detected_language = language(paragraph)
    return detected_language, resolve_primary_unesco_code(
        unesco_mapping[detected_language]
    )


def resolve_target_language(lang: str) -> tuple[str, str]:
    normalized_lang = simple_text(lang)
    if normalized_lang in language_codes:
        return normalized_lang, language_codes[normalized_lang]
    for code, language_name in language_codes.items():
        if language_name == normalized_lang:
            return code, language_name
    return normalized_lang, normalized_lang


def get_sentence_splitter(source_code: str):
    try:
        import importlib

        get_split_algo = importlib.import_module(
            "stopes.pipelines.monolingual.utils.sentence_split"
        ).get_split_algo
    except ImportError:

        def get_split_algo(*_args, **_kwargs):
            return lambda value: [value]

    return get_split_algo(source_code[:3], "default")


def translation_error(paragraph: str) -> str:
    return f"[Translation Error: {paragraph[:30]}...]"


def get_logger():
    active_logger = sys.modules.get("definers.logger")
    if active_logger is not None:
        return active_logger
    from definers import logger

    return logger


def matches_empty_text(value: str | None) -> bool:
    return value is None or value.strip() == ""


def translate_text_segment(
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


def ai_translate(text: str, lang: str = "en") -> str:
    from sacremoses import MosesPunctNormalizer

    from definers.system import catch

    if not text or not text.strip():
        return ""
    normalized_text = strip_nikud(text)
    long_paragraph_threshold = 800
    target_code = resolve_target_code(lang)
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
                resolve_source_translation_context(paragraph)
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
        splitter = get_sentence_splitter(source_code)
        tokenizer.src_lang = source_code
        generation_kwargs = dict(beam_kwargs)
        generation_kwargs["num_beams"] = higher_beams
        if len(paragraph) < long_paragraph_threshold:
            try:
                translated_paragraphs.append(
                    translate_text_segment(
                        model,
                        tokenizer,
                        paragraph,
                        target_code,
                        generation_kwargs,
                    )
                )
            except Exception as error:
                catch(error)
                translated_paragraphs.append(translation_error(paragraph))
            continue
        try:
            translated_sentences = []
            for sentence in list(splitter(paragraph)):
                if not sentence.strip():
                    continue
                translated_sentences.append(
                    translate_text_segment(
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
            translated_paragraphs.append(translation_error(paragraph))
    return "\n".join(translated_paragraphs)


def google_translate(text: str | None, lang: str = "en") -> str:
    import requests

    if lang is None or matches_empty_text(text):
        return ""
    normalized_lang, _ = resolve_target_language(lang)
    normalized_text = simple_text(text)
    url = (
        "https://translate.googleapis.com/translate_a/single?client=gtx"
        f"&dt=t&q={normalized_text}&sl={language(normalized_text)}&tl={normalized_lang}"
    )
    try:
        response = requests.get(url)
        translated_text = response.text.split('"')[1]
        translated_text = simple_text(translated_text)
        logger = get_logger()
        logger.info(translated_text)
        return translated_text
    except Exception as error:
        logger = get_logger()
        logger.exception(error)
        return ""


def duck_translate(text: str | None, lang: str = "en") -> str:
    if lang is None or matches_empty_text(text):
        return ""
    _, normalized_lang = resolve_target_language(lang)
    normalized_text = simple_text(text)
    url = f"https://duckduckgo.com/?q={normalized_lang} translate: {normalized_text}&ia=web"
    scraped = extract_text(
        url,
        ".module--translations-translatedtext.js-module--translations-translatedtext",
    )
    logger = get_logger()
    if scraped is None or scraped == "":
        logger.warning("Translation Warning: Failed To Translate!")
    else:
        normalized_text = scraped
    normalized_text = simple_text(normalized_text)
    logger.info(normalized_text)
    return normalized_text


def translate_with_code(text_to_translate: str, lang: str) -> str:
    return translate_with_code_using(text_to_translate, lang, ai_translate)
