from __future__ import annotations

from definers.application_ml.contracts import (
    PromptProcessingPort,
    SummaryServicePort,
)
from definers.constants import (
    MODELS,
    TOKENIZERS,
    beam_kwargs,
    general_positive_prompt,
    higher_beams,
)
from definers.cuda import device
from definers.system import log
from definers.text import ai_translate, language, simple_text, strip_nikud


def _encode_summary_prompt(text_to_summarize):
    prefix = "summarize: "
    encoded = TOKENIZERS["summary"](
        prefix + text_to_summarize,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    return {key: tensor.to(device()) for key, tensor in encoded.items()}


def _summary_generation_kwargs():
    generation_kwargs = dict(beam_kwargs)
    generation_kwargs["num_beams"] = higher_beams
    return generation_kwargs


def _summary_chunks(text, chunk_size, overlap):
    words = text.split()
    for index in range(0, len(words), chunk_size - overlap):
        yield " ".join(words[index : index + chunk_size])


def _normalize_prompt_language(prompt):
    processed_prompt = prompt
    if language(processed_prompt) != "en":
        processed_prompt = ai_translate(processed_prompt)
    return processed_prompt


def summarize(text_to_summarize):
    encoded = _encode_summary_prompt(text_to_summarize)
    generated = MODELS["summary"].generate(
        **encoded,
        **_summary_generation_kwargs(),
        max_length=512,
    )
    return TOKENIZERS["summary"].decode(generated[0], skip_special_tokens=True)


def map_reduce_summary(text, max_words):
    chunk_size = 60
    overlap = 10
    while len(text.split()) > max_words:
        chunk_summaries = []
        for chunk_text in _summary_chunks(text, chunk_size, overlap):
            chunk_summaries.append(summarize(chunk_text))
        text = " ".join(chunk_summaries)
    return summarize(text)


def summary(text, max_words=20, min_loops=1):
    summarized_text = strip_nikud(text)
    words_count = len(summarized_text.split())
    while words_count > max_words or min_loops > 0:
        if words_count > 80:
            summarized_text = map_reduce_summary(summarized_text, max_words)
        else:
            summarized_text = summarize(summarized_text)
        min_loops = min_loops - 1
        words_count = len(summarized_text.split())
    log("Summary", summarized_text)
    return summarized_text


def preprocess_prompt(prompt):
    processed_prompt = _normalize_prompt_language(prompt)
    processed_prompt = simple_text(processed_prompt)
    processed_prompt = summary(processed_prompt, max_words=14)
    return simple_text(processed_prompt)


def optimize_prompt_realism(prompt):
    processed_prompt = preprocess_prompt(prompt)
    return f"{processed_prompt}, {general_positive_prompt}, {general_positive_prompt}."


class TextGenerationService(PromptProcessingPort, SummaryServicePort):
    def summarize(self, text_to_summarize: str) -> str:
        return summarize(text_to_summarize)

    def map_reduce_summary(self, text: str, max_words: int) -> str:
        return map_reduce_summary(text, max_words)

    def summary(
        self,
        text: str,
        max_words: int = 20,
        min_loops: int = 1,
    ) -> str:
        return summary(text, max_words=max_words, min_loops=min_loops)

    def preprocess_prompt(self, prompt: str) -> str:
        return preprocess_prompt(prompt)

    def optimize_prompt_realism(self, prompt: str) -> str:
        return optimize_prompt_realism(prompt)


text_generation_service = TextGenerationService()
prompt_processing_service = text_generation_service
summary_service = text_generation_service
