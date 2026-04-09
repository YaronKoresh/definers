def ensure_summary_runtime():
    from definers.constants import MODELS, TOKENIZERS

    if MODELS["summary"] is None or TOKENIZERS["summary"] is None:
        from definers.ml import init_pretrained_model

        init_pretrained_model("summary")


def encode_summary_prompt(text_to_summarize):
    from definers.constants import TOKENIZERS
    from definers.cuda import device

    ensure_summary_runtime()
    prefix = "summarize: "
    encoded = TOKENIZERS["summary"](
        prefix + text_to_summarize,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    return {key: tensor.to(device()) for key, tensor in encoded.items()}


def summary_generation_kwargs():
    from definers.constants import beam_kwargs, higher_beams

    generation_kwargs = dict(beam_kwargs)
    generation_kwargs["num_beams"] = higher_beams
    return generation_kwargs


def summary_chunks(text, chunk_size, overlap):
    words = text.split()
    for index in range(0, len(words), chunk_size - overlap):
        yield " ".join(words[index : index + chunk_size])


def normalize_prompt_language(prompt):
    from definers.text import ai_translate, language

    processed_prompt = prompt
    if language(processed_prompt) != "en":
        processed_prompt = ai_translate(processed_prompt)
    return processed_prompt


def summarize(text_to_summarize: str) -> str:
    from definers.constants import MODELS, TOKENIZERS

    ensure_summary_runtime()
    encoded = encode_summary_prompt(text_to_summarize)
    generated = MODELS["summary"].generate(
        **encoded,
        **summary_generation_kwargs(),
        max_length=512,
    )
    return TOKENIZERS["summary"].decode(generated[0], skip_special_tokens=True)


def map_reduce_summary(text: str, max_words: int) -> str:
    chunk_size = 60
    overlap = 10
    while len(text.split()) > max_words:
        chunk_summaries = []
        for chunk_text in summary_chunks(text, chunk_size, overlap):
            chunk_summaries.append(summarize(chunk_text))
        text = " ".join(chunk_summaries)
    return summarize(text)


def summary(text: str, max_words: int = 20, min_loops: int = 1) -> str:
    from definers.system import log
    from definers.text import strip_nikud

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


def preprocess_prompt(prompt: str) -> str:
    from definers.text import simple_text

    processed_prompt = normalize_prompt_language(prompt)
    processed_prompt = simple_text(processed_prompt)
    processed_prompt = summary(processed_prompt, max_words=14)
    return simple_text(processed_prompt)


def optimize_prompt_realism(prompt: str) -> str:
    from definers.constants import general_positive_prompt

    processed_prompt = preprocess_prompt(prompt)
    return (
        f"{processed_prompt}, {general_positive_prompt}, "
        f"{general_positive_prompt}."
    )
