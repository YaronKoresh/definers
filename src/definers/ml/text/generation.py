def ensure_summary_runtime():
    from definers.constants import MODELS, TOKENIZERS
    from definers.system.download_activity import report_download_activity

    if MODELS["summary"] is None or TOKENIZERS["summary"] is None:
        from definers.ml import init_pretrained_model

        report_download_activity(
            "Load summary model",
            detail="Initializing the summary runtime.",
            phase="model",
        )
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
    from definers.system.download_activity import create_activity_reporter

    report = create_activity_reporter(2)
    ensure_summary_runtime()
    report(
        1,
        "Encode summary prompt",
        detail="Encoding the summary prompt.",
    )
    encoded = encode_summary_prompt(text_to_summarize)
    report(
        2,
        "Generate summary text",
        detail="Generating the summary output.",
    )
    generated = MODELS["summary"].generate(
        **encoded,
        **summary_generation_kwargs(),
        max_length=512,
    )
    return TOKENIZERS["summary"].decode(generated[0], skip_special_tokens=True)


def map_reduce_summary(text: str, max_words: int) -> str:
    from definers.system.download_activity import (
        create_activity_reporter,
        report_download_activity,
    )

    chunk_size = 60
    overlap = 10
    iteration = 0
    while len(text.split()) > max_words:
        iteration += 1
        report_download_activity(
            "Map-reduce summary pass",
            detail=f"Running map-reduce pass {iteration}.",
            phase="step",
        )
        chunk_summaries = []
        chunk_texts = list(summary_chunks(text, chunk_size, overlap))
        chunk_report = create_activity_reporter(len(chunk_texts) or 1)
        for chunk_index, chunk_text in enumerate(chunk_texts, start=1):
            chunk_report(
                chunk_index,
                "Summarize chunks",
                detail=(
                    f"Summarizing chunk {chunk_index}/{len(chunk_texts) or 1} "
                    f"in pass {iteration}."
                ),
            )
            chunk_summaries.append(summarize(chunk_text))
        text = " ".join(chunk_summaries)
    return summarize(text)


def summary(text: str, max_words: int = 20, min_loops: int = 1) -> str:
    from definers.system import log
    from definers.system.download_activity import report_download_activity
    from definers.text import strip_nikud

    summarized_text = strip_nikud(text)
    words_count = len(summarized_text.split())
    iteration = 0
    while words_count > max_words or min_loops > 0:
        iteration += 1
        report_download_activity(
            "Iterative summary pass",
            detail=f"Running summary pass {iteration}.",
            phase="step",
        )
        if words_count > 80:
            summarized_text = map_reduce_summary(summarized_text, max_words)
        else:
            summarized_text = summarize(summarized_text)
        min_loops = min_loops - 1
        words_count = len(summarized_text.split())
    log("Summary", summarized_text)
    return summarized_text


def preprocess_prompt(prompt: str) -> str:
    from definers.system.download_activity import create_activity_reporter
    from definers.text import simple_text

    report = create_activity_reporter(3)
    report(
        1,
        "Normalize prompt language",
        detail="Normalizing the prompt language.",
    )
    processed_prompt = normalize_prompt_language(prompt)
    report(
        2,
        "Clean prompt text",
        detail="Cleaning the prompt text.",
    )
    processed_prompt = simple_text(processed_prompt)
    report(
        3,
        "Summarize prompt",
        detail="Condensing the prompt for generation.",
    )
    processed_prompt = summary(processed_prompt, max_words=14)
    return simple_text(processed_prompt)


def optimize_prompt_realism(prompt: str) -> str:
    from definers.constants import general_positive_prompt

    processed_prompt = preprocess_prompt(prompt)
    return (
        f"{processed_prompt}, {general_positive_prompt}, "
        f"{general_positive_prompt}."
    )
