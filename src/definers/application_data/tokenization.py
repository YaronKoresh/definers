from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from definers.constants import TOKENIZERS


def _runtime():
    import definers.data as data_module

    return data_module


def _row_to_text(row: dict[str, Any]) -> str:
    features_strings: list[str] = []
    for value in row.values():
        if isinstance(value, (list, np.ndarray)):
            features_strings.extend(map(str, value))
        elif value is not None:
            features_strings.append(str(value))
    return " ".join(features_strings)


def init_tokenizer(
    model_name: str | None = None,
    tokenizer_type: str | None = None,
):
    model_name = model_name or "google-bert/bert-base-multilingual-cased"
    tokenizer_type = tokenizer_type or "general"
    current_model = TOKENIZERS.get(tokenizer_type, {}).get("model_name")
    if (not TOKENIZERS[tokenizer_type]["tokenizer"]) or current_model != model_name:
        from transformers import AutoTokenizer

        TOKENIZERS[tokenizer_type] = {
            "tokenizer": AutoTokenizer.from_pretrained(model_name),
            "model_name": model_name,
        }
    return TOKENIZERS[tokenizer_type]["tokenizer"]


def tokenize_and_pad(rows, tokenizer=None):
    runtime = _runtime()
    if tokenizer is None:
        tokenizer = init_tokenizer()
    features_list: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            features_list.append(_row_to_text(row))
        elif isinstance(row, str):
            features_list.append(row)
        else:
            return rows
    tokenized_inputs = tokenizer(
        features_list,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return runtime.two_dim_numpy(tokenized_inputs["input_ids"])


def tokenize_or_vectorize(data, tokenizer=None):
    if data is None:
        return data
    try:
        arr = np.array(data)
        if arr.dtype.kind in ("U", "S", "O"):
            return tokenize_and_pad(arr.tolist(), tokenizer)
    except Exception:
        return data
    return data
