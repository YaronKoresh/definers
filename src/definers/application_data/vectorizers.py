from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from definers.application_data.contracts import FittedVectorizerPort, TextValues


def _normalize_texts(texts: Sequence[str]) -> list[str]:
    return [str(text) for text in texts]


def create_vectorizer(texts: TextValues) -> FittedVectorizerPort:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(token_pattern="(?u)\\b\\w+\\b")
    vectorizer.fit(_normalize_texts(texts))
    return vectorizer


def _empty_vectorized_rows(vectorizer) -> np.ndarray:
    vocabulary_size = len(getattr(vectorizer, "vocabulary_", {}) or {})
    return np.empty((0, vocabulary_size))


def vectorize(vectorizer: FittedVectorizerPort | None, texts: TextValues | None):
    if vectorizer is None or texts is None:
        return None
    if isinstance(texts, list) and not texts:
        return _empty_vectorized_rows(vectorizer)
    tfidf_matrix = vectorizer.transform(_normalize_texts(texts))
    return np.asarray(tfidf_matrix.toarray())


def _invert_vocabulary(vocabulary: Mapping[str, int]) -> dict[int, str]:
    return {index: token for token, index in vocabulary.items()}


def unvectorize(
    vectorizer: FittedVectorizerPort | None,
    vectorized_data,
):
    if vectorizer is None or vectorized_data is None:
        return None
    index_to_word = _invert_vocabulary(vectorizer.vocabulary_)
    unvectorized_texts: list[str] = []
    for row in vectorized_data:
        words = [
            index_to_word[index]
            for index, value in enumerate(row)
            if value > 0 and index in index_to_word
        ]
        unvectorized_texts.append(" ".join(words))
    return unvectorized_texts