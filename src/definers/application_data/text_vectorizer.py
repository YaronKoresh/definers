from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping

import numpy as np


class DenseFeatureMatrix:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def toarray(self) -> np.ndarray:
        return np.asarray(self._values, dtype=np.float32)


class LightweightTfidfVectorizer:
    def __init__(self, token_pattern: str = r"(?u)\b\w+\b"):
        self.token_pattern = token_pattern
        self._token_regex = re.compile(token_pattern)
        self.vocabulary_: dict[str, int] = {}
        self._idf = np.empty((0,), dtype=np.float32)

    def _tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in self._token_regex.findall(str(text))]

    def fit(self, texts):
        documents = [self._tokenize(text) for text in texts]
        vocabulary: dict[str, int] = {}
        document_frequency: Counter[str] = Counter()
        for tokens in documents:
            for token in tokens:
                if token not in vocabulary:
                    vocabulary[token] = len(vocabulary)
            for token in set(tokens):
                document_frequency[token] += 1
        document_count = max(len(documents), 1)
        self.vocabulary_ = vocabulary
        self._idf = np.ones(len(vocabulary), dtype=np.float32)
        for token, index in vocabulary.items():
            self._idf[index] = float(
                math.log(
                    (1.0 + document_count) / (1.0 + document_frequency[token])
                )
                + 1.0
            )
        return self

    def transform(self, texts) -> DenseFeatureMatrix:
        if not self.vocabulary_:
            raise ValueError("Vectorizer is not fitted")
        rows = np.zeros((len(texts), len(self.vocabulary_)), dtype=np.float32)
        for row_index, text in enumerate(texts):
            token_counts = Counter(self._tokenize(text))
            if not token_counts:
                continue
            total_tokens = float(sum(token_counts.values()))
            for token, count in token_counts.items():
                token_index = self.vocabulary_.get(token)
                if token_index is None:
                    continue
                rows[row_index, token_index] = (
                    count / total_tokens
                ) * self._idf[token_index]
            row_norm = float(np.linalg.norm(rows[row_index]))
            if row_norm > 0.0:
                rows[row_index] /= row_norm
        return DenseFeatureMatrix(rows)

    def fit_transform(self, texts) -> DenseFeatureMatrix:
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names_out(self) -> np.ndarray:
        feature_names = sorted(
            self.vocabulary_.items(), key=lambda item: item[1]
        )
        return np.asarray([token for token, _ in feature_names], dtype=object)


def create_text_vectorizer(token_pattern: str = r"(?u)\b\w+\b"):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        return TfidfVectorizer(token_pattern=token_pattern)
    except Exception:
        return LightweightTfidfVectorizer(token_pattern=token_pattern)


def vectorizer_from_vocabulary(vocabulary, token_pattern: str = r"(?u)\b\w+\b"):
    vectorizer = LightweightTfidfVectorizer(token_pattern=token_pattern)
    if isinstance(vocabulary, Mapping):
        ordered_tokens = [
            str(token)
            for token, _ in sorted(
                vocabulary.items(), key=lambda item: int(item[1])
            )
        ]
    else:
        ordered_tokens = [str(token) for token in vocabulary]
    vectorizer.vocabulary_ = {
        token: index for index, token in enumerate(ordered_tokens)
    }
    vectorizer._idf = np.ones(len(ordered_tokens), dtype=np.float32)
    return vectorizer
