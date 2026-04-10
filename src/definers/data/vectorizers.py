from definers.runtime_numpy import get_array_module


def normalize_texts(texts) -> list[str]:
    return [str(text) for text in texts]


def create_vectorizer(texts):
    from definers.data.text.vectorizer import (
        create_text_vectorizer,
    )

    normalized_texts = normalize_texts(texts)
    if not normalized_texts:
        raise ValueError("texts must not be empty")
    vectorizer = create_text_vectorizer(token_pattern="(?u)\\b\\w+\\b")
    vectorizer.fit(normalized_texts)
    return vectorizer


def empty_vectorized_rows(vectorizer):
    np = get_array_module()

    vocabulary_size = len(getattr(vectorizer, "vocabulary_", {}) or {})
    return np.empty((0, vocabulary_size))


def vectorize(vectorizer, texts):
    np = get_array_module()

    if vectorizer is None or texts is None:
        return None
    if isinstance(texts, list) and not texts:
        return empty_vectorized_rows(vectorizer)
    tfidf_matrix = vectorizer.transform(normalize_texts(texts))
    return np.asarray(tfidf_matrix.toarray())


def invert_vocabulary(vocabulary) -> dict[int, str]:
    return {index: token for token, index in vocabulary.items()}


def unvectorize(vectorizer, vectorized_data):
    if vectorizer is None or vectorized_data is None:
        return None
    index_to_word = invert_vocabulary(vectorizer.vocabulary_)
    unvectorized_texts: list[str] = []
    for row in vectorized_data:
        words = [
            index_to_word[index]
            for index, value in enumerate(row)
            if value > 0 and index in index_to_word
        ]
        unvectorized_texts.append(" ".join(words))
    return unvectorized_texts
