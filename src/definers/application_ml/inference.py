from __future__ import annotations

from typing import Any

import numpy as np

from definers.application_ml.contracts import (
    LinearRegressionFactoryPort,
    TextFeatureNameVectorizerPort,
    TextFeatureVectorizerPort,
)


def _catch(error: Exception) -> None:
    try:
        from definers.system import catch as runtime_catch

        runtime_catch(error)
    except Exception:
        return None


def extract_text_features(
    text: str | None,
    vectorizer: TextFeatureVectorizerPort | None = None,
):
    from sklearn.feature_extraction.text import TfidfVectorizer

    try:
        active_vectorizer = vectorizer or TfidfVectorizer(
            token_pattern="(?u)\\b\\w+\\b"
        )
        tfidf_matrix = active_vectorizer.fit_transform([text])
        return tfidf_matrix.toarray().flatten().astype(np.float32)
    except Exception as error:
        _catch(error)
        return None


def features_to_text(
    predicted_features: Any,
    vectorizer: TextFeatureNameVectorizerPort | None = None,
    vocabulary: list[str] | None = None,
) -> str | None:
    from sklearn.feature_extraction.text import TfidfVectorizer

    if vectorizer is None and vocabulary is None:
        raise ValueError(
            "Either a vectorizer or a vocabulary must be provided."
        )
    try:
        active_vectorizer = vectorizer
        if active_vectorizer is None:
            active_vectorizer = TfidfVectorizer(
                token_pattern="(?u)\\b\\w+\\b"
            )
            active_vectorizer.fit(vocabulary)
        tfidf_matrix = np.asarray(predicted_features).reshape(1, -1)
        word_indices = tfidf_matrix.nonzero()[1]
        feature_names = active_vectorizer.get_feature_names_out()
        reconstructed_words = [feature_names[i] for i in word_indices]
        return " ".join(reconstructed_words)
    except Exception as error:
        _catch(error)
        return None


def _sanitize_prediction_path(model_path: str) -> str | None:
    from definers.system import secure_path

    try:
        return secure_path(model_path)
    except Exception as error:
        _catch(error)
        return None


def predict_linear_regression(
    X_new,
    model_path: str,
    *,
    factory: LinearRegressionFactoryPort,
):
    import torch

    sanitized_path = _sanitize_prediction_path(model_path)
    if sanitized_path is None:
        return None
    try:
        input_dim = X_new.shape[1]
        model_torch = factory(input_dim)
        model_torch.load_state_dict(
            torch.load(sanitized_path, map_location="cpu")
        )
        model_torch.eval()
        target_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if hasattr(model_torch, "to"):
            model_torch.to(target_device)
        X_new_torch = torch.tensor(
            X_new,
            dtype=torch.float32,
            device=target_device,
        )
        with torch.no_grad():
            predictions_torch = model_torch(X_new_torch).reshape(-1)
        return predictions_torch.cpu().numpy().reshape(-1)
    except Exception as error:
        _catch(error)
        return None
