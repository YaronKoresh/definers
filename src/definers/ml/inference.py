def catch(error: Exception) -> None:
    try:
        from definers.system import catch as runtime_catch

        runtime_catch(error)
    except Exception:
        return None


def extract_text_features(text, vectorizer=None):
    from definers.ml.text.extract import TextFeatureExtractor

    return TextFeatureExtractor.extract(text, vectorizer)


def rank_reconstructed_tokens(predicted_features, feature_names) -> list[str]:
    from definers.ml.text.reconstruct import TextFeatureReconstructor

    return TextFeatureReconstructor.rank_tokens(
        predicted_features,
        feature_names,
    )


def features_to_text(predicted_features, vectorizer=None, vocabulary=None):
    from definers.ml.text.reconstruct import TextFeatureReconstructor

    return TextFeatureReconstructor.reconstruct(
        predicted_features,
        vectorizer=vectorizer,
        vocabulary=vocabulary,
    )


def sanitize_prediction_path(model_path: str):
    from definers.ml.regression_predictor import RegressionPredictor

    return RegressionPredictor.sanitize_path(model_path)


def predict_linear_regression(X_new, model_path: str, *, factory):
    sanitized_path = _sanitize_prediction_path(model_path)
    if sanitized_path is None:
        return None
    from definers.ml.regression_predictor import RegressionPredictor

    return RegressionPredictor.predict(
        X_new,
        sanitized_path,
        factory=factory,
    )


_catch = catch
_rank_reconstructed_tokens = rank_reconstructed_tokens
_sanitize_prediction_path = sanitize_prediction_path
