class InferenceService:
    @staticmethod
    def catch(error: Exception) -> None:
        try:
            from definers.system import catch as runtime_catch

            runtime_catch(error)
        except Exception:
            return None

    @staticmethod
    def extract_text_features(text, vectorizer=None):
        from definers.ml.text_feature_extractor import (
            TextFeatureExtractor,
        )

        return TextFeatureExtractor.extract(text, vectorizer)

    @staticmethod
    def rank_reconstructed_tokens(
        predicted_features, feature_names
    ) -> list[str]:
        from definers.ml.text_feature_reconstructor import (
            TextFeatureReconstructor,
        )

        return TextFeatureReconstructor.rank_tokens(
            predicted_features,
            feature_names,
        )

    @staticmethod
    def features_to_text(predicted_features, vectorizer=None, vocabulary=None):
        from definers.ml.text_feature_reconstructor import (
            TextFeatureReconstructor,
        )

        return TextFeatureReconstructor.reconstruct(
            predicted_features,
            vectorizer=vectorizer,
            vocabulary=vocabulary,
        )

    @staticmethod
    def sanitize_prediction_path(model_path: str):
        from definers.ml.regression_predictor import (
            RegressionPredictor,
        )

        return RegressionPredictor.sanitize_path(model_path)

    @staticmethod
    def predict_linear_regression(X_new, model_path: str, *, factory):
        sanitized_path = _sanitize_prediction_path(model_path)
        if sanitized_path is None:
            return None
        from definers.ml.regression_predictor import (
            RegressionPredictor,
        )

        return RegressionPredictor.predict(
            X_new,
            sanitized_path,
            factory=factory,
        )


_catch = InferenceService.catch
_rank_reconstructed_tokens = InferenceService.rank_reconstructed_tokens
_sanitize_prediction_path = InferenceService.sanitize_prediction_path
extract_text_features = InferenceService.extract_text_features
features_to_text = InferenceService.features_to_text
predict_linear_regression = InferenceService.predict_linear_regression
