class TextFeatureReconstructor:
    @staticmethod
    def catch(error: Exception) -> None:
        try:
            from definers.system import catch as runtime_catch

            runtime_catch(error)
        except Exception:
            return None

    @staticmethod
    def rank_tokens(predicted_features, feature_names) -> list[str]:
        import numpy as np

        weighted_tokens = []
        flattened_features = np.asarray(predicted_features).reshape(-1)
        for index, value in enumerate(flattened_features):
            if value > 0 and index < len(feature_names):
                weighted_tokens.append(
                    (float(value), str(feature_names[index]))
                )
        weighted_tokens.sort(key=lambda item: (-item[0], item[1]))
        return [token for _, token in weighted_tokens]

    @classmethod
    def reconstruct(cls, predicted_features, vectorizer=None, vocabulary=None):
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
            feature_names = active_vectorizer.get_feature_names_out()
            reconstructed_words = cls.rank_tokens(
                predicted_features,
                feature_names,
            )
            return " ".join(reconstructed_words)
        except Exception as error:
            cls.catch(error)
            return None
