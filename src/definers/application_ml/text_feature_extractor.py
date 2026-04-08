class TextFeatureExtractor:
    @staticmethod
    def catch(error: Exception) -> None:
        try:
            from definers.system import catch as runtime_catch

            runtime_catch(error)
        except Exception:
            return None

    @classmethod
    def extract(cls, text, vectorizer=None):
        import numpy as np

        from definers.application_data.text_vectorizer import (
            create_text_vectorizer,
        )

        try:
            active_vectorizer = vectorizer
            if active_vectorizer is None:
                active_vectorizer = create_text_vectorizer(
                    token_pattern="(?u)\\b\\w+\\b"
                )
                tfidf_matrix = active_vectorizer.fit_transform([text])
            else:
                tfidf_matrix = active_vectorizer.transform([text])
            return tfidf_matrix.toarray().flatten().astype(np.float32)
        except Exception as error:
            cls.catch(error)
            return None
