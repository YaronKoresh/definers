from __future__ import annotations

from definers.application_ml.facade_runtime import MlFacadeRuntime


class MlFacadeApi:
    @staticmethod
    def map_reduce_summary(text, max_words):
        from definers.application_ml.text_generation import map_reduce_summary

        return map_reduce_summary(text, max_words)

    @staticmethod
    def optimize_prompt_realism(prompt):
        from definers.application_ml.text_generation import optimize_prompt_realism

        return optimize_prompt_realism(prompt)

    @staticmethod
    def preprocess_prompt(prompt):
        from definers.application_ml.text_generation import preprocess_prompt

        return preprocess_prompt(prompt)

    @staticmethod
    def summarize(text_to_summarize):
        from definers.application_ml.text_generation import summarize

        return summarize(text_to_summarize)

    @staticmethod
    def summary(text, max_words=20, min_loops=1):
        from definers.application_ml.text_generation import summary

        return summary(text, max_words=max_words, min_loops=min_loops)

    @staticmethod
    def answer(history, *, answer_fn, models, processors):
        return answer_fn(
            history,
            runtime=MlFacadeRuntime.build_answer_runtime(models, processors),
        )

    @staticmethod
    def linear_regression(X, y, *, linear_regression_fn, learning_rate=0.01, epochs=50):
        return linear_regression_fn(
            X,
            y,
            learning_rate=learning_rate,
            epochs=epochs,
        )

    @staticmethod
    def initialize_linear_regression(
        input_dim,
        model_path,
        *,
        initialize_linear_regression_fn,
        factory,
        device_fn,
        logger,
    ):
        return initialize_linear_regression_fn(
            input_dim,
            model_path,
            runtime=MlFacadeRuntime.build_linear_regression_runtime(device_fn),
            factory=factory,
            logger=logger,
        )

    @staticmethod
    def train_linear_regression(
        X,
        y,
        model_path,
        *,
        train_linear_regression_fn,
        initialize_linear_regression_fn,
        device_fn,
        learning_rate=0.01,
    ):
        return train_linear_regression_fn(
            X,
            y,
            model_path,
            learning_rate=learning_rate,
            runtime=MlFacadeRuntime.build_train_linear_regression_runtime(
                device_fn,
                initialize_linear_regression_fn,
            ),
        )

    @staticmethod
    def init_model_file(task, *, init_model_file_fn, turbo=True, model_type=None):
        return init_model_file_fn(task, turbo=turbo, model_type=model_type)

    @staticmethod
    def extract_text_features(text, *, extract_text_features_fn, vectorizer=None):
        return extract_text_features_fn(text, vectorizer)

    @staticmethod
    def predict_linear_regression(
        X_new,
        model_path,
        *,
        predict_linear_regression_fn,
        factory,
    ):
        return predict_linear_regression_fn(X_new, model_path, factory=factory)

    @staticmethod
    def features_to_text(
        predicted_features,
        *,
        features_to_text_fn,
        vectorizer=None,
        vocabulary=None,
    ):
        return features_to_text_fn(
            predicted_features,
            vectorizer=vectorizer,
            vocabulary=vocabulary,
        )