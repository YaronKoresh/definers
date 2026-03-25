class ApplicationMlFacade:
    @staticmethod
    def contracts_module():
        import definers.application_ml.contracts as contracts_module

        return contracts_module

    @classmethod
    def normalize_answer_text(cls, text: str, required_lang: str) -> str:
        from definers.application_ml.answer_text_service import (
            AnswerTextService,
        )

        return AnswerTextService.normalize_answer_text(text, required_lang)

    @staticmethod
    def content_paths(content: object) -> list[str]:
        from definers.application_ml.answer_content_path_resolver import (
            AnswerContentPathResolver,
        )

        return AnswerContentPathResolver.content_paths(content)

    @staticmethod
    def read_answer_audio(path: str, soundfile_module, librosa_module):
        from definers.application_ml.answer_audio_loader import (
            AnswerAudioLoader,
        )

        return AnswerAudioLoader.read_answer_audio(
            path,
            soundfile_module,
            librosa_module,
        )

    @staticmethod
    def read_answer_image(path: str, image_module):
        from definers.application_ml.answer_image_loader import (
            AnswerImageLoader,
        )

        return AnswerImageLoader.read_answer_image(path, image_module)

    @staticmethod
    def append_history_message(history_items, role: str, content: str) -> None:
        from definers.application_ml.answer_text_service import (
            AnswerTextService,
        )

        AnswerTextService.append_history_message(history_items, role, content)

    @classmethod
    def prepare_answer_history(
        cls,
        history,
        runtime,
        image_module,
        soundfile_module,
        librosa_module,
    ):
        from definers.application_ml.answer_history_preparer import (
            AnswerHistoryPreparer,
        )

        return AnswerHistoryPreparer.prepare_answer_history(
            history,
            runtime,
            image_module,
            soundfile_module,
            librosa_module,
        )

    @staticmethod
    def generate_answer_without_processor(
        model,
        history,
        image_items,
        audio_items,
    ):
        from definers.application_ml.answer_generation_service import (
            AnswerGenerationService,
        )

        return AnswerGenerationService.generate_answer_without_processor(
            model,
            history,
            image_items,
            audio_items,
        )

    @staticmethod
    def generate_answer_with_processor(
        processor,
        model,
        history,
        image_items,
        audio_items,
    ):
        from definers.application_ml.answer_generation_service import (
            AnswerGenerationService,
        )

        return AnswerGenerationService.generate_answer_with_processor(
            processor,
            model,
            history,
            image_items,
            audio_items,
        )

    @classmethod
    def answer(cls, history, runtime):
        from definers.application_ml.answer_service import AnswerService

        return AnswerService.answer(history, runtime)

    @staticmethod
    def fit(model, array_adapter, logger, error_handler):
        from definers.application_ml.training import fit as training_fit

        return training_fit(
            model,
            array_adapter=array_adapter,
            logger=logger,
            error_handler=error_handler,
        )

    @staticmethod
    def feed(
        model, X_new, y_new=None, epochs: int = 1, logger=None, concatenate=None
    ):
        from definers.application_ml.training import feed as training_feed

        return training_feed(
            model,
            X_new,
            y_new,
            epochs=epochs,
            logger=logger,
            concatenate=concatenate,
        )

    @staticmethod
    def initialize_linear_regression(
        input_dim: int, model_path: str, *, runtime, factory, logger
    ):
        from definers.application_ml.training import (
            initialize_linear_regression as initialize_runtime_linear_regression,
        )

        return initialize_runtime_linear_regression(
            input_dim,
            model_path,
            runtime=runtime,
            factory=factory,
            logger=logger,
        )

    @staticmethod
    def train_linear_regression(
        X, y, model_path: str, *, learning_rate: float = 0.01, runtime=None
    ):
        from definers.application_ml.training import (
            train_linear_regression as train_runtime_linear_regression,
        )

        return train_runtime_linear_regression(
            X,
            y,
            model_path,
            learning_rate=learning_rate,
            runtime=runtime,
        )

    @staticmethod
    def linear_regression(X, y, learning_rate=0.01, epochs=50):
        from definers.application_ml.training import (
            linear_regression as training_linear_regression,
        )

        return training_linear_regression(
            X,
            y,
            learning_rate=learning_rate,
            epochs=epochs,
        )

    @staticmethod
    def extract_text_features(text: str | None, vectorizer=None):
        from definers.application_ml.inference import (
            extract_text_features as inference_extract_text_features,
        )

        return inference_extract_text_features(text, vectorizer)

    @staticmethod
    def features_to_text(predicted_features, vectorizer=None, vocabulary=None):
        from definers.application_ml.inference import (
            features_to_text as inference_features_to_text,
        )

        return inference_features_to_text(
            predicted_features,
            vectorizer=vectorizer,
            vocabulary=vocabulary,
        )

    @staticmethod
    def predict_linear_regression(X_new, model_path: str, *, factory):
        from definers.application_ml.inference import (
            predict_linear_regression as inference_predict_linear_regression,
        )

        return inference_predict_linear_regression(
            X_new, model_path, factory=factory
        )

    @staticmethod
    def init_model_file(
        task: str, turbo: bool = True, model_type: str | None = None
    ):
        from definers.application_ml.repository_sync import (
            init_model_file as repository_init_model_file,
        )

        return repository_init_model_file(
            task, turbo=turbo, model_type=model_type
        )

    @staticmethod
    def find_latest_rvc_checkpoint(folder_path: str, model_name: str):
        from definers.application_ml.rvc import (
            find_latest_rvc_checkpoint as rvc_find_latest_rvc_checkpoint,
        )

        return rvc_find_latest_rvc_checkpoint(folder_path, model_name)

    @staticmethod
    def lang_code_to_name(code: str):
        from definers.application_ml.introspection import (
            lang_code_to_name as introspection_lang_code_to_name,
        )

        return introspection_lang_code_to_name(code)

    @staticmethod
    def get_cluster_content(model, cluster_index):
        from definers.application_ml.introspection import (
            get_cluster_content as introspection_get_cluster_content,
        )

        return introspection_get_cluster_content(model, cluster_index)

    @staticmethod
    def is_clusters_model(model) -> bool:
        from definers.application_ml.introspection import (
            is_clusters_model as introspection_is_clusters_model,
        )

        return introspection_is_clusters_model(model)

    @staticmethod
    def summarize(text_to_summarize: str) -> str:
        from definers.application_ml.text_generation import (
            summarize as summarize_text,
        )

        return summarize_text(text_to_summarize)

    @staticmethod
    def map_reduce_summary(text: str, max_words: int) -> str:
        from definers.application_ml.text_generation import (
            map_reduce_summary as reduce_summary,
        )

        return reduce_summary(text, max_words)

    @staticmethod
    def summary(text: str, max_words: int = 20, min_loops: int = 1) -> str:
        from definers.application_ml.text_generation import (
            summary as summarize_runtime_text,
        )

        return summarize_runtime_text(
            text, max_words=max_words, min_loops=min_loops
        )

    @staticmethod
    def preprocess_prompt(prompt: str) -> str:
        from definers.application_ml.text_generation import (
            preprocess_prompt as preprocess_runtime_prompt,
        )

        return preprocess_runtime_prompt(prompt)

    @staticmethod
    def optimize_prompt_realism(prompt: str) -> str:
        from definers.application_ml.text_generation import (
            optimize_prompt_realism as optimize_runtime_prompt_realism,
        )

        return optimize_runtime_prompt_realism(prompt)

    @classmethod
    def getattr(cls, name: str):
        if name in {"HybridModel", "LinearRegressionTorch"}:
            import definers.application_ml.training as training_module

            return getattr(training_module, name)
        raise AttributeError(name)

    @classmethod
    def answer_model_port(cls):
        return cls.contracts_module().AnswerModelPort

    @classmethod
    def answer_processor_port(cls):
        return cls.contracts_module().AnswerProcessorPort

    @classmethod
    def answer_runtime_port(cls):
        return cls.contracts_module().AnswerRuntimePort

    @classmethod
    def answer_service_port(cls):
        return cls.contracts_module().AnswerServicePort

    @classmethod
    def error_handler_port(cls):
        return cls.contracts_module().ErrorHandlerPort

    @classmethod
    def history_message(cls):
        return cls.contracts_module().HistoryMessage

    @classmethod
    def log_port(cls):
        return cls.contracts_module().LogPort

    @classmethod
    def model_registry_port(cls):
        return cls.contracts_module().ModelRegistryPort

    @classmethod
    def prompt_processing_port(cls):
        return cls.contracts_module().PromptProcessingPort

    @classmethod
    def summary_service_port(cls):
        return cls.contracts_module().SummaryServicePort

    @classmethod
    def text_feature_extraction_port(cls):
        return cls.contracts_module().TextFeatureExtractionPort

    @classmethod
    def text_feature_vectorizer_port(cls):
        return cls.contracts_module().TextFeatureVectorizerPort

    @classmethod
    def trainable_model_port(cls):
        return cls.contracts_module().TrainableModelPort

    @classmethod
    def training_array_adapter_port(cls):
        return cls.contracts_module().TrainingArrayAdapterPort

    @classmethod
    def training_service_port(cls):
        return cls.contracts_module().TrainingServicePort


AnswerModelPort = ApplicationMlFacade.answer_model_port()
AnswerProcessorPort = ApplicationMlFacade.answer_processor_port()
AnswerRuntimePort = ApplicationMlFacade.answer_runtime_port()
AnswerServicePort = ApplicationMlFacade.answer_service_port()
ErrorHandlerPort = ApplicationMlFacade.error_handler_port()
HistoryMessage = ApplicationMlFacade.history_message()
LogPort = ApplicationMlFacade.log_port()
ModelRegistryPort = ApplicationMlFacade.model_registry_port()
PromptProcessingPort = ApplicationMlFacade.prompt_processing_port()
SummaryServicePort = ApplicationMlFacade.summary_service_port()
TextFeatureExtractionPort = ApplicationMlFacade.text_feature_extraction_port()
TextFeatureVectorizerPort = ApplicationMlFacade.text_feature_vectorizer_port()
TrainableModelPort = ApplicationMlFacade.trainable_model_port()
TrainingArrayAdapterPort = ApplicationMlFacade.training_array_adapter_port()
TrainingServicePort = ApplicationMlFacade.training_service_port()
answer = ApplicationMlFacade.answer
fit = ApplicationMlFacade.fit
feed = ApplicationMlFacade.feed
initialize_linear_regression = ApplicationMlFacade.initialize_linear_regression
train_linear_regression = ApplicationMlFacade.train_linear_regression
linear_regression = ApplicationMlFacade.linear_regression
extract_text_features = ApplicationMlFacade.extract_text_features
features_to_text = ApplicationMlFacade.features_to_text
predict_linear_regression = ApplicationMlFacade.predict_linear_regression
init_model_file = ApplicationMlFacade.init_model_file
find_latest_rvc_checkpoint = ApplicationMlFacade.find_latest_rvc_checkpoint
lang_code_to_name = ApplicationMlFacade.lang_code_to_name
get_cluster_content = ApplicationMlFacade.get_cluster_content
is_clusters_model = ApplicationMlFacade.is_clusters_model
summarize = ApplicationMlFacade.summarize
map_reduce_summary = ApplicationMlFacade.map_reduce_summary
summary = ApplicationMlFacade.summary
preprocess_prompt = ApplicationMlFacade.preprocess_prompt
optimize_prompt_realism = ApplicationMlFacade.optimize_prompt_realism


def __getattr__(name: str):
    return ApplicationMlFacade.getattr(name)


__all__ = [
    "AnswerModelPort",
    "AnswerProcessorPort",
    "AnswerServicePort",
    "AnswerRuntimePort",
    "HybridModel",
    "LinearRegressionTorch",
    "ModelRegistryPort",
    "PromptProcessingPort",
    "SummaryServicePort",
    "TextFeatureExtractionPort",
    "features_to_text",
    "TextFeatureVectorizerPort",
    "answer",
    "extract_text_features",
    "feed",
    "find_latest_rvc_checkpoint",
    "fit",
    "get_cluster_content",
    "init_model_file",
    "initialize_linear_regression",
    "is_clusters_model",
    "lang_code_to_name",
    "linear_regression",
    "predict_linear_regression",
    "TrainingServicePort",
    "train_linear_regression",
    "map_reduce_summary",
    "optimize_prompt_realism",
    "preprocess_prompt",
    "summarize",
    "summary",
]
