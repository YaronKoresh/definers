from __future__ import annotations

from typing import Any, Protocol, TypedDict


class HistoryMessage(TypedDict):
    role: str
    content: object


class ChatTokenizerPort(Protocol):
    def apply_chat_template(
        self,
        history: list[HistoryMessage],
        _tokenize: bool,
        _add_generation_prompt: bool,
    ) -> str: ...


class AnswerBatchPort(Protocol):
    def to(self, target: Any) -> Any: ...


class AnswerProcessorPort(Protocol):
    tokenizer: ChatTokenizerPort

    def __call__(
        self,
        *,
        text: str,
        images: list[Any] | None,
        _audios: list[Any] | None,
        _return_tensors: str,
    ) -> AnswerBatchPort: ...

    def batch_decode(
        self,
        output_ids: Any,
        *,
        _skip_special_tokens: bool,
        _clean_up_tokenization_spaces: bool,
    ) -> list[str]: ...


class AnswerModelPort(Protocol):
    def generate(self, *args: Any, **kwargs: Any) -> Any: ...


class RuntimeMappingPort(Protocol):
    def get(self, key: str, default: Any = None) -> Any: ...


class AnswerRuntimePort(Protocol):
    MODELS: RuntimeMappingPort
    PROCESSORS: RuntimeMappingPort
    SYSTEM_MESSAGE: str
    common_audio_formats: list[str]
    iio_formats: list[str]


class TrainableModelPort(Protocol):
    X_all: Any

    def fit(self, X: Any, y: Any = None) -> Any: ...


class TrainingArrayAdapterPort(Protocol):
    def get_max_shapes(self, *data: Any) -> Any: ...

    def cupy_to_numpy(self, value: Any) -> Any: ...

    def reshape_numpy(self, value: Any, lengths: Any) -> Any: ...

    def numpy_to_cupy(self, value: Any) -> Any: ...

    def catch(self, error: Exception) -> Any: ...


class ArrayConcatenatePort(Protocol):
    def __call__(self, arrays: Any, _axis: int = 0) -> Any: ...


class LogPort(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class ErrorHandlerPort(Protocol):
    def __call__(self, error: Exception) -> Any: ...


class TextFeatureMatrixPort(Protocol):
    def toarray(self) -> Any: ...


class TextFeatureVectorizerPort(Protocol):
    def fit_transform(
        self, _raw_documents: list[str]
    ) -> TextFeatureMatrixPort: ...


class TextFeatureNameVectorizerPort(TextFeatureVectorizerPort, Protocol):
    def fit(self, _raw_documents: list[str]) -> Any: ...

    def get_feature_names_out(self) -> Any: ...


class AnswerServicePort(Protocol):
    def answer(self, history: list[HistoryMessage]) -> Any: ...


class TrainingServicePort(Protocol):
    def fit(self, model: TrainableModelPort) -> TrainableModelPort: ...


class ModelRegistryPort(Protocol):
    def resolve(self, task: str) -> Any: ...


class TextFeatureExtractionPort(Protocol):
    def extract_text_features(
        self,
        text: str | None,
        vectorizer: TextFeatureVectorizerPort | None = None,
    ) -> Any: ...


class TextFeatureReconstructionPort(Protocol):
    def features_to_text(
        self,
        predicted_features: Any,
        vectorizer: TextFeatureNameVectorizerPort | None = None,
        vocabulary: list[str] | None = None,
    ) -> str | None: ...


class PromptProcessingPort(Protocol):
    def preprocess_prompt(self, prompt: str) -> str: ...

    def optimize_prompt_realism(self, prompt: str) -> str: ...


class SummaryServicePort(Protocol):
    def summarize(self, text_to_summarize: str) -> str: ...

    def map_reduce_summary(self, text: str, max_words: int) -> str: ...

    def summary(
        self,
        text: str,
        max_words: int = 20,
        min_loops: int = 1,
    ) -> str: ...


class PathSecurityPort(Protocol):
    def __call__(self, path: str) -> str: ...


class LinearRegressionModelPort(Protocol):
    def __call__(self, inputs: Any) -> Any: ...

    def parameters(self) -> Any: ...

    def state_dict(self) -> Any: ...

    def load_state_dict(self, state_dict: Any) -> Any: ...

    def eval(self) -> Any: ...

    def to(self, target: Any) -> Any: ...

    def cuda(self) -> Any: ...


class LinearRegressionFactoryPort(Protocol):
    def __call__(self, input_dim: int) -> LinearRegressionModelPort: ...


class LinearRegressionRuntimePort(Protocol):
    LinearRegressionTorch: LinearRegressionFactoryPort

    def initialize_linear_regression(
        self, input_dim: int, model_path: str
    ) -> LinearRegressionModelPort: ...

    def device(self) -> Any: ...
