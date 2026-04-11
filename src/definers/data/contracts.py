from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeAlias, TypedDict

ColumnName: TypeAlias = str
ColumnNames: TypeAlias = Sequence[ColumnName]
DatasetRow: TypeAlias = Mapping[str, Any]
DatasetRows: TypeAlias = Sequence[DatasetRow]
TextValues: TypeAlias = Sequence[str]
VectorizedValue: TypeAlias = Sequence[Sequence[float]] | Any
LoadedValue: TypeAlias = Any
FeatureLabelPair: TypeAlias = tuple[Any, Any]
FeatureLabelBatch: TypeAlias = tuple[list[dict[str, Any]], list[dict[str, Any]]]


class PrepareDataCacheEntry(TypedDict):
    remote_src: Any
    features: Any
    labels: Any
    url_type: Any
    revision: Any
    drop: Any
    order_by: Any
    stratify: Any
    val_frac: float
    test_frac: float
    batch_size: int


class ColumnDatasetPort(Protocol):
    column_names: ColumnNames

    def __getitem__(self, column_name: ColumnName) -> Sequence[Any]: ...

    def remove_columns(self, column_names: ColumnNames) -> Any: ...


class DatasetShaperPort(Protocol):
    def drop_columns(
        self,
        dataset: ColumnDatasetPort,
        drop_list: ColumnNames | None,
    ) -> Any: ...

    def select_columns(
        self,
        dataset: ColumnDatasetPort,
        cols: ColumnNames | None,
    ) -> Any: ...

    def select_rows(
        self,
        dataset: ColumnDatasetPort,
        start_index: int,
        end_index: int,
    ) -> Any: ...

    def split_columns(
        self,
        data: Any,
        labels: ColumnNames | None,
        is_batch: bool = False,
    ) -> FeatureLabelPair | FeatureLabelBatch: ...


class DatasetLoaderPort(Protocol):
    def load_as_numpy(
        self, path: str, training: bool = False
    ) -> LoadedValue: ...


class PrepareDataCacheControlPort(Protocol):
    def clear_prepare_data_cache(self) -> int: ...

    def prepare_data_cache_manifest(self) -> list[PrepareDataCacheEntry]: ...

    def fetch_dataset(
        self,
        src: str,
        url_type: str | None = None,
        revision: str | None = None,
        sample_rows: int | None = None,
    ) -> LoadedValue: ...

    def files_to_dataset(
        self,
        features_paths: Sequence[str],
        labels_paths: Sequence[str] | None = None,
    ) -> LoadedValue: ...

    def load_source(
        self,
        remote_src: str | None = None,
        features: Sequence[str] | None = None,
        labels: Sequence[str] | None = None,
        url_type: str | None = None,
        revision: str | None = None,
    ) -> LoadedValue: ...


class FittedVectorizerPort(Protocol):
    vocabulary_: Mapping[str, int]

    def fit(self, texts: TextValues) -> Any: ...

    def transform(self, texts: TextValues) -> Any: ...


class VectorizationPort(Protocol):
    def create_vectorizer(self, texts: TextValues) -> FittedVectorizerPort: ...

    def vectorize(
        self,
        vectorizer: FittedVectorizerPort | None,
        texts: TextValues | None,
    ) -> VectorizedValue | None: ...

    def unvectorize(
        self,
        vectorizer: FittedVectorizerPort | None,
        vectorized_data: VectorizedValue | None,
    ) -> list[str] | None: ...


class TensorAdapterPort(Protocol):
    def cupy_to_numpy(self, value: Any) -> Any: ...

    def numpy_to_cupy(self, value: Any) -> Any: ...

    def dtype(self, size: int = 16, is_float: bool = True) -> Any: ...

    def reshape_numpy(
        self,
        value: Any,
        fill_value: int = 0,
        lengths: Sequence[int] | None = None,
    ) -> Any: ...

    def get_max_shapes(self, *data: Any) -> list[int]: ...

    def convert_tensor_dtype(self, tensor: Any) -> Any: ...

    def tokenize_and_pad(
        self,
        rows: Sequence[str] | Sequence[dict[str, Any]],
        tokenizer: Any | None = None,
    ) -> Any: ...

    def init_tokenizer(
        self,
        model_name: str | None = None,
        tokenizer_type: str | None = None,
    ) -> Any: ...
