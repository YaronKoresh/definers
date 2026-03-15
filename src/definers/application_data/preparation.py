from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TrainingData:
    train: Any
    val: Any | None = None
    test: Any | None = None
    metadata: dict[str, Any] | None = None


def _catch(error: Exception) -> None:
    try:
        from definers.system import catch as runtime_catch

        runtime_catch(error)
    except Exception:
        return None


def _log(*args) -> None:
    try:
        from definers.system import log as runtime_log

        runtime_log(*args)
    except Exception:
        return None


def _runtime():
    import definers

    return definers


def _data_runtime():
    import definers.data as data_module

    return data_module


def merge_columns(X, y=None):
    runtime = _runtime()
    if y is not None:
        tensor_dataset_cls = getattr(runtime, "TensorDataset", None)
        if tensor_dataset_cls is None:
            from torch.utils.data import TensorDataset

            tensor_dataset_cls = TensorDataset
        return tensor_dataset_cls(X, y)
    return X


def to_loader(dataset, batch_size=1):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        pin_memory=False,
        num_workers=0,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )


def make_loader(dataset, batch_size=1, sampler=None):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        pin_memory=False,
        num_workers=0,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=False,
    )


def order_dataset(dataset, order_by=None):
    from torch.utils.data import Subset

    if order_by is None or order_by == "shuffle":
        return dataset
    indices = list(range(len(dataset)))
    if callable(order_by):
        try:
            keys = [order_by(dataset[index]) for index in indices]
            sorted_idx = [index for _, index in sorted(zip(keys, indices))]
            return Subset(dataset, sorted_idx)
        except Exception:
            return dataset
    if isinstance(order_by, str):
        try:
            keys = []
            for index in indices:
                item = dataset[index]
                if isinstance(item, dict) and order_by in item:
                    keys.append(item[order_by])
                elif (
                    hasattr(item, "__len__")
                    and isinstance(item, (list, tuple))
                    and len(item) > 0
                ):
                    try:
                        keys.append(
                            item[0] if order_by == 0 else item[order_by]
                        )
                    except Exception:
                        keys.append(0)
                else:
                    keys.append(0)
            sorted_idx = [index for _, index in sorted(zip(keys, indices))]
            return Subset(dataset, sorted_idx)
        except Exception:
            return dataset
    return dataset


def split_dataset(
    dataset,
    stratify=None,
    val_frac=0.0,
    test_frac=0.0,
    random_state=None,
    batch_size=1,
):
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset

    indices = list(range(len(dataset)))
    labels = None
    if stratify is not None:
        if isinstance(stratify, str):
            if hasattr(dataset, "column_names"):
                labels = list(dataset[stratify])
            elif isinstance(dataset, list):
                try:
                    labels = [
                        item.get(stratify) if isinstance(item, dict) else None
                        for item in dataset
                    ]
                except Exception:
                    labels = None
        else:
            labels = stratify
    rest_frac = val_frac + test_frac
    if rest_frac <= 0:
        train_idx = indices
        val_idx = []
        test_idx = []
    else:
        try:
            train_idx, rest_idx = train_test_split(
                indices,
                test_size=rest_frac,
                stratify=labels,
                random_state=random_state,
            )
        except Exception:
            train_idx = indices
            rest_idx = []
        if val_frac > 0 and test_frac > 0 and rest_idx:
            try:
                val_idx, test_idx = train_test_split(
                    rest_idx,
                    test_size=test_frac / rest_frac,
                    random_state=random_state,
                )
            except Exception:
                val_idx = rest_idx
                test_idx = []
        elif val_frac > 0:
            val_idx = rest_idx
            test_idx = []
        else:
            val_idx = []
            test_idx = rest_idx

    def subset_from(idx):
        return Subset(dataset, idx)

    train_ds = subset_from(train_idx)
    val_ds = subset_from(val_idx) if val_idx else None
    test_ds = subset_from(test_idx) if test_idx else None
    train_loader = make_loader(train_ds, batch_size=batch_size)
    val_loader = make_loader(val_ds, batch_size=batch_size) if val_ds else None
    test_loader = (
        make_loader(test_ds, batch_size=batch_size) if test_ds else None
    )
    metadata = {
        "stratify": stratify,
        "val_frac": val_frac,
        "test_frac": test_frac,
    }
    return TrainingData(
        train=train_loader, val=val_loader, test=test_loader, metadata=metadata
    )


_prepare_data_cache: dict[str, TrainingData] = {}


def prepare_data(
    remote_src=None,
    features=None,
    labels=None,
    url_type=None,
    revision=None,
    drop=None,
    order_by=None,
    stratify=None,
    val_frac=0.0,
    test_frac=0.0,
    batch_size=1,
):
    data_module = _data_runtime()

    def make_key():
        parts = []
        for name, val in [
            ("remote_src", remote_src),
            ("features", tuple(features) if features is not None else None),
            ("labels", tuple(labels) if labels is not None else None),
            ("url_type", url_type),
            ("revision", revision),
            ("drop", tuple(drop) if drop is not None else None),
            ("order_by", id(order_by) if callable(order_by) else order_by),
            ("stratify", stratify),
            ("val_frac", val_frac),
            ("test_frac", test_frac),
            ("batch_size", batch_size),
        ]:
            parts.append(f"{name}={val}")
        return "|".join(map(str, parts))

    cache_key = make_key()
    if cache_key in _prepare_data_cache:
        return _prepare_data_cache[cache_key]
    dataset = data_module.load_source(
        remote_src, features, labels, url_type, revision
    )
    if dataset is None:
        return None
    if drop:
        dataset = data_module.drop_columns(dataset, drop)
    if order_by:
        dataset = order_dataset(dataset, order_by)
    if val_frac > 0 or test_frac > 0 or stratify is not None:
        result = split_dataset(
            dataset,
            stratify=stratify,
            val_frac=val_frac,
            test_frac=test_frac,
            batch_size=batch_size,
        )
    else:
        loader = make_loader(dataset, batch_size=batch_size)
        result = TrainingData(train=loader, metadata={})
    _prepare_data_cache[cache_key] = result
    return result


def pad_sequences(X):
    import torch

    runtime = _data_runtime()
    if X is None or hasattr(X, "__len__") and len(X) == 0:
        return torch.tensor([])
    try:
        X = runtime.three_dim_numpy(X)
    except Exception:
        X = X
    X = runtime.cupy_to_numpy(X)
    sequences = [torch.as_tensor(seq) for seq in X]
    if len(sequences) == 0:
        return torch.tensor([])
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)


def process_rows(batch):
    runtime = _data_runtime()
    if not batch:
        return runtime._np.empty((0, 0))
    scaler_cls = getattr(runtime, "StandardScaler", None)
    normalizer_cls = getattr(runtime, "Normalizer", None)
    imputer_cls = getattr(runtime, "SimpleImputer", None)
    if scaler_cls is None or normalizer_cls is None or imputer_cls is None:
        try:
            import importlib

            preprocessing_module = importlib.import_module("cuml.preprocessing")

            scaler_cls = preprocessing_module.StandardScaler
            normalizer_cls = preprocessing_module.Normalizer
            imputer_cls = preprocessing_module.SimpleImputer
        except Exception as error:
            _catch(error)
            _log("Falling back to sklearn (CPU)")
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import Normalizer, StandardScaler

            scaler_cls = StandardScaler
            normalizer_cls = Normalizer
            imputer_cls = SimpleImputer
    rows = []
    for index, row in enumerate(batch):
        current = runtime.two_dim_numpy(row)
        _log(f"Scaling {index + 1}", current)
        scaler = scaler_cls()
        current = scaler.fit_transform(current)
        _log(f"Normalizing {index + 1}", current)
        normalizer = normalizer_cls()
        current = normalizer.fit_transform(current)
        _log(f"Imputing {index + 1}", current)
        imputer = imputer_cls()
        current = imputer.fit_transform(current)
        _log(f"Reshaping {index + 1}", current)
        current_np = runtime.cupy_to_numpy(current)
        if isinstance(current_np, runtime._np.ndarray) and current_np.ndim >= 2:
            if current_np.shape[1] == 1:
                rows.append(current_np.flatten())
            else:
                rows.append(current_np[0])
        else:
            rows.append(runtime._np.array(current_np).reshape(-1))
    return runtime._np.array(rows)
