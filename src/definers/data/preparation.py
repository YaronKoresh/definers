class PreparationService:
    class TrainingData:
        def __init__(self, train, val=None, test=None, metadata=None):
            self.train = train
            self.val = val
            self.test = test
            self.metadata = metadata

    @staticmethod
    def prepare_data_cache_store():
        from collections import OrderedDict

        return OrderedDict()

    @staticmethod
    def catch(error: Exception) -> None:
        try:
            from definers.system import catch as runtime_catch

            runtime_catch(error)
        except Exception:
            return None

    @staticmethod
    def log(*args) -> None:
        try:
            from definers.system import log as runtime_log

            runtime_log(*args)
        except Exception:
            return None

    @staticmethod
    def runtime():
        import definers

        return definers

    @staticmethod
    def data_runtime():
        from types import SimpleNamespace

        import definers.data.arrays as arrays_module
        import definers.data.loaders as loaders_module
        import definers.data.runtime_patches as runtime_patches

        _, numpy_module = runtime_patches.init_cupy_numpy()
        return SimpleNamespace(
            load_source=loaders_module.load_source,
            drop_columns=loaders_module.drop_columns,
            three_dim_numpy=arrays_module.three_dim_numpy,
            cupy_to_numpy=arrays_module.cupy_to_numpy,
            two_dim_numpy=arrays_module.two_dim_numpy,
            _np=numpy_module,
        )

    @staticmethod
    def merge_columns(X, y=None):
        runtime = _runtime()
        if y is not None:
            tensor_dataset_cls = getattr(runtime, "TensorDataset", None)
            if tensor_dataset_cls is None:
                from definers.data.lightweight_datasets import (
                    resolve_tensor_dataset,
                )

                tensor_dataset_cls = resolve_tensor_dataset()
            return tensor_dataset_cls(X, y)
        return X

    @staticmethod
    def to_loader(dataset, batch_size=1):
        from definers.data.lightweight_datasets import (
            resolve_data_loader,
        )

        data_loader_cls = resolve_data_loader()

        return data_loader_cls(
            dataset,
            pin_memory=False,
            num_workers=0,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

    @staticmethod
    def make_loader(dataset, batch_size=1, sampler=None):
        from definers.data.lightweight_datasets import (
            resolve_data_loader,
        )

        data_loader_cls = resolve_data_loader()

        return data_loader_cls(
            dataset,
            pin_memory=False,
            num_workers=0,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=False,
        )

    @staticmethod
    def order_dataset(dataset, order_by=None):
        from definers.data.lightweight_datasets import (
            resolve_subset,
        )

        subset_cls = resolve_subset()

        if order_by is None or order_by == "shuffle":
            return dataset
        indices = list(range(len(dataset)))
        if callable(order_by):
            try:
                keys = [order_by(dataset[index]) for index in indices]
                sorted_idx = [index for _, index in sorted(zip(keys, indices))]
                return subset_cls(dataset, sorted_idx)
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
                return subset_cls(dataset, sorted_idx)
            except Exception:
                return dataset
        return dataset

    @classmethod
    def split_dataset(
        cls,
        dataset,
        stratify=None,
        val_frac=0.0,
        test_frac=0.0,
        random_state=None,
        batch_size=1,
    ):
        from definers.data.lightweight_datasets import (
            resolve_subset,
            split_indices,
        )

        subset_cls = resolve_subset()
        try:
            from sklearn.model_selection import (
                train_test_split as sklearn_train_test_split,
            )
        except Exception:
            sklearn_train_test_split = None

        indices = list(range(len(dataset)))
        labels = None
        if stratify is not None:
            if isinstance(stratify, str):
                if hasattr(dataset, "column_names"):
                    labels = list(dataset[stratify])
                elif isinstance(dataset, list):
                    try:
                        labels = [
                            item.get(stratify)
                            if isinstance(item, dict)
                            else None
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
                if sklearn_train_test_split is None:
                    raise RuntimeError
                train_idx, rest_idx = sklearn_train_test_split(
                    indices,
                    test_size=rest_frac,
                    stratify=labels,
                    random_state=random_state,
                )
            except Exception:
                train_idx, rest_idx = split_indices(
                    indices,
                    test_size=rest_frac,
                    random_state=random_state,
                    stratify=labels,
                )
            if val_frac > 0 and test_frac > 0 and rest_idx:
                try:
                    if sklearn_train_test_split is None:
                        raise RuntimeError
                    val_idx, test_idx = sklearn_train_test_split(
                        rest_idx,
                        test_size=test_frac / rest_frac,
                        random_state=random_state,
                    )
                except Exception:
                    val_idx, test_idx = split_indices(
                        rest_idx,
                        test_size=test_frac / rest_frac,
                        random_state=random_state,
                    )
            elif val_frac > 0:
                val_idx = rest_idx
                test_idx = []
            else:
                val_idx = []
                test_idx = rest_idx
        train_ds = subset_cls(dataset, train_idx)
        val_ds = subset_cls(dataset, val_idx) if val_idx else None
        test_ds = subset_cls(dataset, test_idx) if test_idx else None
        train_loader = cls.make_loader(train_ds, batch_size=batch_size)
        val_loader = (
            cls.make_loader(val_ds, batch_size=batch_size) if val_ds else None
        )
        test_loader = (
            cls.make_loader(test_ds, batch_size=batch_size) if test_ds else None
        )
        metadata = {
            "stratify": stratify,
            "val_frac": val_frac,
            "test_frac": test_frac,
        }
        return TrainingData(
            train=train_loader,
            val=val_loader,
            test=test_loader,
            metadata=metadata,
        )

    @staticmethod
    def callable_cache_token(value):
        code = getattr(value, "__code__", None)
        return (
            getattr(value, "__module__", type(value).__module__),
            getattr(value, "__qualname__", type(value).__qualname__),
            getattr(code, "co_filename", None),
            getattr(code, "co_firstlineno", None),
        )

    @classmethod
    def normalize_cache_value(cls, value):
        if callable(value):
            return ("callable",) + _callable_cache_token(value)
        if isinstance(value, dict):
            return tuple(
                (str(key), cls.normalize_cache_value(nested_value))
                for key, nested_value in sorted(
                    value.items(), key=lambda item: str(item[0])
                )
            )
        if isinstance(value, (list, tuple)):
            return tuple(cls.normalize_cache_value(item) for item in value)
        if isinstance(value, set):
            return tuple(
                sorted(cls.normalize_cache_value(item) for item in value)
            )
        if hasattr(value, "tolist"):
            try:
                return cls.normalize_cache_value(value.tolist())
            except Exception:
                return str(value)
        try:
            hash(value)
        except TypeError:
            return str(value)
        return value

    @classmethod
    def prepare_data_cache_entry(
        cls,
        remote_src,
        features,
        labels,
        url_type,
        revision,
        drop,
        order_by,
        stratify,
        val_frac,
        test_frac,
        batch_size,
    ):
        return {
            "remote_src": remote_src,
            "features": tuple(features) if features is not None else None,
            "labels": tuple(labels) if labels is not None else None,
            "url_type": url_type,
            "revision": revision,
            "drop": tuple(drop) if drop is not None else None,
            "order_by": (
                ":".join(str(part) for part in _callable_cache_token(order_by))
                if callable(order_by)
                else order_by
            ),
            "stratify": cls.normalize_cache_value(stratify),
            "val_frac": val_frac,
            "test_frac": test_frac,
            "batch_size": batch_size,
        }

    @classmethod
    def prepare_data_cache_key(cls, cache_entry):
        return tuple(
            (name, cls.normalize_cache_value(value))
            for name, value in cache_entry.items()
        )

    @staticmethod
    def cache_prepared_data(cache_key, cache_entry, value) -> None:
        if cache_key in _prepare_data_cache:
            _prepare_data_cache.move_to_end(cache_key)
        _prepare_data_cache[cache_key] = (dict(cache_entry), value)
        while len(_prepare_data_cache) > _PREPARE_DATA_CACHE_MAX_SIZE:
            _prepare_data_cache.popitem(last=False)

    @staticmethod
    def clear_prepare_data_cache() -> int:
        cached_entries = len(_prepare_data_cache)
        _prepare_data_cache.clear()
        return cached_entries

    @staticmethod
    def prepare_data_cache_manifest() -> list[dict[str, object]]:
        return [
            dict(cache_entry) for cache_entry, _ in _prepare_data_cache.values()
        ]

    @classmethod
    def prepare_data(
        cls,
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
        cache_entry = _prepare_data_cache_entry(
            remote_src,
            features,
            labels,
            url_type,
            revision,
            drop,
            order_by,
            stratify,
            val_frac,
            test_frac,
            batch_size,
        )
        cache_key = _prepare_data_cache_key(cache_entry)
        cached_value = _prepare_data_cache.get(cache_key)
        if cached_value is not None:
            _prepare_data_cache.move_to_end(cache_key)
            return cached_value[1]
        dataset = data_module.load_source(
            remote_src,
            features,
            labels,
            url_type,
            revision,
        )
        if dataset is None:
            return None
        if drop:
            dataset = data_module.drop_columns(dataset, drop)
        if order_by:
            dataset = cls.order_dataset(dataset, order_by)
        if val_frac > 0 or test_frac > 0 or stratify is not None:
            result = cls.split_dataset(
                dataset,
                stratify=stratify,
                val_frac=val_frac,
                test_frac=test_frac,
                batch_size=batch_size,
            )
        else:
            loader = cls.make_loader(dataset, batch_size=batch_size)
            result = TrainingData(train=loader, metadata={})
        _cache_prepared_data(cache_key, cache_entry, result)
        return result

    @staticmethod
    def pad_sequences(X):
        runtime = _data_runtime()
        try:
            import torch
        except Exception:
            torch = None
        if X is None or hasattr(X, "__len__") and len(X) == 0:
            if torch is None:
                return runtime._np.array([])
            return torch.tensor([])
        try:
            X = runtime.three_dim_numpy(X)
        except Exception:
            X = X
        X = runtime.cupy_to_numpy(X)
        if torch is None:
            sequences = [runtime._np.asarray(seq) for seq in X]
            if len(sequences) == 0:
                return runtime._np.array([])
            max_length = max(sequence.shape[0] for sequence in sequences)
            trailing_shape = sequences[0].shape[1:]
            padded = []
            for sequence in sequences:
                target = runtime._np.zeros(
                    (max_length,) + trailing_shape,
                    dtype=sequence.dtype,
                )
                target[: sequence.shape[0]] = sequence
                padded.append(target)
            return runtime._np.stack(padded, axis=0)
        sequences = [torch.as_tensor(seq) for seq in X]
        if len(sequences) == 0:
            return torch.tensor([])
        return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)

    @staticmethod
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

                preprocessing_module = importlib.import_module(
                    "cuml.preprocessing"
                )
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
            if (
                isinstance(current_np, runtime._np.ndarray)
                and current_np.ndim >= 2
            ):
                if current_np.shape[1] == 1:
                    rows.append(current_np.flatten())
                else:
                    rows.append(current_np[0])
            else:
                rows.append(runtime._np.array(current_np).reshape(-1))
        return runtime._np.array(rows)


_PREPARE_DATA_CACHE_MAX_SIZE = 32
_prepare_data_cache = PreparationService.prepare_data_cache_store()
_catch = PreparationService.catch
_log = PreparationService.log
_runtime = PreparationService.runtime
_data_runtime = PreparationService.data_runtime
_callable_cache_token = PreparationService.callable_cache_token
_normalize_cache_value = PreparationService.normalize_cache_value
_prepare_data_cache_entry = PreparationService.prepare_data_cache_entry
_prepare_data_cache_key = PreparationService.prepare_data_cache_key
_cache_prepared_data = PreparationService.cache_prepared_data

TrainingData = PreparationService.TrainingData
merge_columns = PreparationService.merge_columns
to_loader = PreparationService.to_loader
make_loader = PreparationService.make_loader
order_dataset = PreparationService.order_dataset
split_dataset = PreparationService.split_dataset
clear_prepare_data_cache = PreparationService.clear_prepare_data_cache
prepare_data_cache_manifest = PreparationService.prepare_data_cache_manifest
prepare_data = PreparationService.prepare_data
pad_sequences = PreparationService.pad_sequences
process_rows = PreparationService.process_rows
