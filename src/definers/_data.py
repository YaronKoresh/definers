import importlib
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as _np

from definers._constants import TOKENIZERS, iio_formats, tasks
from definers._system import catch, delete, log, read, tmp

try:
    import cupy as np
except Exception:
    import numpy as np


@dataclass
class TrainingData:
    train: Any
    val: Any | None = None
    test: Any | None = None
    metadata: dict[str, Any] = None


def patch_cupy_numpy():
    import numpy as np
    from numpy.lib import recfunctions

    def _set_aliases(module, aliases):
        for alias, target in aliases.items():
            if alias not in getattr(module, "__dict__", {}):
                setattr(module, alias, target)

    type_aliases = {
        "intp": np.int_,
        "float": np.float64,
        "int": np.int64,
        "bool": np.bool_,
        "complex": np.complex128,
        "object": np.object_,
        "str": np.str_,
        "str_": np.str_,
        "string_": np.bytes_,
        "strings": np.bytes_,
        "unicode": np.str_,
        "inf": float("inf"),
        "Inf": float("inf"),
    }
    func_aliases = {
        "round_": np.round,
        "product": np.prod,
        "cumproduct": np.cumprod,
        "alltrue": np.all,
        "sometrue": np.any,
        "rank": np.ndim,
    }
    _set_aliases(np, type_aliases)
    _set_aliases(np, func_aliases)
    if "char" not in getattr(np, "__dict__", {}):
        import types

        setattr(np, "char", types.SimpleNamespace())
    char_funcs = {
        "encode": lambda s, encoding=None: bytes(s, encoding or "utf-8"),
        "decode": lambda b, encoding=None: b.decode(encoding or "utf-8"),
        "lower": lambda s: s.lower(),
        "upper": lambda s: s.upper(),
        "capitalize": lambda s: s.capitalize(),
        "casefold": lambda s: s.casefold(),
        "title": lambda s: s.title(),
        "swapcase": lambda s: s.swapcase(),
        "startswith": lambda s, prefix, *args: s.startswith(prefix, *args),
        "endswith": lambda s, suffix, *args: s.endswith(suffix, *args),
        "strip": lambda s, chars=None: (
            s.strip(chars) if chars is not None else s.strip()
        ),
        "lstrip": lambda s, chars=None: (
            s.lstrip(chars) if chars is not None else s.lstrip()
        ),
        "rstrip": lambda s, chars=None: (
            s.rstrip(chars) if chars is not None else s.rstrip()
        ),
        "replace": lambda s, old, new, count=-1: (
            s.replace(old, new, count) if count != -1 else s.replace(old, new)
        ),
        "split": lambda s, sep=None, maxsplit=-1: (
            s.split(sep, maxsplit) if sep is not None else s.split()
        ),
        "rsplit": lambda s, sep=None, maxsplit=-1: (
            s.rsplit(sep, maxsplit) if sep is not None else s.rsplit()
        ),
        "splitlines": lambda s, keepends=False: s.splitlines(keepends),
        "partition": lambda s, sep: s.partition(sep),
        "rpartition": lambda s, sep: s.rpartition(sep),
        "join": lambda sep, iterable: sep.join(iterable),
        "count": lambda s, sub, start=None, end=None: (
            s.count(sub, start, end)
            if start is not None and end is not None
            else s.count(sub, start)
            if start is not None
            else s.count(sub)
        ),
        "find": lambda s, sub, start=None, end=None: (
            s.find(sub, start, end)
            if start is not None and end is not None
            else s.find(sub, start)
            if start is not None
            else s.find(sub)
        ),
        "rfind": lambda s, sub, start=None, end=None: (
            s.rfind(sub, start, end)
            if start is not None and end is not None
            else s.rfind(sub, start)
            if start is not None
            else s.rfind(sub)
        ),
        "index": lambda s, sub, start=None, end=None: (
            s.index(sub, start, end)
            if start is not None and end is not None
            else s.index(sub, start)
            if start is not None
            else s.index(sub)
        ),
        "rindex": lambda s, sub, start=None, end=None: (
            s.rindex(sub, start, end)
            if start is not None and end is not None
            else s.rindex(sub, start)
            if start is not None
            else s.rindex(sub)
        ),
        "zfill": lambda s, width: s.zfill(width),
        "center": lambda s, width, fillchar=" ": s.center(width, fillchar),
        "ljust": lambda s, width, fillchar=" ": s.ljust(width, fillchar),
        "rjust": lambda s, width, fillchar=" ": s.rjust(width, fillchar),
        "isalpha": lambda s: s.isalpha(),
        "isalnum": lambda s: s.isalnum(),
        "isdigit": lambda s: s.isdigit(),
        "isdecimal": lambda s: s.isdecimal(),
        "isnumeric": lambda s: s.isnumeric(),
        "isspace": lambda s: s.isspace(),
        "islower": lambda s: s.islower(),
        "isupper": lambda s: s.isupper(),
        "istitle": lambda s: s.istitle(),
        "add": lambda a, b: a + b,
        "multiply": lambda a, n: a * n,
        "mod": lambda s, values: s % values,
        "string_": lambda s: str(s),
        "bytes_": lambda s: (
            bytes(s, "utf-8") if not isinstance(s, bytes) else s
        ),
        "equal": lambda a, b: a == b,
        "not_equal": lambda a, b: a != b,
        "greater": lambda a, b: a > b,
        "greater_equal": lambda a, b: a >= b,
        "less": lambda a, b: a < b,
        "less_equal": lambda a, b: a <= b,
    }
    for name, func in char_funcs.items():
        if not hasattr(np.char, name):
            setattr(np.char, name, func)
    if "asscalar" not in getattr(np, "__dict__", {}):
        np.asscalar = lambda a: a.item()
    if "rec" not in getattr(np, "__dict__", {}):

        class NumpyRec:
            @staticmethod
            def append_fields(base, names, data, dtypes=None):
                return recfunctions.append_fields(
                    base, names, data, dtypes=dtypes
                )

            @staticmethod
            def drop_fields(base, names):
                return recfunctions.drop_fields(base, names)

            @staticmethod
            def rename_fields(base, name_dict):
                return recfunctions.rename_fields(base, name_dict)

            @staticmethod
            def merge_arrays(arrays, fill_value=-1, flatten=False):
                return recfunctions.merge_arrays(
                    arrays, fill_value=fill_value, flatten=flatten
                )

        np.rec = NumpyRec()
    if "machar" not in getattr(np, "__dict__", {}):

        class MachAr:
            pass

        np.core.machar = MachAr
    if hasattr(np, "testing") and (not hasattr(np.testing, "Tester")):

        class Tester:
            def test(self, label="fast", _extra_argv=None):
                return True

        np.testing.Tester = Tester
    if "distutils" not in getattr(np, "__dict__", {}):

        class DummyDistutils:
            class MiscUtils:
                def get_info(self, *args, **kwargs):
                    return {}

        np.distutils = DummyDistutils()
    if "set_string_function" not in getattr(np, "__dict__", {}):
        np.set_string_function = lambda *args, **kwargs: None
    _original_finfo = np.finfo

    def patched_finfo(dtype):
        try:
            return _original_finfo(dtype)
        except TypeError:
            return np.iinfo(dtype)

    np.finfo = patched_finfo
    if "_no_nep50_warning" not in getattr(np, "__dict__", {}):

        def dummy_npwarn_decorator_factory():

            def npwarn_decorator(x):
                return x

            return npwarn_decorator

        np._no_nep50_warning = dummy_npwarn_decorator_factory
    try:
        import cupy

        return (cupy, np)
    except ImportError:
        return (np, np)


def _find_spec(mod_name):
    try:
        mod = importlib.import_module(mod_name)
        return mod.__spec__
    except:
        return None


def _init_cupy_numpy():
    import numpy as _numpy_module

    np_module = None
    cupy_in_sys = sys.modules.get("cupy")
    if cupy_in_sys is not None and importlib.util.find_spec("cupy"):
        np_module = cupy_in_sys
    if np_module is None:
        np_module = _numpy_module
    if "float" not in getattr(np_module, "__dict__", {}):
        np_module.float = np_module.float64
    if "int" not in getattr(np_module, "__dict__", {}):
        np_module.int = np_module.int64
    return (np_module, _numpy_module)


def patch_torch_proxy_mode():
    import torch
    from torch.fx.experimental import proxy_tensor

    def get_proxy_mode():
        pre_dispatch_mode = torch._ops._get_dispatch_mode_pre_dispatch(
            torch._C._TorchDispatchModeKey.PROXY
        )
        mode = torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY)
        assert pre_dispatch_mode is None or mode is None, (
            f"pre_dispatch_mode={pre_dispatch_mode}, mode={mode}"
        )
        return pre_dispatch_mode or mode

    proxy_tensor.get_proxy_mode = getattr(
        proxy_tensor, "get_proxy_mode", get_proxy_mode
    )


def fetch_dataset(src, url_type=None, revision=None):
    from datasets import load_dataset

    try:
        if revision:
            dataset = load_dataset(src, revision=revision, split="train")
        else:
            dataset = load_dataset(src, split="train")
    except FileNotFoundError:
        logging.error(f"Dataset {src} not found.")
        return None
    except ConnectionError:
        logging.error(f"Connection error while loading dataset {src}.")
        return None
    except Exception as e:
        logging.error(f"Error loading dataset {src}: {e}")
        if url_type:
            try:
                if revision:
                    dataset = load_dataset(
                        url_type,
                        data_files={"train": src},
                        revision=revision,
                        split="train",
                    )
                else:
                    dataset = load_dataset(
                        url_type, data_files={"train": src}, split="train"
                    )
            except FileNotFoundError:
                logging.error(
                    f"Dataset {url_type} with data_files {src} not found."
                )
                return None
            except ConnectionError:
                logging.error(
                    f"Connection error while loading dataset {url_type} with data_files {src}."
                )
                return None
            except Exception as e2:
                logging.error(
                    f"Error loading dataset {url_type} with data_files {src}: {e2}"
                )
                return None
        else:
            return None
    return dataset


def drop_columns(dataset, drop_list):
    import definers as _d

    if not _d.check_parameter(drop_list):
        return dataset
    columns_to_delete = [
        col for col in dataset.column_names if col in drop_list
    ]
    return dataset.remove_columns(columns_to_delete)


def select_columns(dataset, cols):
    import definers as _d

    if not _d.check_parameter(cols):
        return dataset
    all_cols = dataset.column_names
    cols_to_drop = [c for c in all_cols if c not in cols]
    return _d.drop_columns(dataset, cols_to_drop)


def select_rows(dataset, start_index, end_index):
    from datasets import Dataset

    subset_data = {}
    for column_name in dataset.column_names:
        column_data = dataset[column_name]
        subset_data[column_name] = column_data[start_index:end_index]
    subset = Dataset.from_dict(subset_data)
    return subset


def split_columns(data, labels, is_batch=False):
    import definers as _d

    if not _d.check_parameter(labels):
        (X, y) = data
        return (X, y)
    if is_batch:
        X_batch = []
        y_batch = []
        batch_size = 0
        for value in data.values():
            if isinstance(value, list) or isinstance(value, np.ndarray):
                batch_size = len(value)
                break
        if batch_size == 0:
            return ([], [])
        for i in range(batch_size):
            X = {}
            y = {}
            for key, value in data.items():
                if key not in labels:
                    X[key] = value[i]
                else:
                    y[key] = value[i]
            X_batch.append(X)
            y_batch.append(y)
        return (X_batch, y_batch)
    else:
        features = _d.drop_columns(data, labels)
        labels_data = _d.select_columns(data, labels)
        return (features, labels_data)


def tokenize_and_pad(rows, tokenizer=None):
    import numpy as np

    import definers as _d

    if not tokenizer:
        tokenizer = _d.init_tokenizer()
    features_list = []
    for row in rows:
        if isinstance(row, dict):
            features_strings = []
            for key, value in row.items():
                if isinstance(value, (list, np.ndarray)):
                    features_strings.extend(map(str, value))
                elif value is not None:
                    features_strings.append(str(value))
            features_list.append(" ".join(features_strings))
        elif isinstance(row, str):
            features_list.append(row)
        else:
            return rows
    tokenized_inputs = tokenizer(
        features_list, padding=True, truncation=True, return_tensors="pt"
    )
    return two_dim_numpy(tokenized_inputs["input_ids"])


def init_tokenizer(mod="google-bert/bert-base-multilingual-cased"):
    import definers as _d

    model_name = mod or "google-bert/bert-base-multilingual-cased"
    current_model = TOKENIZERS.get("general-tokenizer-model")
    if (not TOKENIZERS["general-tokenizer"]) or (current_model != model_name):
        TOKENIZERS["general-tokenizer"] = _d.AutoTokenizer.from_pretrained(
            model_name
        )
        TOKENIZERS["general-tokenizer-model"] = model_name
    return TOKENIZERS["general-tokenizer"]


def files_to_dataset(features_paths: list, labels_paths: list = None):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    import definers as _d

    features = []
    labels = []
    features_have_strings = False
    labels_have_strings = False
    try:
        for feature_path in features_paths:
            loaded = _d.load_as_numpy(feature_path, training=True)
            if loaded is None:
                print(f"Error loading feature file: {feature_path}")
                return None
            if isinstance(loaded, _np.ndarray) and (
                _np.issubdtype(loaded.dtype, _np.str_)
                or _np.issubdtype(loaded.dtype, _np.object_)
            ):
                features_have_strings = True
            elif isinstance(loaded, list):
                if any(
                    isinstance(l, _np.ndarray)
                    and (
                        _np.issubdtype(l.dtype, _np.str_)
                        or _np.issubdtype(l.dtype, _np.object_)
                    )
                    for l in loaded
                    if l is not None
                ):
                    features_have_strings = True
            if isinstance(loaded, list):
                features.extend(
                    [_d.cupy_to_numpy(l) for l in loaded if l is not None]
                )
            else:
                features.append(_d.cupy_to_numpy(loaded))
        if labels_paths:
            for label_path in labels_paths:
                loaded = _d.load_as_numpy(label_path, training=True)
                if loaded is None:
                    print(f"Error loading label file: {label_path}")
                    return None
                if isinstance(loaded, _np.ndarray) and (
                    _np.issubdtype(loaded.dtype, _np.str_)
                    or _np.issubdtype(loaded.dtype, _np.object_)
                ):
                    labels_have_strings = True
                elif isinstance(loaded, list):
                    if any(
                        isinstance(l, _np.ndarray)
                        and (
                            _np.issubdtype(l.dtype, _np.str_)
                            or _np.issubdtype(l.dtype, _np.object_)
                        )
                        for l in loaded
                        if l is not None
                    ):
                        labels_have_strings = True
                if isinstance(loaded, list):
                    labels.extend(
                        [_d.cupy_to_numpy(l) for l in loaded if l is not None]
                    )
                else:
                    labels.append(_d.cupy_to_numpy(loaded))
    except Exception as e:
        print(f"Error during data loading: {e}")
        return None
    if not features and (not labels):
        print("No valid data loaded.")
        return None
    tokenizer = None
    if features_have_strings:
        print("features_have_strings")
        if not tokenizer:
            tokenizer = _d.init_tokenizer()
        features_as_strings = []
        for f in features:
            if isinstance(f, _np.ndarray):
                features_as_strings.append(" ".join(f.astype(str).flatten()))
            else:
                features_as_strings.append(str(f))
        tokenized_features = tokenize_and_pad(features_as_strings, tokenizer)
        features = [_d.cupy_to_numpy(row) for row in tokenized_features]
    if labels_paths and labels_have_strings:
        print("labels_have_strings")
        if not tokenizer:
            tokenizer = _d.init_tokenizer()
        labels_as_strings = []
        for l in labels:
            if isinstance(l, _np.ndarray):
                labels_as_strings.append(" ".join(l.astype(str).flatten()))
            else:
                labels_as_strings.append(str(l))
        tokenized_labels = tokenize_and_pad(labels_as_strings, tokenizer)
        labels = [_d.cupy_to_numpy(row) for row in tokenized_labels]
    all_data = features + labels if labels else features
    if not all_data:
        return None
    max_lens = get_max_shapes(*all_data)
    try:
        features_tensor = convert_tensor_dtype(
            torch.stack(
                [
                    torch.tensor(reshape_numpy(f, lengths=max_lens))
                    for f in features
                ]
            )
        )
        if labels:
            labels_tensor = convert_tensor_dtype(
                torch.stack(
                    [
                        torch.tensor(reshape_numpy(l, lengths=max_lens))
                        for l in labels
                    ]
                )
            )
            dataset = TensorDataset(features_tensor, labels_tensor)
        else:
            dataset = TensorDataset(features_tensor)
        return dataset
    except Exception as e_tensor:
        _d.catch(f"Error creating tensor dataset: {type(e_tensor)}")
        _d.catch(e_tensor)
        return None


def merge_columns(X, y=None):
    import definers as _d

    if y is not None:
        tensor_dataset_cls = getattr(_d, "TensorDataset", None)
        if tensor_dataset_cls is None:
            from torch.utils.data import TensorDataset

            tensor_dataset_cls = TensorDataset
        return tensor_dataset_cls(X, y)
    return X


def to_loader(dataset, batch_size=1):
    from torch.utils.data import DataLoader, TensorDataset

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
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=False,
    )


def load_source(
    remote_src=None, features=None, labels=None, url_type=None, revision=None
):
    import definers as _d

    if remote_src:
        return _d.fetch_dataset(remote_src, url_type, revision)
    if features:
        return _d.files_to_dataset(features, labels)
    return None


def infer_data_type(array):
    try:
        return guess_numpy_type(array)
    except Exception:
        return None


def tokenize_or_vectorize(data, tokenizer=None):
    import numpy as np

    if data is None:
        return data
    try:
        arr = np.array(data)
        if arr.dtype.kind in ("U", "S", "O"):
            return tokenize_and_pad(arr.tolist(), tokenizer)
    except Exception:
        pass
    return data


def pad_or_reshape(arr_list):
    if not arr_list:
        return []
    max_lens = get_max_shapes(*arr_list)
    return [reshape_numpy(a, lengths=max_lens) for a in arr_list]


def order_dataset(dataset, order_by=None):
    from torch.utils.data import Subset

    if order_by is None or order_by == "shuffle":
        return dataset
    n = len(dataset)
    indices = list(range(n))
    if callable(order_by):
        try:
            keys = [order_by(dataset[i]) for i in indices]
            sorted_idx = [i for _, i in sorted(zip(keys, indices))]
            return Subset(dataset, sorted_idx)
        except Exception:
            return dataset
    if isinstance(order_by, str):
        try:
            keys = []
            for i in indices:
                item = dataset[i]
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
            sorted_idx = [i for _, i in sorted(zip(keys, indices))]
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

    n = len(dataset)
    indices = list(range(n))
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

    dataset = load_source(remote_src, features, labels, url_type, revision)
    if dataset is None:
        return None
    if drop:
        dataset = drop_columns(dataset, drop)
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

    import definers as _d

    if X is None or (hasattr(X, "__len__") and len(X) == 0):
        return torch.tensor([])
    try:
        X = _d.three_dim_numpy(X)
    except Exception:
        X = X
    X = _d.cupy_to_numpy(X)
    sequences = [torch.as_tensor(seq) for seq in X]
    if len(sequences) == 0:
        return torch.tensor([])
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)


def create_vectorizer(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(token_pattern="(?u)\\b\\w+\\b")
    vectorizer.fit(texts)
    return vectorizer


def vectorize(vectorizer, texts):
    if vectorizer is None or texts is None:
        return None
    if isinstance(texts, list) and len(texts) == 0:
        vocab_size = len(getattr(vectorizer, "vocabulary_", {}) or {})
        return np.empty((0, vocab_size))
    X_tfidf = vectorizer.transform(texts)
    return np.array(X_tfidf.toarray())


def unvectorize(vectorizer, vectorized_data):
    if vectorizer is None or vectorized_data is None:
        return None
    vocabulary = vectorizer.vocabulary_
    index_to_word = {v: k for (k, v) in vocabulary.items()}
    unvectorized_texts = []
    for row in vectorized_data:
        words = []
        for i, value in enumerate(row):
            if value > 0:
                if i in index_to_word:
                    words.append(index_to_word[i])
        unvectorized_texts.append(" ".join(words))
    return unvectorized_texts


def load_as_numpy(path, training=False):
    import imageio as iio
    import pandas
    from scipy.io import wavfile

    import definers

    try:
        parts = path.split(".")
        if len(parts) >= 2:
            last = parts[-1].strip().lower()
            if last in ["wav", "mp3"]:
                try:
                    tfm = definers.sox.Transformer()
                    tfm.rate(32000)
                    if training:
                        temp_name = tmp("wav")
                        tfm.build_file(path, temp_name)
                        temp_2 = tmp("mp3")
                        definers.remove_silence(temp_name, temp_2)
                        (dir, num) = definers.split_mp3(temp_2, 5)
                        files = read(dir)
                        if not files:
                            files = [temp_2]
                        x = []
                        for _f in files:
                            _x = numpy_to_cupy(
                                definers.extract_audio_features(_f)
                            )
                            x.append(_x)
                        delete(temp_name)
                        delete(temp_2)
                        delete(dir)
                    else:
                        temp_name = tmp("mp3")
                        tfm.build_file(path, temp_name)
                        x = numpy_to_cupy(
                            definers.extract_audio_features(temp_name)
                        )
                        delete(temp_name)
                    return x
                except Exception as e:
                    catch(e)
                    return None
            elif last in ["csv", "xlsx", "json"]:
                try:
                    if last == "csv":
                        df = pandas.read_csv(path)

                        if df.empty:
                            df = pandas.read_csv(path, header=None)
                    elif last == "xlsx":
                        df = pandas.read_excel(path)
                    elif last == "json":
                        df = pandas.read_json(path)
                    return df.values
                except Exception as e_data:
                    catch(e_data)
                    return None
            elif last == "txt":
                try:
                    txt = read(path)
                    return numpy_to_cupy(definers.extract_text_features(txt))
                except Exception as e_txt:
                    catch(e_txt)
                    return None
            elif last in iio_formats:
                try:
                    resized = definers.resize_image(path, 1024, 1024)
                    if isinstance(resized, tuple):
                        path_resized = resized[0]
                    else:
                        path_resized = resized
                    return numpy_to_cupy(
                        definers.extract_image_features(path_resized)
                    )
                except Exception as e_image:
                    catch(e_image)
                    return None
            else:
                try:
                    resized_video_file = definers.resize_video(path, 1024, 1024)
                    new_fps_video_file = definers.convert_video_fps(
                        resized_video_file, 24
                    )
                    return numpy_to_cupy(
                        definers.extract_video_features(new_fps_video_file)
                    )
                except Exception as e_video:
                    catch(e_video)
                    return None
        else:
            print(f"Invalid path format: {path}")
            return None
    except Exception as e_overall:
        catch(e_overall)
        return None


def read_as_numpy(path: str):
    import definers as _d

    return _d.load_as_numpy(path)


def get_prediction_file_extension(pred_type):
    if pred_type is None:
        return "data"
    pred_type_lower = pred_type.lower().strip()
    if pred_type_lower == "video":
        return "mp4"
    elif pred_type_lower == "image":
        return "png"
    elif pred_type_lower == "audio":
        return "wav"
    elif pred_type_lower == "text":
        return "txt"
    else:
        return "data"


def process_rows(batch):
    import definers as _d

    if not batch:
        return _np.empty((0, 0))
    scaler_cls = getattr(_d, "StandardScaler", None)
    normalizer_cls = getattr(_d, "Normalizer", None)
    imputer_cls = getattr(_d, "SimpleImputer", None)
    if scaler_cls is None or normalizer_cls is None or imputer_cls is None:
        try:
            from cuml.preprocessing import (
                Normalizer,
                SimpleImputer,
                StandardScaler,
            )

            scaler_cls = StandardScaler
            normalizer_cls = Normalizer
            imputer_cls = SimpleImputer
        except Exception as e:
            catch(e)
            print("Falling back to sklearn (CPU)")
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import Normalizer, StandardScaler

            scaler_cls = StandardScaler
            normalizer_cls = Normalizer
            imputer_cls = SimpleImputer
    lst = []
    for i, row in enumerate(batch):
        r = two_dim_numpy(row)
        log(f"Scaling {i + 1}", r)
        scaler = scaler_cls()
        r = scaler.fit_transform(r)
        log(f"Normalizing {i + 1}", r)
        normalizer = normalizer_cls()
        r = normalizer.fit_transform(r)
        log(f"Imputing {i + 1}", r)
        imputer = imputer_cls()
        r = imputer.fit_transform(r)
        log(f"Reshaping {i + 1}", r)
        r_np = cupy_to_numpy(r)
        if isinstance(r_np, _np.ndarray) and r_np.ndim >= 2:
            if r_np.shape[1] == 1:
                lst.append(r_np.flatten())
            else:
                lst.append(r_np[0])
        else:
            lst.append(_np.array(r_np).reshape(-1))
    return _np.array(lst)


def tensor_length(tensor):
    nums = list(tensor.size())
    ret = 1
    for num in nums:
        ret = ret * num
    return ret


def dtype(size=16, is_float=True):
    import torch

    if size == 16 and is_float and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if size == 16 and is_float:
        return torch.float16
    if size == 32 and is_float:
        return torch.float32
    if size == 32 and (not is_float):
        return torch.int
    if size == 16 and (not is_float):
        return torch.int16
    if size == 8 and (not is_float):
        return torch.int8


def str_to_numpy(txt):
    if isinstance(txt, tuple) or isinstance(txt, list):
        txt = "".join(txt)
    vec = create_vectorizer([txt])
    return numpy_to_cupy(vectorize(vec, [txt]))


def numpy_to_str(v):
    return " ".join(v.flatten().astype(str).tolist())


def one_dim_numpy(v):
    return numpy_to_cupy(two_dim_numpy(v).flatten())


def two_dim_numpy(v):
    import torch

    if "cupy" in str(type(v)):
        v = cupy_to_numpy(v)
    if isinstance(v, torch.Tensor):
        v = v.cpu().numpy()
    elif isinstance(v, str):
        v = str_to_numpy(v)
    elif isinstance(v, np.ndarray):
        if _np.issubdtype(v.dtype, _np.str_):
            v = numpy_to_str(v)
            v = str_to_numpy(v)
        elif not np.issubdtype(v.dtype, np.number):
            raise TypeError(f"CuPy array of dtype {v.dtype} is not supported.")
    elif isinstance(v, (list, tuple)):
        v = np.array(v)
    elif not np.issubdtype(type(v), _np.number):
        try:
            v = np.array(v).astype(float)
        except Exception as e:
            raise TypeError(f"Input of type {type(v)} is not supported: {e}")
    else:
        v = np.array([v])
    if v.ndim == 0:
        return numpy_to_cupy(v.reshape(1, 1))
    elif v.ndim == 1:
        return numpy_to_cupy(v.reshape(-1, 1))
    elif v.ndim == 2:
        return numpy_to_cupy(v)
    else:
        try:
            new_shape = (-1, v.shape[-1])
            return numpy_to_cupy(v.reshape(new_shape))
        except ValueError as e:
            raise ValueError(
                f"Cannot reshape array of shape {v.shape} to 2D: {e}"
            )


def three_dim_numpy(v):
    import torch

    if "cupy" in str(type(v)):
        v = cupy_to_numpy(v)
    if isinstance(v, torch.Tensor):
        v = v.cpu().numpy()
    elif isinstance(v, str):
        v = str_to_numpy(v)
    elif isinstance(v, np.ndarray):
        if _np.issubdtype(v.dtype, _np.str_):
            v = numpy_to_str(v)
            v = str_to_numpy(v)
        elif not np.issubdtype(v.dtype, np.number):
            raise TypeError(f"CuPy array of dtype {v.dtype} is not supported.")
    elif isinstance(v, (list, tuple)):
        v = np.array(v)
    elif not np.issubdtype(type(v), _np.number):
        try:
            v = np.array(v).astype(float)
        except Exception as e:
            raise TypeError(f"Input of type {type(v)} is not supported: {e}")
    else:
        v = np.array([v])
    if v.ndim <= 2:
        return numpy_to_cupy(v.reshape(-1, 1, 1))
    elif v.ndim == 3:
        return numpy_to_cupy(v)
    else:
        try:
            new_shape = (-1, v.shape[-2], v.shape[-1])
            return numpy_to_cupy(v.reshape(new_shape))
        except ValueError as e:
            raise ValueError(
                f"Cannot reshape array of shape {v.shape} to 3D: {e}"
            )


def numpy_to_list(np_arr):
    return np.concatenate(np_arr, axis=None).ravel().tolist()


def guess_numpy_sample_rate(
    audio_data,
    possible_sample_rates=None,
    window_type="hann",
    window_size=None,
    peak_prominence=0.01,
    peak_distance=10,
    frequency_threshold=0.05,
):
    from scipy import signal as scipy_signal

    audio_data = cupy_to_numpy(audio_data)
    if window_size is None:
        window_size = len(audio_data)
    window = scipy_signal.get_window(window_type, window_size)
    frequencies = _np.fft.fftfreq(window_size, d=1.0)
    spectrum = _np.abs(_np.fft.fft(audio_data[:window_size] * window))
    peak_indices = scipy_signal.find_peaks(
        spectrum, prominence=peak_prominence, distance=peak_distance
    )[0]
    dominant_frequencies = frequencies[peak_indices]
    if possible_sample_rates is None:
        possible_sample_rates = [22050, 44100, 48000, 88200, 96000, 192000]
    for sr in possible_sample_rates:
        nyquist_frequency = sr / 2
        for freq in dominant_frequencies:
            if (
                abs(freq) < nyquist_frequency
                and abs(freq - round(freq)) / nyquist_frequency
                < frequency_threshold
            ):
                return sr
    return None


def guess_numpy_type(data):
    numpy_to_list(data)
    mean = np.mean(data)
    std = np.std(data)
    ratio = std / mean if mean != 0 else float("inf")
    if data.shape and len(data.shape) > 3:
        return "video"
    elif data.shape and len(data.shape) > 2:
        return "image"
    elif str(data.dtype)[1] in ["U", "S"]:
        return "text"
    elif data.ndim > 1 or str(data.dtype)[1] in ["f"] or ratio > 1:
        return "audio"
    else:
        return "text"


def cupy_to_numpy(v: Any) -> Any:
    try:
        import cupy as cp

        return cp.asnumpy(v)
    except Exception:
        return v


def numpy_to_cupy(v: Any) -> Any:
    try:
        import cupy as cp

        return cp.array(v)
    except Exception:
        return v


def get_max_shapes(*data):
    lengths = []
    shapes = [np_arr.shape for np_arr in data]
    for sh in shapes:
        l = len(lengths)
        while l < len(sh):
            lengths.append(0)
            l = len(lengths)
        for i, dim in enumerate(sh):
            lengths[i] = max(lengths[i], dim)
    return lengths


def pad_nested(nested_data, lengths, fill_value=0):
    if isinstance(nested_data, _np.ndarray):
        nested_data = nested_data.tolist()
    elif isinstance(nested_data, tuple):
        nested_data = list(nested_data)
    if not nested_data:
        return [fill_value] * lengths[0]
    if not isinstance(nested_data[0], list):
        data_len = len(nested_data)
        diff = lengths[0] - data_len
        if diff > 0:
            nested_data.extend([fill_value] * diff)
        return nested_data
    ret = []
    for arr in nested_data:
        ret.append(pad_nested(arr, lengths[1:], fill_value))
    data_len = len(ret)
    diff = lengths[0] - data_len
    if diff > 0:
        ret.extend([pad_nested([], lengths[1:], fill_value)] * diff)
    return ret


def reshape_numpy(data, fill_value=0, lengths=None):
    if isinstance(data, _np.ndarray):
        data = data.tolist()
    if not data:
        return _np.array([])
    try:
        if lengths is None:
            lengths = get_max_shapes(data)
        log("Reshaping data", lengths)
        reshaped_data = pad_nested(data, lengths)
        log("Reshaped data", lengths)
        return _np.array(reshaped_data)
    except (TypeError, IndexError) as e:
        catch(e)
        return _np.array([])
    except Exception as e2:
        catch(e2)
        return _np.array([])


def convert_tensor_dtype(tensor):
    import torch

    if tensor.is_floating_point():
        if tensor.dtype == torch.float64:
            return tensor.to(torch.float32)
        else:
            return tensor
    elif not torch.is_floating_point(tensor):
        max_val = tensor.max()
        min_val = tensor.min()
        if min_val >= 0:
            if max_val <= 255:
                return tensor.to(torch.uint8)
            elif max_val <= 65535:
                return tensor.to(torch.uint16)
            elif max_val <= 4294967295:
                return tensor.to(torch.uint32)
            else:
                return tensor.to(torch.uint64)
        elif min_val >= -128 and max_val <= 127:
            return tensor.to(torch.int8)
        elif min_val >= -32768 and max_val <= 32767:
            return tensor.to(torch.int16)
        elif min_val >= -2147483648 and max_val <= 2147483647:
            return tensor.to(torch.int32)
        else:
            return tensor.to(torch.int64)
    else:
        return tensor


def is_gpu():
    import torch

    return torch.cuda.is_available()


def check_onnx(path):
    import onnx

    model = onnx.load(path)
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError:
        return False
    return True


def pytorch_to_onnx(model_torch, input_dim, onnx_path="model.onnx"):
    import torch

    dummy_input = torch.randn(1, input_dim).cuda()
    torch.onnx.export(model_torch, dummy_input, onnx_path, verbose=True)
    print("ONNX export complete!")
