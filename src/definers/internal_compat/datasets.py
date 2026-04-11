from __future__ import annotations

import csv
import json
from pathlib import Path


class Dataset:
    def __init__(self, data):
        columns = {key: list(values) for key, values in dict(data).items()}
        lengths = {len(values) for values in columns.values()}
        if len(lengths) > 1:
            raise ValueError("All dataset columns must have the same length")
        self._data = columns

    @classmethod
    def from_dict(cls, data):
        return cls(data)

    @property
    def column_names(self):
        return list(self._data)

    def remove_columns(self, column_names):
        names = set(column_names)
        return type(self)(
            {
                key: list(values)
                for key, values in self._data.items()
                if key not in names
            }
        )

    def to_dict(self):
        return {key: list(values) for key, values in self._data.items()}

    def __getitem__(self, key):
        if isinstance(key, str):
            return list(self._data[key])
        if isinstance(key, int):
            return {name: values[key] for name, values in self._data.items()}
        if isinstance(key, slice):
            return type(self)(
                {name: values[key] for name, values in self._data.items()}
            )
        raise TypeError("Dataset indices must be strings, ints, or slices")

    def __len__(self):
        if not self._data:
            return 0
        return len(next(iter(self._data.values())))


def _rows_to_columns(rows):
    columns = {}
    for row in rows:
        for key, value in row.items():
            columns.setdefault(key, []).append(value)
    return columns


def _load_json_dataset(path: Path):
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return Dataset.from_dict(payload)
    if isinstance(payload, list):
        return Dataset.from_dict(_rows_to_columns(payload))
    raise ValueError("Unsupported JSON dataset format")


def _load_csv_dataset(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return Dataset.from_dict(_rows_to_columns(rows))


def load_dataset(name, *, split=None, data_files=None, revision=None):
    _ = (split, revision)
    if not data_files:
        raise FileNotFoundError(f"Dataset {name} was not found")
    path_value = next(iter(dict(data_files).values()))
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file {path} was not found")
    loader_name = str(name).lower()
    if loader_name == "json" or path.suffix.lower() == ".json":
        return _load_json_dataset(path)
    if loader_name == "csv" or path.suffix.lower() == ".csv":
        return _load_csv_dataset(path)
    raise ValueError(f"Unsupported dataset format: {name}")


__all__ = ["Dataset", "load_dataset"]
