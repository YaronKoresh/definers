from types import ModuleType

from definers.data.datasets.source import DatasetSourceLoader


def test_load_remote_dataset_uses_sampled_train_split(monkeypatch):
    fake_datasets = ModuleType("datasets")
    captured = {}

    def _fake_load_dataset(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "dataset"

    fake_datasets.load_dataset = _fake_load_dataset
    monkeypatch.setitem(__import__("sys").modules, "datasets", fake_datasets)

    result = DatasetSourceLoader.load_remote_dataset(
        "owner/dataset",
        "main",
        sample_rows=25,
    )

    assert result == "dataset"
    assert captured == {
        "args": ("owner/dataset",),
        "kwargs": {"revision": "main", "split": "train[:25]"},
    }


def test_load_remote_dataset_defaults_to_full_train_split(monkeypatch):
    fake_datasets = ModuleType("datasets")
    captured = {}

    def _fake_load_dataset(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "dataset"

    fake_datasets.load_dataset = _fake_load_dataset
    monkeypatch.setitem(__import__("sys").modules, "datasets", fake_datasets)

    result = DatasetSourceLoader.load_remote_dataset("owner/dataset", None)

    assert result == "dataset"
    assert captured == {
        "args": ("owner/dataset",),
        "kwargs": {"split": "train"},
    }


def test_load_remote_dataset_fallback_uses_sampled_train_split(monkeypatch):
    fake_datasets = ModuleType("datasets")
    captured = {}

    def _fake_load_dataset(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "dataset"

    fake_datasets.load_dataset = _fake_load_dataset
    monkeypatch.setitem(__import__("sys").modules, "datasets", fake_datasets)

    result = DatasetSourceLoader.load_remote_dataset_fallback(
        "https://example.test/data.jsonl",
        "json",
        None,
        sample_rows=12,
    )

    assert result == "dataset"
    assert captured == {
        "args": ("json",),
        "kwargs": {
            "data_files": {"train": "https://example.test/data.jsonl"},
            "split": "train[:12]",
        },
    }
