from types import ModuleType, SimpleNamespace

from definers.data.datasets.value import DatasetValueLoader
from definers.ui.apps.train import coach as train_coach


def test_load_table_values_ensures_openpyxl_runtime(monkeypatch):
    fake_pandas = ModuleType("pandas")
    captured = {}

    def _fake_read_excel(path):
        captured["path"] = path
        return SimpleNamespace(values=[[1, 2, 3]])

    fake_pandas.read_excel = _fake_read_excel
    monkeypatch.setitem(__import__("sys").modules, "pandas", fake_pandas)
    monkeypatch.setattr(
        "definers.optional_dependencies.ensure_module_runtime",
        lambda module_name: (
            captured.setdefault("module_name", module_name) or True
        ),
    )

    result = DatasetValueLoader.load_table_values("demo.xlsx", "xlsx")

    assert captured["module_name"] == "openpyxl"
    assert captured["path"] == "demo.xlsx"
    assert result == [[1, 2, 3]]


def test_train_coach_reads_xlsx_with_openpyxl_runtime(monkeypatch):
    fake_pandas = ModuleType("pandas")
    sentinel = object()
    captured = {}

    def _fake_read_excel(path):
        captured["path"] = path
        return sentinel

    fake_pandas.read_excel = _fake_read_excel
    monkeypatch.setitem(__import__("sys").modules, "pandas", fake_pandas)
    monkeypatch.setattr(
        "definers.optional_dependencies.ensure_module_runtime",
        lambda module_name: (
            captured.setdefault("module_name", module_name) or True
        ),
    )

    result = train_coach._read_tabular_dataframe("demo.xlsx")

    assert captured["module_name"] == "openpyxl"
    assert captured["path"] == "demo.xlsx"
    assert result is sentinel
