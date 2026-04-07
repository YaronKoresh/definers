import builtins
import types

from definers import optional_dependencies


def test_package_specs_for_translate_task_include_runtime_modules():
    specs = optional_dependencies.package_specs_for_task("translate")

    assert "transformers>=4.36.0" in specs
    assert "langdetect>=1.0.9" in specs
    assert "sacremoses>=0.0.53" in specs


def test_auto_install_import_retries_known_module(monkeypatch):
    sentinel = object()
    calls = []

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        calls.append(name)
        if len(calls) == 1:
            raise ModuleNotFoundError("missing gradio", name="gradio")
        return sentinel

    installed = []

    monkeypatch.setattr(optional_dependencies, "_ORIGINAL_IMPORT", fake_import)
    monkeypatch.setattr(
        optional_dependencies,
        "ensure_module_runtime",
        lambda module_name, installer=None: (
            installed.append(module_name) or True
        ),
    )

    result = optional_dependencies._auto_install_import(
        "gradio",
        globals={"__name__": "definers.presentation.apps.chat_app"},
    )

    assert result is sentinel
    assert installed == ["gradio"]
    assert calls == ["gradio", "gradio"]


def test_auto_install_import_patch_does_not_corrupt_global_imports(
    monkeypatch,
):
    sentinel = object()
    calls = []

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        calls.append(name)
        if len(calls) == 1:
            raise ModuleNotFoundError("missing gradio", name="gradio")
        return sentinel

    installed = []

    monkeypatch.setattr(optional_dependencies, "_ORIGINAL_IMPORT", fake_import)
    monkeypatch.setattr(
        optional_dependencies,
        "ensure_module_runtime",
        lambda module_name, installer=None: (
            installed.append(module_name) or True
        ),
    )

    result = optional_dependencies._auto_install_import(
        "gradio",
        globals={"__name__": "definers.presentation.apps.chat_app"},
    )

    assert result is sentinel
    assert builtins.__import__("warnings").__name__ == "warnings"
    assert installed == ["gradio"]
    assert calls == ["gradio", "gradio"]
