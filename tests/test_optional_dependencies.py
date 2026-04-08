import builtins
import types

from definers import optional_dependencies


def test_package_specs_for_translate_task_include_runtime_modules():
    specs = optional_dependencies.package_specs_for_task("translate")

    assert "transformers>=4.36.0" in specs
    assert "langdetect>=1.0.9" in specs
    assert "sacremoses>=0.0.53" in specs


def test_package_specs_for_ml_group_cover_runtime_gap_packages():
    specs = optional_dependencies.package_specs_for_group("ml")

    assert "scikit-learn>=1.3.0" in specs
    assert "transformers>=4.36.0" in specs
    assert all("fairseq" not in spec for spec in specs)
    assert all("hydra-core" not in spec for spec in specs)


def test_runtime_specs_for_pypi_optional_modules_omit_vcs_links():
    expected_specs = {
        "audio_separator": "audio-separator>=0.30.2,<0.31.0",
        "basic_pitch": "basic-pitch>=0.4.0",
        "madmom": "madmom>=0.16.1",
        "refiners": "refiners>=0.4.0",
        "stable_whisper": "stable-ts>=2.19.1",
        "stopes": 'stopes>=2.2.1; sys_platform != "win32"',
    }

    for module_name, expected_spec in expected_specs.items():
        specs = optional_dependencies.package_specs_for_module(module_name)

        assert expected_spec in specs
        assert all("git+" not in spec for spec in specs)


def test_install_specs_for_madmom_use_pinned_github_commit():
    specs = optional_dependencies.install_specs_for_module("madmom")

    assert specs == (
        "madmom @ https://github.com/CPJKU/madmom/archive/27f032e8947204902c675e5e341a3faf5dc86dae.tar.gz",
    )


def test_install_specs_for_basic_pitch_use_pinned_github_commit():
    specs = optional_dependencies.install_specs_for_module("basic_pitch")

    assert specs == (
        "basic-pitch @ https://github.com/YaronKoresh/basic-pitch/archive/830590229b32e30faebf1626f046bb9d0b80def7.tar.gz",
    )


def test_install_specs_for_stopes_disable_unsupported_runtime():
    assert optional_dependencies.install_specs_for_module("stopes") == ()


def test_unsupported_fairseq_is_not_exposed_as_runtime_target():
    targets = optional_dependencies.optional_runtime_targets()

    assert optional_dependencies.package_specs_for_module("fairseq") == ()
    assert "fairseq" not in targets["modules"]


def test_install_optional_target_installs_group_specs(monkeypatch):
    installed = []

    result = optional_dependencies.install_optional_target(
        "web",
        kind="group",
        installer=lambda package_specs: installed.append(package_specs),
    )

    assert result is True
    assert installed == [
        (
            "fastapi>=0.100.0",
            "googledrivedownloader>=1.1.0",
            "gradio>=6.9.0",
            "lxml[html_clean]>=5.2.0",
            "cssselect>=1.2.0",
            "matplotlib>=3.7.0",
            "playwright>=1.40.0",
        )
    ]


def test_install_optional_target_uses_madmom_install_override():
    installed = []

    result = optional_dependencies.install_optional_target(
        "madmom",
        kind="module",
        installer=lambda package_specs: installed.append(package_specs),
    )

    assert result is True
    assert installed == [
        (
            "madmom @ https://github.com/CPJKU/madmom/archive/27f032e8947204902c675e5e341a3faf5dc86dae.tar.gz",
        )
    ]


def test_audio_group_install_includes_runtime_github_modules():
    audio_package_specs = optional_dependencies.package_specs_for_group("audio")
    audio_install_specs = optional_dependencies.install_specs_for_group("audio")

    assert "audio-separator>=0.30.2,<0.31.0" in audio_package_specs
    assert "basic-pitch>=0.4.0" not in audio_package_specs
    assert "madmom>=0.16.1" not in audio_package_specs
    assert (
        "basic-pitch @ https://github.com/YaronKoresh/basic-pitch/archive/830590229b32e30faebf1626f046bb9d0b80def7.tar.gz"
        in audio_install_specs
    )
    assert (
        "madmom @ https://github.com/CPJKU/madmom/archive/27f032e8947204902c675e5e341a3faf5dc86dae.tar.gz"
        in audio_install_specs
    )


def test_package_specs_for_tts_task_use_local_runtime_modules():
    specs = optional_dependencies.package_specs_for_task("tts")

    assert "transformers>=4.36.0" in specs
    assert "librosa>=0.10.0" in specs
    assert "pydub>=0.25.1" in specs
    assert "soundfile>=0.12.0" in specs


def test_install_specs_for_stopes_use_plain_spec_when_supported(
    monkeypatch,
):
    monkeypatch.setattr(
        optional_dependencies,
        "_supports_stopes_runtime_install",
        lambda: True,
    )

    assert optional_dependencies.install_specs_for_module("stopes") == (
        "stopes>=2.2.1",
    )


def test_runtime_specs_trim_redundant_web_and_ml_packages():
    assert optional_dependencies.package_specs_for_module("bs4") == ()
    assert optional_dependencies.package_specs_for_module("hydra") == ()

    gradio_specs = optional_dependencies.package_specs_for_module("gradio")

    assert gradio_specs == ("gradio>=6.9.0",)
    assert all("gradio-client" not in spec for spec in gradio_specs)


def test_optional_runtime_targets_list_groups_tasks_and_modules():
    targets = optional_dependencies.optional_runtime_targets()

    assert targets["groups"][-1] == "all"
    assert "audio" in targets["groups"]
    assert "tts" in targets["tasks"]
    assert "translate" in targets["tasks"]
    assert "gradio" in targets["modules"]


def test_runtime_groups_omit_vcs_links():
    for group_name in ("audio", "image", "nlp"):
        specs = optional_dependencies.package_specs_for_group(group_name)

        assert specs
        assert all("git+" not in spec for spec in specs)


def test_audio_group_leaves_madmom_as_explicit_module_install():
    audio_specs = optional_dependencies.package_specs_for_group("audio")
    madmom_specs = optional_dependencies.package_specs_for_module("madmom")

    assert "madmom>=0.16.1" not in audio_specs
    assert madmom_specs == ("madmom>=0.16.1",)


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
