import ast
import re
from pathlib import Path

from packaging.requirements import Requirement

from definers import optional_dependencies

PRIMARY_INSTALL_GROUPS = (
    "audio",
    "image",
    "video",
    "ml",
    "nlp",
    "web",
)

EXPECTED_BASE_DEPENDENCIES = {
    "numpy>=1.26.0,<3",
    "requests>=2.28.0",
    "joblib>=1.3.0",
    "pandas>=1.5.0",
    "pillow>=9.1.0",
    "scipy>=1.10.0,<2",
}

SECTION_PATTERN = re.compile(r"^\[(?P<name>[^\]]+)\]\s*$", re.MULTILINE)


def read_pyproject_text() -> str:
    return (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )


def read_toml_section(pyproject_text: str, section_name: str) -> str:
    section_matches = list(SECTION_PATTERN.finditer(pyproject_text))
    for index, match in enumerate(section_matches):
        if match.group("name") != section_name:
            continue
        section_end = (
            section_matches[index + 1].start()
            if index + 1 < len(section_matches)
            else len(pyproject_text)
        )
        return pyproject_text[match.end() : section_end]
    raise KeyError(section_name)


def parse_array_body(section_text: str, body_start: int) -> list[str]:
    array_lines = ["["]
    for line in section_text[body_start:].splitlines():
        stripped_line = line.strip()
        array_lines.append(stripped_line)
        if stripped_line == "]":
            return ast.literal_eval("\n".join(array_lines))
    raise ValueError("Unterminated TOML array")


def parse_array_assignment(section_text: str, key: str) -> list[str]:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*\[\s*$", re.MULTILINE)
    match = pattern.search(section_text)
    if match is None:
        raise KeyError(key)
    return parse_array_body(section_text, match.end())


def parse_string_assignment(section_text: str, key: str) -> str:
    pattern = re.compile(
        rf'^\s*{re.escape(key)}\s*=\s*(?P<value>"(?:[^"\\]|\\.)*")\s*$',
        re.MULTILINE,
    )
    match = pattern.search(section_text)
    if match is None:
        raise KeyError(key)
    return ast.literal_eval(match.group("value"))


def parse_optional_dependencies(section_text: str) -> dict[str, list[str]]:
    optional_dependencies: dict[str, list[str]] = {}
    pattern = re.compile(
        r"^\s*(?P<key>[A-Za-z0-9_-]+)\s*=\s*\[\s*$", re.MULTILINE
    )
    for match in pattern.finditer(section_text):
        optional_dependencies[match.group("key")] = parse_array_body(
            section_text, match.end()
        )
    return optional_dependencies


def read_pyproject_config() -> dict:
    pyproject_text = read_pyproject_text()
    return {
        "project": {
            "dependencies": parse_array_assignment(
                read_toml_section(pyproject_text, "project"),
                "dependencies",
            ),
            "optional-dependencies": parse_optional_dependencies(
                read_toml_section(
                    pyproject_text, "project.optional-dependencies"
                )
            ),
        },
        "tool": {
            "poe": {
                "tasks": {
                    "install-matrix-smoke": {
                        "cmd": parse_string_assignment(
                            read_toml_section(
                                pyproject_text,
                                "tool.poe.tasks.install-matrix-smoke",
                            ),
                            "cmd",
                        ),
                        "env": {
                            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": parse_string_assignment(
                                read_toml_section(
                                    pyproject_text,
                                    "tool.poe.tasks.install-matrix-smoke.env",
                                ),
                                "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
                            )
                        },
                    }
                }
            }
        },
    }


def normalized_requirement_name(spec: str) -> str:
    requirement = Requirement(spec)
    return requirement.name.lower().replace("_", "-")


def test_primary_install_matrix_groups_use_index_safe_requirements():
    optional_dependencies_table = read_pyproject_config()["project"][
        "optional-dependencies"
    ]

    for group_name in PRIMARY_INSTALL_GROUPS:
        requirement_specs = optional_dependencies_table[group_name]

        assert requirement_specs
        for spec in requirement_specs:
            assert Requirement(spec).url is None


def test_base_dependency_ranges_are_broadened_without_unused_runtime_tools():
    dependency_specs = set(read_pyproject_config()["project"]["dependencies"])

    assert dependency_specs == EXPECTED_BASE_DEPENDENCIES
    assert "setuptools<83" not in dependency_specs


def test_runtime_groups_are_covered_by_pyproject_install_matrix():
    optional_dependencies_table = read_pyproject_config()["project"][
        "optional-dependencies"
    ]

    for group_name in PRIMARY_INSTALL_GROUPS:
        runtime_requirement_names = {
            normalized_requirement_name(spec)
            for spec in optional_dependencies.package_specs_for_group(
                group_name
            )
        }
        pyproject_requirement_names = {
            normalized_requirement_name(spec)
            for spec in optional_dependencies_table[group_name]
        }

        assert runtime_requirement_names.issubset(pyproject_requirement_names)


def test_runtime_group_names_match_install_matrix_groups():
    assert optional_dependencies.group_target_names() == (
        *PRIMARY_INSTALL_GROUPS,
        "all",
    )


def test_optional_dependency_groups_omit_trimmed_packages():
    optional_dependencies_table = read_pyproject_config()["project"][
        "optional-dependencies"
    ]
    audio_requirement_names = {
        normalized_requirement_name(spec)
        for spec in optional_dependencies_table["audio"]
    }

    assert "audio-separator>=0.30.0" in optional_dependencies_table["audio"]
    assert (
        'stopes>=2.2.1; sys_platform != "win32"'
        in optional_dependencies_table["nlp"]
    )
    assert "basic-pitch" not in audio_requirement_names
    assert "beautifulsoup4>=4.12.0" not in optional_dependencies_table["web"]
    assert "gradio-client>=2.3.0" not in optional_dependencies_table["web"]
    assert "hydra-core>=1.3.0" not in optional_dependencies_table["ml"]
    assert "tensorflow>=2.15.0" not in optional_dependencies_table["ml"]
    assert "tf-keras>=2.15.0" not in optional_dependencies_table["ml"]
    assert "madmom" not in audio_requirement_names
    assert "transformers" not in audio_requirement_names
    assert "torchvision>=0.16.0" not in optional_dependencies_table["ml"]
    assert "cssselect>=1.2.0" not in optional_dependencies_table["dev"]
    assert "resampy>=0.4.2" in optional_dependencies_table["audio"]
    assert "resampy>=0.4.2,<0.5" not in optional_dependencies_table["audio"]


def test_install_matrix_smoke_task_exists():
    tasks = read_pyproject_config()["tool"]["poe"]["tasks"]

    assert tasks["install-matrix-smoke"]["cmd"] == (
        "python -m pytest tests/test_install_matrix_smoke.py tests/test_pyproject_config.py -q"
    )
    assert (
        tasks["install-matrix-smoke"]["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"]
        == "1"
    )
