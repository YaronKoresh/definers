from pathlib import Path

import tomllib
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


def read_pyproject_config() -> dict:
    return tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )


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

    assert "beautifulsoup4>=4.12.0" not in optional_dependencies_table["web"]
    assert "gradio-client>=2.3.0" not in optional_dependencies_table["web"]
    assert "hydra-core>=1.3.0" not in optional_dependencies_table["ml"]
    assert "torchvision>=0.16.0" in optional_dependencies_table["ml"]
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
