import importlib.util
from pathlib import Path
from types import ModuleType

PROJECT_PACKAGE_EXTRAS: dict[str, tuple[str, ...]] = {
    "audio": (),
    "animation": (),
    "chat": (),
    "faiss": (),
    "image": (),
    "train": (),
    "translate": (),
    "video": (),
}


def load_docker_app_module(file_path: Path) -> ModuleType:
    module_name = f"docker_app_{file_path.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_docker_app_package_specs_install_expected_extras():
    docker_root = Path(__file__).resolve().parents[1] / "docker"

    for project, expected_groups in PROJECT_PACKAGE_EXTRAS.items():
        module = load_docker_app_module(docker_root / project / "app.py")

        assert module.PROJECT == project
        assert module.PACKAGE_EXTRA_GROUPS == expected_groups
        assert (
            module.PACKAGE_SPEC
            == "definers @ git+https://github.com/YaronKoresh/definers.git"
        )
