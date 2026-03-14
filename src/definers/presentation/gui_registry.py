from collections.abc import Callable, Mapping
from typing import Any


GuiLauncher = Callable[[], Any]


def normalize_gui_project_name(project_name: str) -> str:
    return project_name.strip().lower()


def _create_live_gui_launcher(
    namespace: Mapping[str, object],
    launcher_name: str,
) -> GuiLauncher:
    def launch() -> Any:
        launcher = namespace.get(launcher_name)
        if not callable(launcher):
            raise LookupError(f"No GUI launcher called {launcher_name}")
        return launcher()

    return launch


def register_gui_launchers(
    launchers: Mapping[str, object],
    namespace: Mapping[str, object] | None = None,
) -> dict[str, GuiLauncher]:
    registry: dict[str, GuiLauncher] = {}
    for project_name, launcher in launchers.items():
        normalized_project_name = normalize_gui_project_name(project_name)
        if not normalized_project_name:
            continue
        if callable(launcher):
            registry[normalized_project_name] = launcher
            continue
        if isinstance(launcher, str) and namespace is not None:
            registry[normalized_project_name] = _create_live_gui_launcher(
                namespace,
                launcher,
            )
    return registry