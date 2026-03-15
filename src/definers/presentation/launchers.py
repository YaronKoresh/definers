from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from .gui_registry import (
    GuiLauncher,
    normalize_gui_project_name,
)


def get_gui_project_names(
    namespace: Mapping[str, object],
    registry: Mapping[str, GuiLauncher] | None = None,
) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    if registry is not None:
        names.extend(
            normalize_gui_project_name(str(project_name))
            for project_name in registry
        )
    names.extend(
        normalize_gui_project_name(name.removeprefix("_gui_"))
        for name, launcher in namespace.items()
        if name.startswith("_gui_") and callable(launcher)
    )
    return tuple(
        name for name in names if name and not (name in seen or seen.add(name))
    )


@dataclass(frozen=True, slots=True)
class GuiProjectStarter:
    namespace: Mapping[str, object]
    on_missing: Callable[[str], Any]
    registry: Mapping[str, GuiLauncher] | None = None

    def start(self, project: str) -> Any:
        normalized_project = normalize_gui_project_name(project)
        launcher = None
        if self.registry is not None:
            registered_launcher = self.registry.get(normalized_project)
            if callable(registered_launcher):
                launcher = registered_launcher
        if launcher is None:
            namespaced_launcher = self.namespace.get(
                f"_gui_{normalized_project}"
            )
            if callable(namespaced_launcher):
                launcher = namespaced_launcher
        if launcher is not None:
            return launcher()
        return self.on_missing(normalized_project)


def create_gui_project_starter(
    namespace: Mapping[str, object],
    on_missing: Callable[[str], Any],
    registry: Mapping[str, GuiLauncher] | None = None,
) -> GuiProjectStarter:
    return GuiProjectStarter(
        namespace=namespace,
        on_missing=on_missing,
        registry=registry,
    )


def start_project(
    project: str,
    namespace: Mapping[str, object],
    on_missing: Callable[[str], Any],
    registry: Mapping[str, GuiLauncher] | None = None,
) -> Any:
    return create_gui_project_starter(
        namespace,
        on_missing,
        registry=registry,
    ).start(project)


def launch_installed_project(project: str) -> Any:
    return import_module("definers.chat").start(project)
