class GuiProjectStarter:
    def __init__(self, namespace, on_missing, registry=None):
        self.namespace = namespace
        self.on_missing = on_missing
        self.registry = registry

    @staticmethod
    def get_gui_project_names(namespace, registry=None):
        from definers.ui.gui_registry import (
            normalize_gui_project_name,
        )

        if registry:
            return tuple(
                normalize_gui_project_name(str(project_name))
                for project_name in registry
            )
        return tuple(
            normalize_gui_project_name(name.removeprefix("_gui_"))
            for name, launcher in namespace.items()
            if name.startswith("_gui_") and callable(launcher)
        )

    def start(self, project):
        from definers.ui.gui_registry import (
            normalize_gui_project_name,
        )

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

    @classmethod
    def create_gui_project_starter(cls, namespace, on_missing, registry=None):
        return cls(
            namespace=namespace, on_missing=on_missing, registry=registry
        )

    @classmethod
    def start_project(cls, project, namespace, on_missing, registry=None):
        return cls.create_gui_project_starter(
            namespace,
            on_missing,
            registry=registry,
        ).start(project)

    @staticmethod
    def launch_installed_project(project):
        importer = import_module
        if importer is None:
            from importlib import import_module as importer

        return importer("definers.ui.gui_entrypoints").start(project)


get_gui_project_names = GuiProjectStarter.get_gui_project_names
create_gui_project_starter = GuiProjectStarter.create_gui_project_starter
start_project = GuiProjectStarter.start_project
launch_installed_project = GuiProjectStarter.launch_installed_project
import_module = None
