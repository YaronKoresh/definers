class GuiLauncherRegistry:
    @staticmethod
    def normalize_gui_project_name(project_name):
        return project_name.strip().lower()

    @staticmethod
    def call_live_launcher(namespace, launcher_name):
        launcher = namespace.get(launcher_name)
        if not callable(launcher):
            raise LookupError(f"No GUI launcher called {launcher_name}")
        return launcher()

    @classmethod
    def register_gui_launchers(cls, launchers, namespace=None):
        from functools import partial

        registry = {}
        for project_name, launcher in launchers.items():
            normalized_project_name = cls.normalize_gui_project_name(
                project_name
            )
            if not normalized_project_name:
                continue
            if normalized_project_name in registry:
                raise ValueError(
                    f"Duplicate GUI launcher for project {normalized_project_name}"
                )
            if callable(launcher):
                if namespace is not None:
                    launcher_name = getattr(launcher, "__name__", "")
                    if launcher_name and callable(namespace.get(launcher_name)):
                        registry[normalized_project_name] = partial(
                            cls.call_live_launcher,
                            namespace,
                            launcher_name,
                        )
                        continue
                registry[normalized_project_name] = launcher
                continue
            if isinstance(launcher, str):
                if namespace is None:
                    raise ValueError(
                        f"GUI launcher {normalized_project_name} requires a namespace"
                    )
                registry[normalized_project_name] = partial(
                    cls.call_live_launcher,
                    namespace,
                    launcher,
                )
                continue
            raise ValueError(
                f"Unsupported GUI launcher for project {normalized_project_name}"
            )
        return registry


normalize_gui_project_name = GuiLauncherRegistry.normalize_gui_project_name
register_gui_launchers = GuiLauncherRegistry.register_gui_launchers
GuiLauncher = object
