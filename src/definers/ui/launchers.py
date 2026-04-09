def get_gui_project_names(namespace, registry=None):
    from definers.ui.gui_registry import normalize_gui_project_name

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


def start_project(project, namespace, on_missing, registry=None):
    from definers.ui.gui_registry import normalize_gui_project_name

    normalized_project = normalize_gui_project_name(project)
    launcher = None
    if registry is not None:
        registered_launcher = registry.get(normalized_project)
        if callable(registered_launcher):
            launcher = registered_launcher
    if launcher is None:
        namespaced_launcher = namespace.get(f"_gui_{normalized_project}")
        if callable(namespaced_launcher):
            launcher = namespaced_launcher
    if launcher is not None:
        return launcher()
    return on_missing(normalized_project)


def launch_installed_project(project):
    importer = import_module
    if importer is None:
        from importlib import import_module as importer

    return importer("definers.ui.gui_entrypoints").start(project)


import_module = None
