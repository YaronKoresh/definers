from __future__ import annotations

from definers.model_installation import (
    install_model_target,
    model_runtime_targets,
)
from definers.optional_dependencies import (
    install_optional_target,
    install_specs_for_target,
    optional_runtime_targets,
    package_specs_for_target,
)


def format_available_targets() -> str:
    targets = optional_runtime_targets()
    model_targets = model_runtime_targets()
    return "\n".join(
        [
            "available install groups: " + ", ".join(targets["groups"]),
            "available install tasks: " + ", ".join(targets["tasks"]),
            "available install modules: " + ", ".join(targets["modules"]),
            "available model domains: "
            + ", ".join(model_targets["model-domains"]),
            "available model tasks: " + ", ".join(model_targets["model-tasks"]),
        ]
    )


def run_optional_install_command(
    target: str,
    *,
    target_kind: str,
    list_only: bool,
    output,
) -> int:
    if list_only:
        output(format_available_targets())
        return 0
    normalized_target = str(target or "").strip()
    if not normalized_target:
        output("install target is required unless --list is used")
        return 1
    if target_kind in {"model-domain", "model-task"}:
        if not install_model_target(normalized_target, kind=target_kind):
            output(
                f"unknown {target_kind} target {normalized_target}; run 'definers install --list' to inspect available targets"
            )
            return 1
        output(f"installed {target_kind} {normalized_target}")
        return 0
    package_specs = package_specs_for_target(
        normalized_target,
        kind=target_kind,
    )
    if not package_specs:
        output(
            f"unknown {target_kind} target {normalized_target}; run 'definers install --list' to inspect available targets"
        )
        return 1
    specs = install_specs_for_target(normalized_target, kind=target_kind)
    if not specs:
        output(
            f"{target_kind} target {normalized_target} is not installable on this Python/platform"
        )
        return 1
    if not install_optional_target(normalized_target, kind=target_kind):
        output(f"failed to install {target_kind} {normalized_target}")
        return 1
    output(f"installed {target_kind} {normalized_target}: " + ", ".join(specs))
    return 0
