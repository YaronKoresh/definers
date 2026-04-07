from __future__ import annotations

from definers.optional_dependencies import (
    install_optional_target,
    install_specs_for_target,
    optional_runtime_targets,
)


class CliInstallService:
    @staticmethod
    def format_available_targets() -> str:
        targets = optional_runtime_targets()
        return "\n".join(
            [
                "available install groups: " + ", ".join(targets["groups"]),
                "available install tasks: " + ", ".join(targets["tasks"]),
                "available install modules: " + ", ".join(targets["modules"]),
            ]
        )

    @classmethod
    def run_optional_install_command(
        cls,
        target: str,
        *,
        target_kind: str,
        list_only: bool,
        output,
    ) -> int:
        if list_only:
            output(cls.format_available_targets())
            return 0
        normalized_target = str(target or "").strip()
        if not normalized_target:
            output("install target is required unless --list is used")
            return 1
        specs = install_specs_for_target(normalized_target, kind=target_kind)
        if not specs:
            output(
                f"unknown {target_kind} target {normalized_target}; run 'definers install --list' to inspect available targets"
            )
            return 1
        if not install_optional_target(normalized_target, kind=target_kind):
            output(f"failed to install {target_kind} {normalized_target}")
            return 1
        output(
            f"installed {target_kind} {normalized_target}: " + ", ".join(specs)
        )
        return 0


format_available_targets = CliInstallService.format_available_targets
run_optional_install_command = CliInstallService.run_optional_install_command
