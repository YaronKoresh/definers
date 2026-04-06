import subprocess
import sys

PROJECT = "image"
APT_PACKAGES = (
    "git",
    "gcc",
    "build-essential",
    "git-lfs",
    "wget",
    "curl",
    "libssl-dev",
)
PACKAGE_EXTRA_GROUPS: tuple[str, ...] = ()
PACKAGE_SPEC = "definers @ git+https://github.com/YaronKoresh/definers.git"


def run_command(*command: str) -> None:
    subprocess.check_call(list(command))


def install_runtime() -> None:
    run_command("apt-get", "update", "-y")
    run_command("apt-get", "install", "-y", *APT_PACKAGES)
    run_command(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--prefer-binary",
        "--no-cache-dir",
        PACKAGE_SPEC,
    )


def main() -> None:
    install_runtime()
    from definers.presentation.launchers import launch_installed_project

    launch_installed_project(PROJECT)


if __name__ == "__main__":
    main()
