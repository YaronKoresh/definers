import os
import shutil
import subprocess
import sys
import time

PROJECT = "chat"
APT_PACKAGES = (
    "git",
    "gcc",
    "build-essential",
    "git-lfs",
    "wget",
    "curl",
    "libssl-dev",
    "libgl1",
    "libegl1",
    "libxkbcommon0",
    "libxkbcommon-x11-0",
    "libdbus-1-3",
    "libfontconfig1",
    "libxrender1",
    "libxext6",
    "libx11-6",
    "libx11-xcb1",
    "libxcb-cursor0",
    "libxcb-icccm4",
    "libxcb-image0",
    "libxcb-keysyms1",
    "libxcb-randr0",
    "libxcb-render-util0",
    "libxcb-shape0",
    "libxcb-sync1",
    "libxcb-xfixes0",
    "libxcb-xinerama0",
    "libxcb-xkb1",
    "xvfb",
    "x11vnc",
    "novnc",
    "websockify",
)
PACKAGE_EXTRA_GROUPS: tuple[str, ...] = ()
PACKAGE_SPEC = "definers @ git+https://github.com/YaronKoresh/definers.git"

DISPLAY = os.environ.get("DEFINERS_GUI_DISPLAY", ":99")
SCREEN_GEOMETRY = os.environ.get("DEFINERS_GUI_GEOMETRY", "1280x800x24")
NOVNC_PORT = int(os.environ.get("DEFINERS_GUI_PORT", "6080"))
VNC_PORT = int(os.environ.get("DEFINERS_VNC_PORT", "5900"))


def run_command(*command: str) -> None:
    subprocess.check_call(list(command))


def install_runtime() -> None:
    run_command("apt-get", "update", "-y")
    run_command(
        "apt-get",
        "install",
        "-y",
        "--no-install-recommends",
        *APT_PACKAGES,
    )
    run_command(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--prefer-binary",
        "--no-cache-dir",
        PACKAGE_SPEC,
    )


def start_virtual_display() -> None:
    os.environ["DISPLAY"] = DISPLAY
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("QT_X11_NO_MITSHM", "1")

    subprocess.Popen(
        [
            "Xvfb",
            DISPLAY,
            "-screen",
            "0",
            SCREEN_GEOMETRY,
            "-ac",
            "+extension",
            "GLX",
            "+render",
            "-noreset",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.5)

    subprocess.Popen(
        [
            "x11vnc",
            "-display",
            DISPLAY,
            "-nopw",
            "-forever",
            "-shared",
            "-rfbport",
            str(VNC_PORT),
            "-quiet",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    novnc_web = None
    for candidate in ("/usr/share/novnc", "/usr/share/webapps/novnc"):
        if os.path.isdir(candidate):
            novnc_web = candidate
            break
    if shutil.which("websockify") is not None:
        websockify_args = ["websockify"]
        if novnc_web:
            websockify_args.append(f"--web={novnc_web}")
        websockify_args += [str(NOVNC_PORT), f"localhost:{VNC_PORT}"]
        subprocess.Popen(
            websockify_args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    url_suffix = "/vnc.html" if novnc_web else ""
    print(
        f"[definers] {PROJECT} GUI ready at http://localhost:{NOVNC_PORT}{url_suffix}",
        flush=True,
    )


def main() -> None:
    install_runtime()
    start_virtual_display()
    run_command(
        sys.executable,
        "-m",
        "definers",
        "start",
        PROJECT,
    )


if __name__ == "__main__":
    main()
