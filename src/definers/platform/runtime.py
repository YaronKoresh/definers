import ctypes
import importlib.util
import platform
import re
import shutil
import subprocess
import sys


def get_os_name() -> str:
    return platform.system().lower()


def is_admin_windows() -> bool:
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except Exception:
        return False


def cores() -> int | None:
    return __import__("os").cpu_count()


def get_python_version() -> str | None:
    try:
        version_info = sys.version_info
        major = version_info.major
        minor = getattr(version_info, "minor", 0)
        micro = getattr(version_info, "micro", 0)
        return f"{major}.{minor}.{micro}"
    except Exception:
        return None


def importable(name: str) -> bool:
    if not isinstance(name, str):
        return False
    module_name = name.strip()
    if not module_name:
        return False
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def runnable(cmd: str) -> bool:
    if not isinstance(cmd, str):
        return False
    command_line = cmd.strip()
    if not command_line:
        return False
    try:
        command_parts = __import__("shlex").split(command_line, posix=False)
    except ValueError:
        command_parts = command_line.split()
    if not command_parts:
        return False
    command_name = command_parts[0].strip('"').strip("'")
    if not command_name:
        return False
    return shutil.which(command_name) is not None


def check_version_wildcard(version_spec, version_actual):
    import fnmatch

    if version_spec is None or version_actual is None:
        return version_spec == version_actual
    return fnmatch.fnmatchcase(version_actual, version_spec)


def _normalize_name(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _version_matches(version_spec: str | None, version_actual: str | None) -> bool:
    if version_spec is None:
        return True
    if not version_actual:
        return False
    return version_actual.startswith(version_spec) or (
        "*" in version_spec and check_version_wildcard(version_spec, version_actual)
    )


def _parse_pip_list_line(line: str) -> tuple[str, str] | None:
    parts = re.split(r"\s{2,}", line.strip())
    if len(parts) != 2:
        return None
    return _normalize_name(parts[0]), _normalize_name(parts[1])


def installed(pack: str, version: str | None = None) -> bool:
    pack_lower = _normalize_name(pack)
    if not pack_lower:
        return False
    version_lower = _normalize_name(version) or None
    system_name = get_os_name()

    if system_name == "windows":
        command = "powershell.exe -Command \"Get-ItemProperty HKLM:\\Software\\Wow6432Node\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*, HKLM:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*, HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\* | Select-Object DisplayName, DisplayVersion | Format-Table -HideTableHeaders\""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=True,
            )
            for line in result.stdout.splitlines():
                parts = re.split(r"\s{2,}", line.strip())
                if not parts or not parts[0]:
                    continue
                name = _normalize_name(parts[0])
                current_version = _normalize_name(parts[1] if len(parts) > 1 else "")
                if pack_lower in name and _version_matches(version_lower, current_version):
                    return True
        except Exception:
            pass
    elif system_name == "linux":
        which_result = shutil.which(pack)
        if which_result:
            if version_lower is None:
                return True
            try:
                result = subprocess.run(
                    [pack, "--version"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                lines = result.stdout.splitlines()
                if not lines:
                    result = subprocess.run(
                        [pack, "-v"],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    lines = result.stdout.splitlines()
                if lines:
                    match = re.search(r"(\d+\.\d+(\.\d+)*)", lines[0])
                    if match:
                        actual_version = _normalize_name(match.group(0))
                        if _version_matches(version_lower, actual_version):
                            return True
            except Exception:
                pass
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.splitlines():
            parsed_line = _parse_pip_list_line(line)
            if parsed_line is None:
                continue
            name, current_version = parsed_line
            if name == pack_lower and _version_matches(version_lower, current_version):
                return True
        return False
    except Exception:
        return False