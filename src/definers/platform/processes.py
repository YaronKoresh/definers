import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence

from definers.platform.contracts import CommandInput, ProcessEnvironment
from definers.platform.paths import secure_path


def _normalize_absolute_path(path: str) -> str:
    return os.path.normcase(
        os.path.normpath(os.path.abspath(os.path.expanduser(path)))
    )


def _secure_executable(executable: str) -> str:
    if "/" not in executable and "\\" not in executable:
        return executable

    if not os.path.isabs(executable):
        return secure_path(executable)

    normalized_executable = _normalize_absolute_path(executable)
    normalized_python = _normalize_absolute_path(sys.executable)

    if normalized_executable == normalized_python:
        return normalized_executable

    resolved_from_path = shutil.which(os.path.basename(normalized_executable))
    if resolved_from_path is not None:
        normalized_resolved = _normalize_absolute_path(resolved_from_path)
        if normalized_executable == normalized_resolved:
            return normalized_executable

    return secure_path(executable)


def secure_command(command: CommandInput) -> list[str]:
    if isinstance(command, str):
        text_cmd = command.strip()
        if not text_cmd:
            raise ValueError("Command is empty.")
        if re.search(r"[;&|`$]", text_cmd):
            raise ValueError(
                f"Security Error: Unsafe characters in command: {text_cmd}"
            )
        try:
            cmd_list = shlex.split(text_cmd)
        except ValueError as error:
            raise ValueError(f"Invalid command syntax: {error}")
    elif isinstance(command, Sequence):
        cmd_list = [str(arg).strip() for arg in command if str(arg).strip()]
        if not cmd_list:
            raise ValueError("Command list is empty.")
        for arg in cmd_list:
            if len(arg) > 1024:
                raise ValueError(
                    f"Security Error: Argument too long: {arg[:50]}..."
                )
    else:
        raise TypeError("Command must be a string or a list.")

    cmd_list[0] = _secure_executable(cmd_list[0])
    return cmd_list


def _run_command(
    command: CommandInput,
    silent: bool = False,
    env: ProcessEnvironment = None,
):
    command_environment = dict(os.environ)
    if env is not None:
        for key, value in env.items():
            normalized_key = str(key)
            if not normalized_key:
                continue
            if value is None:
                command_environment.pop(normalized_key, None)
                continue
            command_environment[normalized_key] = str(value)
    try:
        args = secure_command(command)
    except ValueError as error:
        print("Error: Command rejected")
        try:
            from definers.file_ops import catch

            catch(error)
        except Exception:
            pass
        return False

    try:
        process = subprocess.Popen(
            args,
            shell=False,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=command_environment,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stdout, stderr = process.communicate()
        stdout = stdout or ""
        stderr = stderr or ""
        if not silent:
            if stdout:
                print(stdout, end="", flush=True)
            if stderr:
                print(stderr, end="", flush=True)
        if process.returncode != 0:
            if not silent:
                from definers.file_ops import log

                log(f"Script failed [{process.returncode}]", " ".join(args))
                log(f"Stderr: {stderr.strip()}", "")
            return False
        if not silent:
            from definers.file_ops import log

            log("Script completed", " ".join(args))
        return [
            line.strip() for line in stdout.strip().splitlines() if line.strip()
        ]
    except Exception as error:
        try:
            from definers.file_ops import catch

            catch(error)
        except Exception:
            pass
    return False


def _parse_pid_output(output: bytes | str) -> int | None:
    normalized_output = (
        output.decode("utf-8", errors="ignore")
        if isinstance(output, bytes)
        else str(output)
    )
    pid_tokens = [token for token in normalized_output.strip().split() if token]
    if len(pid_tokens) != 1:
        return None
    try:
        return int(pid_tokens[0])
    except ValueError:
        return None


def run_linux(
    command: CommandInput,
    silent: bool = False,
    env: ProcessEnvironment = None,
):
    return _run_command(command, silent=silent, env=env)


def run_windows(
    command: CommandInput,
    silent: bool = False,
    env: ProcessEnvironment = None,
):
    return _run_command(command, silent=silent, env=env)


def run(
    command: CommandInput,
    silent: bool = False,
    env: ProcessEnvironment = None,
):
    if sys.platform.startswith("win"):
        return run_windows(command, silent=silent, env=env)
    return run_linux(command, silent=silent, env=env)


def get_process_pid(process_name: str) -> int | None:
    try:
        output = subprocess.check_output(["pidof", process_name])
    except subprocess.CalledProcessError:
        return None
    return _parse_pid_output(output)


def send_signal_to_process(pid: int, signal_number: int) -> bool:
    try:
        os.kill(pid, signal_number)
        return True
    except OSError as error:
        print(f"Error sending signal: {error}")
        return False
