from definers.platform.services import get_infrastructure_services


def secure_command(command):
    return get_infrastructure_services().processes.secure_command(command)


def run_linux(command, silent: bool = False, env=None):
    return get_infrastructure_services().processes.run_linux(
        command,
        silent=silent,
        env=env,
    )


def run_windows(command, silent: bool = False, env=None):
    return get_infrastructure_services().processes.run_windows(
        command,
        silent=silent,
        env=env,
    )


def run(command, silent: bool = False, env=None):
    return get_infrastructure_services().processes.run(
        command,
        silent=silent,
        env=env,
    )


def get_process_pid(process_name: str) -> int | None:
    return get_infrastructure_services().processes.get_process_pid(process_name)


def send_signal_to_process(pid: int, signal_number: int) -> bool:
    return get_infrastructure_services().processes.send_signal_to_process(
        pid,
        signal_number,
    )


__all__ = (
    "get_process_pid",
    "run",
    "run_linux",
    "run_windows",
    "secure_command",
    "send_signal_to_process",
)
