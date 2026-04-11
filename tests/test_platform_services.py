import os
from unittest.mock import patch

import definers.command_runner as command_runner
import definers.system as system
import definers.system.processes as process_module
from definers.system.services import (
    EnvironmentService,
    FileSystemService,
    InfrastructureServices,
    ProcessService,
    reset_infrastructure_services,
    set_infrastructure_services,
)


def teardown_function() -> None:
    reset_infrastructure_services()


def test_command_runner_uses_configured_process_service():
    calls: list[tuple[object, bool, dict | None]] = []

    def fake_run(command, silent=False, env=None):
        calls.append((command, silent, env))
        return ["ok"]

    set_infrastructure_services(
        InfrastructureServices(processes=ProcessService(run_fn=fake_run))
    )

    result = command_runner.run(["echo", "hello"], silent=True, env={"A": "1"})

    assert result == ["ok"]
    assert calls == [(["echo", "hello"], True, {"A": "1"})]


def test_system_read_uses_configured_filesystem_service():
    calls: list[str] = []

    def fake_read(path: str):
        calls.append(path)
        return "payload"

    set_infrastructure_services(
        InfrastructureServices(filesystem=FileSystemService(read_fn=fake_read))
    )

    assert system.read("demo.txt") == "payload"
    assert calls == ["demo.txt"]


def test_system_permit_preserves_compatibility_injected_dependencies():
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_permit(path: str, **kwargs):
        calls.append((path, kwargs))
        return True

    set_infrastructure_services(
        InfrastructureServices(
            filesystem=FileSystemService(permit_fn=fake_permit)
        )
    )

    assert system.permit("demo.txt") is True
    assert calls
    path, kwargs = calls[0]
    assert path == "demo.txt"
    assert kwargs["exists_func"] is system.exist
    assert kwargs["get_os_name_func"] is system.get_os_name
    assert kwargs["subprocess_module"] is __import__("subprocess")


def test_system_get_os_name_uses_configured_environment_service():
    set_infrastructure_services(
        InfrastructureServices(
            environment=EnvironmentService(get_os_name_fn=lambda: "custom-os")
        )
    )

    assert system.get_os_name() == "custom-os"


def test_system_run_uses_configured_process_service():
    calls: list[tuple[object, bool, dict | None]] = []

    def fake_run(command, silent=False, env=None):
        calls.append((command, silent, env))
        return ["done"]

    set_infrastructure_services(
        InfrastructureServices(processes=ProcessService(run_fn=fake_run))
    )

    assert system.run(["echo", "hello"], silent=True, env={"A": "1"}) == [
        "done"
    ]
    assert calls == [(["echo", "hello"], True, {"A": "1"})]


def test_process_run_isolates_environment_overrides(monkeypatch):
    captured_env: dict[str, str] = {}

    class FakeProcess:
        def __init__(self, *args, **kwargs):
            nonlocal captured_env
            captured_env = dict(kwargs["env"])
            self.returncode = 0

        def communicate(self):
            return ("ok\n", "")

    monkeypatch.setenv("DEFINERS_PROCESS_KEEP", "present")
    monkeypatch.setenv("DEFINERS_PROCESS_REMOVE", "present")

    with patch.object(
        process_module, "secure_command", return_value=["echo", "ok"]
    ):
        with patch.object(process_module.subprocess, "Popen", FakeProcess):
            result = process_module.run(
                ["echo", "ok"],
                silent=True,
                env={
                    "DEFINERS_PROCESS_REMOVE": None,
                    "DEFINERS_PROCESS_NEW": 7,
                },
            )

    assert result == ["ok"]
    assert captured_env["DEFINERS_PROCESS_KEEP"] == "present"
    assert captured_env["DEFINERS_PROCESS_NEW"] == "7"
    assert "DEFINERS_PROCESS_REMOVE" not in captured_env
    assert os.environ["DEFINERS_PROCESS_REMOVE"] == "present"
