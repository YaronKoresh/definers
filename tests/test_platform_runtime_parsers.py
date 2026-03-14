from __future__ import annotations

from definers.platform import processes, runtime


def test_parse_pid_output_accepts_single_pid_bytes() -> None:
    assert processes._parse_pid_output(b"12345\n") == 12345


def test_parse_pid_output_accepts_single_pid_string() -> None:
    assert processes._parse_pid_output("67890") == 67890


def test_parse_pid_output_rejects_multiple_tokens() -> None:
    assert processes._parse_pid_output("123 456") is None


def test_parse_pid_output_rejects_invalid_integer() -> None:
    assert processes._parse_pid_output("not-a-pid") is None


def test_parse_pip_list_line_returns_normalized_name_and_version() -> None:
    assert runtime._parse_pip_list_line("Requests  2.31.0") == ("requests", "2.31.0")


def test_parse_pip_list_line_rejects_unstructured_lines() -> None:
    assert runtime._parse_pip_list_line("Package Version Location") is None
    assert runtime._parse_pip_list_line("requests") is None