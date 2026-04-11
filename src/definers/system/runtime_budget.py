from __future__ import annotations

import os
from pathlib import Path


def detect_hosted_runtime(environ: dict[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    if any(
        str(env.get(name, "")).strip()
        for name in (
            "ZEROGPU",
            "ZERO_GPU",
            "SPACES_ZERO_GPU",
        )
    ):
        return "zerogpu"
    if Path("/data-nvme/zerogpu-offload").exists():
        return "zerogpu"
    if any(
        str(env.get(name, "")).strip()
        for name in (
            "SPACE_ID",
            "SPACE_HOST",
            "HF_SPACE_ID",
        )
    ):
        return "huggingface-spaces"
    return "local"


def hosted_preview_row_limit(runtime: str | None = None) -> int | None:
    resolved_runtime = detect_hosted_runtime() if runtime is None else runtime
    if resolved_runtime == "zerogpu":
        return 10001
    if resolved_runtime == "huggingface-spaces":
        return 50001
    return None


def hosted_guided_row_limit(runtime: str | None = None) -> int | None:
    resolved_runtime = detect_hosted_runtime() if runtime is None else runtime
    if resolved_runtime == "zerogpu":
        return 10000
    if resolved_runtime == "huggingface-spaces":
        return 50000
    return None


def hosted_guided_media_file_limit(runtime: str | None = None) -> int | None:
    resolved_runtime = detect_hosted_runtime() if runtime is None else runtime
    if resolved_runtime == "zerogpu":
        return 64
    if resolved_runtime == "huggingface-spaces":
        return 256
    return None


def estimate_session_retention_seconds(
    environ: dict[str, str] | None = None,
) -> float:
    env = os.environ if environ is None else environ
    configured_value = str(
        env.get("DEFINERS_GUI_SESSION_RETENTION_SECONDS", "")
    ).strip()
    if configured_value:
        try:
            return max(float(configured_value), 0.0)
        except Exception:
            return 86400.0
    runtime = detect_hosted_runtime(env)
    if runtime == "zerogpu":
        return 1800.0
    if runtime == "huggingface-spaces":
        return 7200.0
    return 86400.0


def should_cleanup_after_guided_training(runtime: str | None = None) -> bool:
    resolved_runtime = detect_hosted_runtime() if runtime is None else runtime
    return resolved_runtime in {"zerogpu", "huggingface-spaces"}


def hosted_runtime_label(runtime: str | None = None) -> str:
    resolved_runtime = detect_hosted_runtime() if runtime is None else runtime
    return {
        "local": "Local runtime",
        "huggingface-spaces": "Hugging Face Spaces",
        "zerogpu": "Hugging Face ZeroGPU",
    }.get(resolved_runtime, "Hosted runtime")


__all__ = (
    "detect_hosted_runtime",
    "estimate_session_retention_seconds",
    "hosted_guided_media_file_limit",
    "hosted_guided_row_limit",
    "hosted_preview_row_limit",
    "hosted_runtime_label",
    "should_cleanup_after_guided_training",
)
