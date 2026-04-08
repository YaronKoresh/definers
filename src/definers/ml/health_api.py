from definers.ml.health import (
    collect_live_ml_health_snapshot,
    render_ml_health_markdown,
    run_ml_health_check,
)


def get_ml_health_snapshot():
    return collect_live_ml_health_snapshot()


def validate_ml_health():
    return run_ml_health_check()


def ml_health_markdown():
    return render_ml_health_markdown(get_ml_health_snapshot())


__all__ = [
    "get_ml_health_snapshot",
    "ml_health_markdown",
    "validate_ml_health",
]
