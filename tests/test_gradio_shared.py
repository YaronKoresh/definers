import sys
import time
from types import ModuleType

from definers.system.download_activity import report_download_activity
from definers.ui.gradio_shared import _progress_markup, wrap_progress_handler


def test_progress_markup_idle_starts_empty():
    markup = _progress_markup("Workspace ready")

    assert 'style="width: 0.00%"' in markup
    assert "Waiting to start" in markup
    assert "Start a workflow to load steps" in markup


def test_progress_markup_tracks_done_current_and_remaining_steps():
    markup = _progress_markup(
        "Master Audio",
        "running",
        "Running the mastering engine.",
        steps=(
            "Validate source",
            "Configure mastering",
            "Run mastering engine",
            "Load reports and stems",
            "Publish output",
        ),
        active_step=3,
    )

    assert "3/5" in markup
    assert "Validate source, Configure mastering" in markup
    assert "Run mastering engine" in markup
    assert "Load reports and stems, Publish output" in markup
    assert "definers-progress-shell__step--done" in markup
    assert "definers-progress-shell__step--active" in markup
    assert 'style="width: 40.00%"' in markup


def test_wrap_progress_handler_surfaces_publish_step(monkeypatch):
    fake_gradio = ModuleType("gradio")
    fake_gradio.update = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "gradio", fake_gradio)

    wrapped = wrap_progress_handler(
        lambda value: value.upper(),
        output_count=1,
        action_label="Generate Image",
        steps=(
            "Validate prompt",
            "Generate image",
            "Publish result",
        ),
        running_detail="Generating the image.",
        success_detail="Generated image is ready.",
    )

    updates = list(wrapped("cover.png"))

    assert len(updates) == 4
    assert "1/3" in updates[0][1]["value"]
    assert "2/3" in updates[1][1]["value"]
    assert "Publishing the result." in updates[2][1]["value"]
    assert "3/3" in updates[2][1]["value"]
    assert updates[2][0] == "COVER.PNG"
    assert "Generated image is ready." in updates[3][1]["value"]


def test_wrap_progress_handler_surfaces_runtime_download_activity(
    monkeypatch,
):
    fake_gradio = ModuleType("gradio")
    fake_gradio.update = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "gradio", fake_gradio)

    def handler(value: str) -> str:
        report_download_activity(
            "Phi-4",
            detail="Downloading answer model.",
            phase="download",
        )
        time.sleep(0.03)
        return value.upper()

    wrapped = wrap_progress_handler(
        handler,
        output_count=1,
        action_label="Generate Image",
        steps=(
            "Validate prompt",
            "Generate image",
            "Publish result",
        ),
        running_detail="Generating the image.",
        success_detail="Generated image is ready.",
        poll_interval_seconds=0.01,
    )

    updates = list(wrapped("cover.png"))
    progress_values = [update[1]["value"] for update in updates]

    assert any("Phi-4" in value for value in progress_values)
    assert any(
        "Downloading answer model." in value for value in progress_values
    )


def test_progress_markup_interpolates_runtime_activity_progress():
    markup = _progress_markup(
        "Master Audio",
        "running",
        "Downloading model.",
        steps=("Validate source", "Run mastering engine", "Publish output"),
        active_step=2,
        activity_completed=3,
        activity_total=4,
    )

    assert "2/3 · 3/4" in markup
    assert "Run mastering engine (3/4)" in markup
    assert 'style="width: 58.33%"' in markup
