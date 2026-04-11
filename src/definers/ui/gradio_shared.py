from html import escape
from typing import Any


def _progress_markup(
    title: str,
    state: str = "idle",
    detail: str | None = None,
    steps: tuple[str, ...] | list[str] | None = None,
    active_step: int | None = None,
    activity_completed: int | None = None,
    activity_total: int | None = None,
    bytes_downloaded: int | None = None,
    bytes_total: int | None = None,
) -> str:
    def _format_bytes(value: int | None) -> str | None:
        if value is None:
            return None
        try:
            normalized = max(int(value), 0)
        except Exception:
            return None
        suffixes = ("B", "KiB", "MiB", "GiB", "TiB")
        scaled = float(normalized)
        suffix = suffixes[0]
        for suffix in suffixes:
            if scaled < 1024.0 or suffix == suffixes[-1]:
                break
            scaled /= 1024.0
        if suffix == "B":
            return f"{int(scaled)} {suffix}"
        return f"{scaled:.1f} {suffix}"

    def _activity_fraction() -> float | None:
        if activity_total and activity_completed is not None:
            return max(
                0.0,
                min(float(activity_completed) / float(activity_total), 1.0),
            )
        if bytes_total and bytes_downloaded is not None:
            return max(
                0.0,
                min(float(bytes_downloaded) / float(bytes_total), 1.0),
            )
        return None

    normalized_state = str(state or "idle").strip().lower()
    detail_text = (
        "Choose an action to begin."
        if detail is None
        else str(detail).strip() or "Choose an action to begin."
    )
    state_label = {
        "idle": "Ready",
        "running": "Running",
        "success": "Done",
        "error": "Failed",
    }.get(normalized_state, "Ready")
    resolved_steps = tuple(
        str(step).strip() for step in tuple(steps or ()) if str(step).strip()
    )
    progress_percent = 0.0
    completed_steps: tuple[str, ...] = ()
    current_step_label: str | None = None
    remaining_steps: tuple[str, ...] = resolved_steps
    activity_fraction = _activity_fraction()
    if resolved_steps:
        if normalized_state == "success":
            progress_percent = 100.0
            completed_steps = resolved_steps
            remaining_steps = ()
        elif normalized_state == "idle":
            progress_percent = 0.0
        else:
            bounded_active_step = max(
                1,
                min(
                    int(active_step or 1),
                    len(resolved_steps),
                ),
            )
            completed_steps = resolved_steps[: bounded_active_step - 1]
            current_step_label = resolved_steps[bounded_active_step - 1]
            remaining_steps = resolved_steps[bounded_active_step:]
            resolved_fraction = (
                activity_fraction
                if activity_fraction is not None
                and normalized_state == "running"
                else 0.0
                if normalized_state == "running"
                else 1.0
            )
            progress_percent = (
                (len(completed_steps) + resolved_fraction) / len(resolved_steps)
            ) * 100.0
    elif normalized_state == "running" and activity_fraction is not None:
        progress_percent = activity_fraction * 100.0
    done_text = ", ".join(completed_steps) or "Nothing completed yet"
    current_text = current_step_label or (
        "Waiting to start"
        if normalized_state == "idle"
        else "Completed"
        if normalized_state == "success"
        else detail_text
    )
    if normalized_state == "running" and activity_total:
        completed_count = max(int(activity_completed or 0), 0)
        current_text = (
            f"{current_text} ({completed_count}/{int(activity_total)})"
        )
    elif normalized_state == "running" and bytes_total:
        downloaded_text = _format_bytes(bytes_downloaded)
        total_text = _format_bytes(bytes_total)
        if downloaded_text and total_text:
            current_text = f"{current_text} ({downloaded_text} / {total_text})"
    remaining_text = (
        ", ".join(remaining_steps)
        if remaining_steps
        else "Start a workflow to load steps"
        if not resolved_steps and normalized_state == "idle"
        else "Nothing left"
    )
    if normalized_state == "running" and activity_total:
        remaining_count = max(
            int(activity_total) - int(activity_completed or 0), 0
        )
        if remaining_count > 0 and remaining_steps:
            remaining_text = (
                f"{remaining_count} items left in the current batch, then "
                + ", ".join(remaining_steps)
            )
        elif remaining_count > 0:
            remaining_text = (
                f"{remaining_count} items left in the current batch"
            )
    step_count_markup = (
        f'<em class="definers-progress-shell__count">{int(active_step or 0)}/{len(resolved_steps)}'
        + (
            f" · {max(int(activity_completed or 0), 0)}/{int(activity_total)}"
            if resolved_steps
            and normalized_state == "running"
            and activity_total
            else ""
        )
        + "</em>"
        if resolved_steps and normalized_state == "running"
        else f'<em class="definers-progress-shell__count">{len(completed_steps)}/{len(resolved_steps)}</em>'
        if resolved_steps
        else ""
    )
    step_items_markup = ""
    if resolved_steps:
        step_items = []
        for index, step_label in enumerate(resolved_steps, start=1):
            step_state = "todo"
            if normalized_state == "success" or index <= len(completed_steps):
                step_state = "done"
            elif index == len(completed_steps) + 1:
                step_state = (
                    "error" if normalized_state == "error" else "active"
                )
            step_items.append(
                '<li class="definers-progress-shell__step '
                f'definers-progress-shell__step--{escape(step_state)}">'
                f"<span>{index}</span>"
                f"<strong>{escape(step_label)}</strong>"
                "</li>"
            )
        step_items_markup = (
            '<ol class="definers-progress-shell__steps">'
            + "".join(step_items)
            + "</ol>"
        )
    return (
        '<section class="definers-progress-shell '
        f'definers-progress-shell--{escape(normalized_state)}">'
        '<div class="definers-progress-shell__meta">'
        f"<span>{escape(state_label)}</span>"
        f"<strong>{escape(str(title or 'Workspace status'))}</strong>"
        f"{step_count_markup}"
        "</div>"
        f"<p>{escape(detail_text)}</p>"
        '<div class="definers-progress-shell__summary">'
        "<div><span>Done</span>"
        f"<strong>{escape(done_text)}</strong></div>"
        "<div><span>Now</span>"
        f"<strong>{escape(current_text)}</strong></div>"
        "<div><span>Left</span>"
        f"<strong>{escape(remaining_text)}</strong></div>"
        "</div>"
        '<div class="definers-progress-shell__rail">'
        f'<div class="definers-progress-shell__bar" style="width: {progress_percent:.2f}%"></div>'
        "</div>"
        f"{step_items_markup}"
        "</section>"
    )


def init_progress_tracker(
    title: str = "Workspace ready",
    detail: str = "Choose an action to begin.",
):
    import gradio as gr

    return gr.Markdown(
        value=_progress_markup(title, "idle", detail),
        elem_classes="definers-progress-panel",
    )


def _status_value_text(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def status_card_markdown(
    title: str,
    detail: str | None = None,
    items: tuple[tuple[str, object], ...] | list[tuple[str, object]] = (),
) -> str:
    normalized_title = str(title or "Status").strip() or "Status"
    normalized_detail = str(detail or "").strip() or "Ready."
    lines = [f"## {normalized_title}", normalized_detail]
    for label, value in items:
        normalized_label = str(label).strip()
        if not normalized_label:
            continue
        lines.append(f"- **{normalized_label}:** {_status_value_text(value)}")
    return "\n".join(lines)


def init_status_card(
    title: str = "Status",
    detail: str = "Ready.",
    items: tuple[tuple[str, object], ...] | list[tuple[str, object]] = (),
):
    import gradio as gr

    return gr.Markdown(
        value=status_card_markdown(title, detail, items),
        elem_classes="definers-status-card",
    )


def _output_folder_markup(
    path: str,
    detail: str,
    *,
    state: str = "ready",
) -> str:
    return (
        '<section class="definers-output-toolbar__card '
        f'definers-output-toolbar__card--{escape(state)}">'
        "<span>Outputs Folder</span>"
        f"<strong>{escape(path)}</strong>"
        f"<p>{escape(detail)}</p>"
        "</section>"
    )


def _open_directory(path: str) -> None:
    import os
    import subprocess
    import sys

    if sys.platform.startswith("win") and hasattr(os, "startfile"):
        os.startfile(path)
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", path])
        return
    subprocess.Popen(["xdg-open", path])


def _managed_output_folder_path(section: str | None = None) -> str:
    from definers.system.output_paths import (
        managed_output_dir,
        managed_output_root,
    )

    if section is None or not str(section).strip():
        return managed_output_root()
    return managed_output_dir(section)


def open_managed_output_root(section: str | None = None) -> str:
    from pathlib import Path

    output_root = Path(_managed_output_folder_path(section))
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        _open_directory(str(output_root))
    except Exception as error:
        return _output_folder_markup(
            str(output_root),
            f"Could not open the folder automatically: {error}",
            state="error",
        )
    return _output_folder_markup(
        str(output_root),
        "Opened in the system file manager.",
        state="open",
    )


def init_output_folder_controls(
    button_label: str = "Open Outputs Folder",
    *,
    section: str | None = None,
):
    import gradio as gr

    resolved_output_path = _managed_output_folder_path(section)
    status_detail = (
        "Artifacts produced by this GUI are saved here."
        if section is not None and str(section).strip()
        else "Artifacts produced by the GUI are saved here."
    )

    with gr.Row(elem_classes="definers-output-toolbar"):
        button = gr.Button(button_label, variant="secondary")
        status = gr.Markdown(
            value=_output_folder_markup(
                resolved_output_path,
                status_detail,
            ),
            elem_classes="definers-output-toolbar__status",
        )
    button.click(
        fn=lambda: open_managed_output_root(section),
        outputs=[status],
        show_progress="hidden",
    )
    return button, status


def progress_update(
    title: str,
    state: str = "idle",
    detail: str | None = None,
    steps: tuple[str, ...] | list[str] | None = None,
    active_step: int | None = None,
    activity_completed: int | None = None,
    activity_total: int | None = None,
    bytes_downloaded: int | None = None,
    bytes_total: int | None = None,
):
    import gradio as gr

    return gr.update(
        value=_progress_markup(
            title,
            state,
            detail,
            steps=steps,
            active_step=active_step,
            activity_completed=activity_completed,
            activity_total=activity_total,
            bytes_downloaded=bytes_downloaded,
            bytes_total=bytes_total,
        )
    )


def _coerce_outputs(outputs):
    if outputs is None:
        return []
    if isinstance(outputs, (list, tuple)):
        return list(outputs)
    return [outputs]


def _coerce_result_values(result, output_count: int):
    if output_count == 0:
        return ()
    if output_count == 1:
        return (result,)
    if not isinstance(result, (list, tuple)):
        raise TypeError("Expected a list or tuple of outputs")
    if len(result) != output_count:
        raise ValueError("Unexpected number of outputs returned")
    return tuple(result)


def _activity_detail(
    fallback_detail: str,
    activity_snapshot: Any,
) -> str:
    if activity_snapshot is None:
        return fallback_detail
    activity_message = str(getattr(activity_snapshot, "message", "")).strip()
    return activity_message or fallback_detail


def wrap_progress_handler(
    handler,
    *,
    output_count: int,
    action_label: str,
    steps: tuple[str, ...] | list[str] | None = None,
    running_detail: str | None = None,
    success_detail: str | None = None,
    poll_interval_seconds: float = 0.12,
):
    def wrapped(*args):
        import gradio as gr

        from definers.system.download_activity import (
            create_download_activity_task,
            get_download_activity_snapshot,
            resolve_download_activity_task,
            wait_for_download_activity_task,
        )

        idle_updates = tuple(gr.update() for _ in range(output_count))
        resolved_steps = tuple(steps or ()) or (
            "Validate request",
            "Run workflow",
            "Publish result",
        )
        yield (
            *idle_updates,
            progress_update(
                action_label,
                "running",
                "Checking the request.",
                steps=resolved_steps,
                active_step=1,
            ),
        )
        activity_task = create_download_activity_task(handler, *args)
        running_step = min(2, len(resolved_steps))
        yield (
            *idle_updates,
            progress_update(
                action_label,
                "running",
                running_detail or "Working on your request.",
                steps=resolved_steps,
                active_step=running_step,
            ),
        )
        last_sequence = -1
        while True:
            task_done = wait_for_download_activity_task(
                activity_task,
                poll_interval_seconds,
            )
            activity_snapshot = get_download_activity_snapshot(
                activity_task.scope_id
            )
            if (
                activity_snapshot is not None
                and activity_snapshot.sequence != last_sequence
            ):
                last_sequence = activity_snapshot.sequence
                yield (
                    *idle_updates,
                    progress_update(
                        action_label,
                        "running",
                        _activity_detail(
                            running_detail or "Working on your request.",
                            activity_snapshot,
                        ),
                        steps=resolved_steps,
                        active_step=running_step,
                        activity_completed=getattr(
                            activity_snapshot,
                            "completed",
                            None,
                        ),
                        activity_total=getattr(
                            activity_snapshot,
                            "total",
                            None,
                        ),
                        bytes_downloaded=getattr(
                            activity_snapshot,
                            "bytes_downloaded",
                            None,
                        ),
                        bytes_total=getattr(
                            activity_snapshot,
                            "bytes_total",
                            None,
                        ),
                    ),
                )
            if task_done:
                break
        try:
            result, activity_snapshot = resolve_download_activity_task(
                activity_task
            )
            normalized_result = _coerce_result_values(result, output_count)
        except Exception as error:
            yield (
                *idle_updates,
                progress_update(
                    action_label,
                    "error",
                    _activity_detail(
                        str(error),
                        getattr(error, "download_activity_snapshot", None),
                    ),
                    steps=resolved_steps,
                    active_step=running_step,
                    activity_completed=getattr(
                        getattr(error, "download_activity_snapshot", None),
                        "completed",
                        None,
                    ),
                    activity_total=getattr(
                        getattr(error, "download_activity_snapshot", None),
                        "total",
                        None,
                    ),
                    bytes_downloaded=getattr(
                        getattr(error, "download_activity_snapshot", None),
                        "bytes_downloaded",
                        None,
                    ),
                    bytes_total=getattr(
                        getattr(error, "download_activity_snapshot", None),
                        "bytes_total",
                        None,
                    ),
                ),
            )
            raise
        if len(resolved_steps) > 2:
            yield (
                *normalized_result,
                progress_update(
                    action_label,
                    "running",
                    "Publishing the result.",
                    steps=resolved_steps,
                    active_step=len(resolved_steps),
                ),
            )
        yield (
            *normalized_result,
            progress_update(
                action_label,
                "success",
                success_detail or "Result is ready.",
                steps=resolved_steps,
                active_step=len(resolved_steps),
            ),
        )

    return wrapped


def bind_progress_click(
    button,
    handler,
    *,
    progress_output,
    inputs=None,
    outputs=None,
    action_label: str,
    steps: tuple[str, ...] | list[str] | None = None,
    running_detail: str | None = None,
    success_detail: str | None = None,
    show_progress: str = "minimal",
):
    resolved_outputs = _coerce_outputs(outputs)
    return button.click(
        fn=wrap_progress_handler(
            handler,
            output_count=len(resolved_outputs),
            action_label=action_label,
            steps=steps,
            running_detail=running_detail,
            success_detail=success_detail,
        ),
        inputs=inputs,
        outputs=[*resolved_outputs, progress_output],
        show_progress=show_progress,
    )


class GradioShared:
    @staticmethod
    def theme():
        import gradio as gr

        return gr.themes.Base(
            primary_hue=gr.themes.colors.cyan,
            secondary_hue=gr.themes.colors.amber,
            neutral_hue=gr.themes.colors.slate,
            font=(
                gr.themes.GoogleFont("Oxanium"),
                gr.themes.GoogleFont("IBM Plex Sans"),
                "ui-sans-serif",
                "sans-serif",
            ),
            font_mono=(
                gr.themes.GoogleFont("IBM Plex Mono"),
                "ui-monospace",
                "monospace",
            ),
        ).set(
            body_background_fill="radial-gradient(circle at top, #293447 0%, #111722 24%, #090d15 54%, #03050a 100%)",
            body_background_fill_dark="radial-gradient(circle at top, #293447 0%, #111722 24%, #090d15 54%, #03050a 100%)",
            block_background_fill="linear-gradient(180deg, rgba(74, 84, 99, 0.2) 0%, rgba(30, 36, 46, 0.94) 12%, rgba(12, 15, 22, 0.98) 100%)",
            block_background_fill_dark="linear-gradient(180deg, rgba(74, 84, 99, 0.2) 0%, rgba(30, 36, 46, 0.94) 12%, rgba(12, 15, 22, 0.98) 100%)",
            block_border_width="1px",
            block_border_color="rgba(151, 164, 184, 0.28)",
            block_border_color_dark="rgba(151, 164, 184, 0.28)",
            block_title_text_color="#f3f8ff",
            block_title_text_color_dark="#f3f8ff",
            body_text_color="#d6e1ee",
            body_text_color_dark="#d6e1ee",
            input_background_fill="linear-gradient(180deg, rgba(55, 65, 81, 0.22) 0%, rgba(16, 20, 28, 0.96) 100%)",
            input_background_fill_dark="linear-gradient(180deg, rgba(55, 65, 81, 0.22) 0%, rgba(16, 20, 28, 0.96) 100%)",
            button_primary_background_fill="linear-gradient(180deg, #7be7ff 0%, #27b8ea 16%, #0e597b 54%, #0a2433 100%)",
            button_primary_background_fill_hover="linear-gradient(180deg, #a0f1ff 0%, #42d0ff 18%, #1276a3 54%, #0b2d40 100%)",
            button_primary_background_fill_dark="linear-gradient(180deg, #7be7ff 0%, #27b8ea 16%, #0e597b 54%, #0a2433 100%)",
            button_primary_text_color_dark="#031018",
            button_secondary_background_fill="linear-gradient(180deg, #687385 0%, #313946 18%, #171c25 52%, #0b0f16 100%)",
            button_secondary_background_fill_hover="linear-gradient(180deg, #7a869a 0%, #404a59 18%, #1b212c 52%, #0d121a 100%)",
            button_secondary_background_fill_dark="linear-gradient(180deg, #687385 0%, #313946 18%, #171c25 52%, #0b0f16 100%)",
            button_secondary_text_color="#eef6ff",
            button_secondary_text_color_dark="#eef6ff",
            slider_color="#3fdcff",
            slider_color_dark="#3fdcff",
        )

    @staticmethod
    def css() -> str:
        return """
:root {
    color-scheme: dark;
    --definers-ink: #eaf3ff;
    --definers-ink-strong: #f9fcff;
    --definers-muted: #8d9caf;
    --definers-accent: #54e2ff;
    --definers-accent-strong: #bdf6ff;
    --definers-surface: linear-gradient(180deg, rgba(41, 49, 61, 0.2) 0%, rgba(16, 20, 28, 0.94) 18%, rgba(7, 9, 14, 0.99) 100%);
    --definers-surface-raised: linear-gradient(180deg, rgba(86, 99, 118, 0.24) 0%, rgba(24, 29, 39, 0.96) 12%, rgba(9, 12, 18, 0.99) 100%);
    --definers-surface-deep: linear-gradient(180deg, rgba(48, 57, 70, 0.18) 0%, rgba(12, 16, 23, 0.97) 100%);
    --definers-border: rgba(145, 160, 182, 0.2);
    --definers-border-strong: rgba(84, 226, 255, 0.28);
    --definers-shadow: 0 18px 42px rgba(0, 0, 0, 0.46), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    --definers-shadow-soft: 0 10px 24px rgba(0, 0, 0, 0.34);
    --definers-glow: 0 0 0 1px rgba(84, 226, 255, 0.1), 0 0 28px rgba(84, 226, 255, 0.08);
    --definers-grid: rgba(122, 140, 165, 0.08);
}

@keyframes definers-cockpit-rise {
    from {
        opacity: 0;
        transform: translateY(10px) scale(0.985);
    }

    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

@keyframes definers-panel-sheen {
    from {
        transform: translateX(-120%);
    }

    to {
        transform: translateX(120%);
    }
}

tbody > tr:nth-child(odd) {
    background: unset !important;
}

.timestamps time {
    display: flex;
    margin-top: 4mm;
    color: var(--definers-muted);
}

html > body > gradio-app > .gradio-container > .main .contain .block {
    margin-bottom: 18px !important;
}

html > body span.toast-title.error {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

.icon-button-wrapper {
    background: transparent !important;
}

html > body > gradio-app > .gradio-container > .main .contain :is(.settings-wrapper, .control-wrapper) > button {
    min-height: 4mm !important;
    min-width: 4mm !important;
    padding: 0 !important;
}

.label-wrap.open {
    margin-bottom: 0 !important;
}

.app {
    padding: 0 !important;
}

.row > button {
    margin-inline: 1mm;
}

.row > div:not(.icon-wrap) {
    flex: 1 1 0 !important;
    min-width: 0 !important;
}

.main {
    margin-block: 1mm !important;
    max-width: 99% !important;
    position: relative;
    animation: definers-cockpit-rise 360ms ease-out both;
}

h1, h2, p {
    padding: 4mm !important;
}

html > body {
    background: var(--definers-surface) !important;
    color: var(--definers-ink) !important;
    position: relative;
}

html > body::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background:
        radial-gradient(circle at top, rgba(84, 226, 255, 0.12) 0%, rgba(84, 226, 255, 0) 34%),
        radial-gradient(circle at 82% 14%, rgba(255, 191, 102, 0.08) 0%, rgba(255, 191, 102, 0) 20%),
        repeating-linear-gradient(
            90deg,
            transparent 0,
            transparent 22px,
            var(--definers-grid) 23px,
            transparent 24px
        ),
        repeating-linear-gradient(
            180deg,
            transparent 0,
            transparent 22px,
            rgba(84, 226, 255, 0.03) 23px,
            transparent 24px
        );
    opacity: 0.75;
}

html > body::after {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0));
    mix-blend-mode: screen;
}

html > body .gradio-container {
    padding: 4mm !important;
    background: transparent !important;
}

html > body .gradio-container > main {
    max-width: 1440px;
    margin: 0 auto;
    position: relative;
    z-index: 1;
}

html > body :is(.toast-wrap .toast, .toast, .toast-body, .toast-title, [role="alertdialog"], [role="alert"]) {
    background: linear-gradient(180deg, rgba(56, 12, 12, 0.98) 0%, rgba(27, 6, 6, 0.99) 100%) !important;
    color: #fff3f3 !important;
    border: 1px solid rgba(255, 120, 120, 0.38) !important;
    box-shadow: 0 18px 42px rgba(0, 0, 0, 0.52), 0 0 0 1px rgba(255, 120, 120, 0.14) !important;
}

html > body :is(.toast-wrap .toast *, .toast *, .toast-body *, .toast-title *, [role="alertdialog"] *, [role="alert"] *) {
    color: #fff3f3 !important;
}

.controls button {
    margin: auto !important;
}

html > body > gradio-app > .gradio-container > .main .contain button {
    padding: 2mm !important;
    min-height: 0 !important;
    border: 1px solid var(--definers-border) !important;
    color: var(--definers-ink-strong) !important;
    background: var(--definers-surface-deep) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08), 0 8px 20px rgba(0, 0, 0, 0.34) !important;
    transition:
        transform 160ms ease,
        filter 160ms ease,
        box-shadow 180ms ease,
        border-color 180ms ease,
        background 180ms ease;
}

html > body > gradio-app > .gradio-container > .main .contain button.primary {
    filter: brightness(1.5) sepia(1);
    text-shadow: none !important;
    box-shadow: 0 0 0 1px rgba(84, 226, 255, 0.22), 0 14px 28px rgba(0, 0, 0, 0.42), inset 0 1px 0 rgba(255, 255, 255, 0.32) !important;
}

button:hover {
    transform: translateY(-1px);
    filter: brightness(1.05);
}

button:active {
    transform: translateY(0);
    filter: brightness(0.98);
}

html > body footer {
    display: none !important;
}

.options, .options * {
    position: static !important;
    background: white !important;
    color: black !important;
}

#header,
.audio-hero,
.tool-container,
.definers-chat-shell,
.block,
.accordion {
    position: relative;
    overflow: hidden;
    margin-bottom: 18px;
    padding: 30px 34px;
    border: 1px solid var(--definers-border);
    border-radius: 28px;
    background: var(--definers-surface-raised);
    box-shadow: var(--definers-shadow);
    animation: definers-cockpit-rise 420ms ease-out both;
}

#header::before,
.audio-hero::before,
.tool-container::before,
.definers-chat-shell::before,
.block::before,
.accordion::before {
    content: "";
    position: absolute;
    inset: 1px;
    border-radius: inherit;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0) 18%, rgba(84, 226, 255, 0.08) 58%, rgba(0, 0, 0, 0) 100%);
    pointer-events: none;
}

#header::after,
.audio-hero::after,
.tool-container::after,
.definers-chat-shell::after {
    content: "";
    position: absolute;
    inset: auto -8% -35% auto;
    width: 260px;
    height: 260px;
    border-radius: 999px;
    background: radial-gradient(circle, rgba(84, 226, 255, 0.18), rgba(84, 226, 255, 0));
    filter: blur(2px);
    pointer-events: none;
}

.audio-hero .eyebrow,
#header .eyebrow {
    margin: 0 0 10px !important;
    color: var(--definers-accent-strong);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    text-shadow: 0 0 18px rgba(84, 226, 255, 0.18);
}

.audio-hero h1,
#header h1 {
    margin: 0;
    color: var(--definers-ink-strong) !important;
    font-size: clamp(2.1rem, 3vw, 3.4rem);
    line-height: 1.02;
    text-shadow: 0 10px 26px rgba(0, 0, 0, 0.34);
}

.audio-hero p:last-child,
#header p:last-child {
    max-width: 760px;
    margin: 12px 0 0 !important;
    color: var(--definers-muted);
    font-size: 1rem;
    line-height: 1.6;
}

.tool-container {
    padding: 10px 12px !important;
    border-radius: 24px !important;
    backdrop-filter: blur(18px);
    box-shadow: var(--definers-shadow), var(--definers-glow);
}

.definers-progress-panel {
    margin-bottom: 18px !important;
}

.definers-output-toolbar {
    gap: 14px !important;
    align-items: stretch !important;
    margin-bottom: 18px !important;
}

.definers-output-toolbar > .column,
.definers-output-toolbar > .gradio-container,
.definers-output-toolbar > div {
    min-width: 0;
}

.definers-output-toolbar button {
    min-width: 220px;
}

.definers-output-toolbar__status {
    flex: 1 1 auto;
    margin: 0 !important;
}

.definers-output-toolbar__status > * {
    margin: 0 !important;
}

.definers-output-toolbar__card {
    margin: 0 !important;
    padding: 16px 18px !important;
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 20px;
    background: linear-gradient(180deg, rgba(57, 68, 82, 0.24) 0%, rgba(16, 21, 30, 0.94) 100%);
    box-shadow: var(--definers-shadow-soft);
}

.definers-output-toolbar__card span {
    display: block;
    margin-bottom: 6px;
    color: var(--definers-accent-strong);
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.definers-output-toolbar__card strong {
    display: block;
    min-width: 0;
    color: var(--definers-ink-strong);
    font-size: 0.9rem;
    line-height: 1.45;
    word-break: break-word;
}

.definers-output-toolbar__card p {
    margin: 8px 0 0 0 !important;
    padding: 0 !important;
    color: var(--definers-muted);
    line-height: 1.45;
}

.definers-output-toolbar__card--open {
    border-color: rgba(74, 222, 128, 0.28);
}

.definers-output-toolbar__card--error {
    border-color: rgba(248, 113, 113, 0.3);
}

.definers-progress-shell {
    position: relative;
    overflow: hidden;
    margin: 0 !important;
    padding: 18px 22px !important;
    border: 1px solid var(--definers-border);
    border-radius: 22px;
    background: linear-gradient(180deg, rgba(57, 68, 82, 0.28) 0%, rgba(16, 21, 30, 0.96) 100%);
    box-shadow: var(--definers-shadow-soft);
}

.definers-progress-shell__meta {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 8px;
}

.definers-progress-shell__meta span {
    color: var(--definers-accent-strong);
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}

.definers-progress-shell__meta strong {
    color: var(--definers-ink-strong);
    font-size: 1rem;
}

.definers-progress-shell__count {
    margin-left: auto;
    color: var(--definers-muted);
    font-size: 0.76rem;
    font-style: normal;
}

.definers-progress-shell p {
    margin: 0 0 12px 0 !important;
    padding: 0 !important;
    color: var(--definers-muted);
    line-height: 1.5;
}

.definers-progress-shell__summary {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    margin-bottom: 12px;
}

.definers-progress-shell__summary div {
    min-width: 0;
    padding: 10px 12px;
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 16px;
    background: rgba(15, 23, 42, 0.34);
}

.definers-progress-shell__summary span {
    display: block;
    margin-bottom: 6px;
    color: var(--definers-muted);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.definers-progress-shell__summary strong {
    display: block;
    min-width: 0;
    color: var(--definers-ink-strong);
    font-size: 0.88rem;
    line-height: 1.4;
}

.definers-progress-shell__rail {
    position: relative;
    height: 8px;
    border-radius: 999px;
    background: rgba(148, 163, 184, 0.18);
    overflow: hidden;
}

.definers-progress-shell__bar {
    position: absolute;
    inset: 0 auto 0 0;
    width: 0;
    border-radius: inherit;
    background: linear-gradient(90deg, rgba(84, 226, 255, 0.25), rgba(84, 226, 255, 0.92), rgba(255, 191, 102, 0.85));
    transition: width 180ms ease, opacity 180ms ease;
}

.definers-progress-shell--idle .definers-progress-shell__bar {
    opacity: 0.48;
}

.definers-progress-shell--running .definers-progress-shell__bar {
    animation: definers-panel-sheen 1.2s linear infinite;
}

.definers-progress-shell--success .definers-progress-shell__bar {
    background: linear-gradient(90deg, rgba(74, 222, 128, 0.62), rgba(134, 239, 172, 0.95));
}

.definers-progress-shell--error .definers-progress-shell__bar {
    background: linear-gradient(90deg, rgba(248, 113, 113, 0.75), rgba(252, 165, 165, 0.95));
}

.definers-progress-shell__steps {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 10px;
    margin: 12px 0 0 0 !important;
    padding: 0 !important;
    list-style: none;
}

.definers-progress-shell__step {
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 0;
    padding: 10px 12px;
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 16px;
    background: rgba(15, 23, 42, 0.26);
}

.definers-progress-shell__step span {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    flex: 0 0 24px;
    border-radius: 999px;
    border: 1px solid rgba(148, 163, 184, 0.26);
    color: var(--definers-muted);
    font-size: 0.76rem;
    font-weight: 700;
}

.definers-progress-shell__step strong {
    min-width: 0;
    color: var(--definers-muted);
    font-size: 0.86rem;
    line-height: 1.35;
}

.definers-progress-shell__step--done {
    border-color: rgba(74, 222, 128, 0.28);
    background: rgba(21, 128, 61, 0.16);
}

.definers-progress-shell__step--done span {
    border-color: rgba(74, 222, 128, 0.32);
    color: rgb(187, 247, 208);
}

.definers-progress-shell__step--done strong {
    color: rgb(220, 252, 231);
}

.definers-progress-shell__step--active {
    border-color: rgba(84, 226, 255, 0.34);
    background: rgba(8, 145, 178, 0.16);
}

.definers-progress-shell__step--active span {
    border-color: rgba(84, 226, 255, 0.36);
    color: rgb(186, 230, 253);
}

.definers-progress-shell__step--active strong {
    color: var(--definers-ink-strong);
}

.definers-progress-shell__step--error {
    border-color: rgba(248, 113, 113, 0.34);
    background: rgba(153, 27, 27, 0.16);
}

.definers-progress-shell__step--error span {
    border-color: rgba(248, 113, 113, 0.34);
    color: rgb(254, 202, 202);
}

.definers-progress-shell__step--error strong {
    color: rgb(254, 226, 226);
}

@media (max-width: 900px) {
    .definers-output-toolbar {
        flex-direction: column;
    }

    .definers-progress-shell__summary {
        grid-template-columns: 1fr;
    }
}

.tool-container h2,
.tool-container h3 {
    color: var(--definers-ink-strong) !important;
    letter-spacing: 0.01em;
}

.row {
    gap: 16px !important;
    align-items: stretch !important;
    padding-block: 12px !important;
}

button {
    min-height: 44px !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em;
}

button.secondary {
    box-shadow: var(--definers-shadow-soft), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
}

html > body > gradio-app > .gradio-container > .main .contain .source-selection,
html > body > gradio-app > .gradio-container > .main .contain .source-selection > button {
    min-height: 1cm !important;
    min-width: 1cm !important;
    box-shadow: none;
}

code {
    background: transparent !important;
    border: none !important;
}

textarea,
input:not([type="radio"]):not([type="checkbox"]):not([type="range"]):not([type="color"]):not([role="listbox"]),
select {
    border-radius: 18px !important;
    border: 1px solid var(--definers-border) !important;
    background: var(--definers-surface-deep) !important;
    color: var(--definers-ink) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06) !important;
    transition:
        transform 160ms ease,
        border-color 180ms ease,
        box-shadow 180ms ease,
        background 180ms ease;
}

textarea {
    padding: 12px 14px !important;
}

textarea:focus,
input:focus,
select:focus {
    border-color: var(--definers-border-strong) !important;
    outline: none !important;
    box-shadow: 0 0 0 4px rgba(84, 226, 255, 0.12), 0 0 12px rgba(84, 226, 255, 0.08) !important;
    transform: translateY(-1px);
}

.form,
.block,
fieldset,
.panel,
.accordion {
    border-radius: 24px !important;
}

.accordion {
    border: 1px solid var(--definers-border) !important;
    background: var(--definers-surface-deep) !important;
    box-shadow: var(--definers-shadow-soft) !important;
}

video,
img {
    border-radius: 20px;
    box-shadow: var(--definers-shadow-soft);
}

tr.file > td.download {
    min-width: unset !important;
    width: auto !important;
}

.definers-chat-shell {
    padding: 18px !important;
    border-radius: 28px !important;
    box-shadow: var(--definers-shadow), var(--definers-glow);
}

.definers-chat-shell h1 {
    color: var(--definers-ink-strong) !important;
}

.definers-chat-shell p {
    color: var(--definers-muted) !important;
}

.definers-chatbot {
    border: none !important;
    background: transparent !important;
}

.definers-chatbot [class*="message"] {
    border: 1px solid var(--definers-border) !important;
    border-radius: 22px !important;
    box-shadow: 0 12px 26px rgba(0, 0, 0, 0.34);
    backdrop-filter: blur(18px);
    transition:
        transform 180ms ease,
        box-shadow 180ms ease,
        border-color 180ms ease;
}

.definers-chatbot [class*="user"] [class*="message"] {
    background: linear-gradient(135deg, rgba(86, 235, 255, 0.98) 0%, rgba(46, 181, 232, 0.94) 34%, rgba(14, 77, 113, 0.98) 100%) !important;
    color: #f8fdff !important;
    border-color: rgba(124, 239, 255, 0.28) !important;
}

.definers-chatbot [class*="bot"] [class*="message"],
.definers-chatbot [class*="assistant"] [class*="message"] {
    background: linear-gradient(180deg, rgba(60, 70, 85, 0.2) 0%, rgba(14, 18, 26, 0.96) 100%) !important;
    color: var(--definers-ink) !important;
}

.definers-chatbot [class*="message"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 16px 34px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(84, 226, 255, 0.08);
}

.definers-chat-input {
    border: 1px solid var(--definers-border) !important;
    border-radius: 24px !important;
    background: var(--definers-surface-raised) !important;
    box-shadow: var(--definers-shadow-soft), inset 0 1px 0 rgba(255, 255, 255, 0.08);
}

.definers-chat-input textarea {
    min-height: 92px !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    color: var(--definers-ink) !important;
}

.block {
    border: 1px solid var(--definers-border) !important;
    background: var(--definers-surface) !important;
    box-shadow: var(--definers-shadow), var(--definers-glow);
    margin-bottom: 18px !important;
    overflow-x: hidden !important;
}

html > body > gradio-app > .gradio-container > .main .contain .styler {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

.gr-group {
    border: none !important;
    padding: 6px;
    margin-top: 12px;
    border-radius: 16px;
}

div.form {
    background: transparent !important;
    overflow-x: hidden !important;
}

label + .tab-like-container {
    border: none !important;
    padding-inline: 2px;
    filter: drop-shadow(2px 4px 6px black);
}

.tool-container:hover,
.definers-chat-shell:hover,
.block:hover,
.accordion:hover {
    border-color: var(--definers-border-strong) !important;
    box-shadow: var(--definers-shadow), 0 0 0 1px rgba(84, 226, 255, 0.08), 0 0 34px rgba(84, 226, 255, 0.08) !important;
}

.tool-container:hover::before,
.definers-chat-shell:hover::before,
#header:hover::before,
.audio-hero:hover::before {
    animation: definers-panel-sheen 900ms ease;
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
    }
}

@media (max-width: 900px) {
    html > body .gradio-container {
        padding: 14px 14px 24px !important;
    }

    #header,
    .audio-hero,
    .definers-chat-shell {
        padding: 22px 18px !important;
        border-radius: 22px !important;
    }
}
"""

    @staticmethod
    def init_chat(title: str = "Chatbot", handler=None):
        import gradio as gr

        from definers.ui.chat_handlers import get_chat_response_stream

        active_handler = (
            get_chat_response_stream if handler is None else handler
        )
        with gr.Group(elem_classes="definers-chat-shell"):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                elem_classes="definers-chatbot",
                height=560,
                buttons=["copy", "copy_all"],
            )
            textbox = gr.MultimodalTextbox(
                elem_classes="definers-chat-input",
                placeholder="Describe the task, attach context if needed, then send.",
                show_label=False,
                submit_btn="Send",
                lines=3,
                max_lines=8,
                sources=["upload", "microphone"],
            )
            return gr.ChatInterface(
                fn=active_handler,
                chatbot=chatbot,
                textbox=textbox,
                multimodal=True,
                title=title,
                description="Production-grade assistant for generation, analysis, debugging, and delivery decisions.",
                save_history=True,
                show_progress="minimal",
                concurrency_limit=None,
            )

    @classmethod
    def launch_blocks(
        cls,
        app,
        *,
        custom_css: str | None = None,
        custom_theme=None,
        server_port: int = 7860,
    ):
        app.launch(
            server_name="0.0.0.0",
            theme=cls.theme() if custom_theme is None else custom_theme,
            css=cls.css() if custom_css is None else custom_css,
            server_port=server_port,
        )


ChatHandler = object
theme = GradioShared.theme
css = GradioShared.css
init_chat = GradioShared.init_chat
launch_blocks = GradioShared.launch_blocks
