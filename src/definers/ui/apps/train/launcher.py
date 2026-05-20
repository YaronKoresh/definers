from definers.ui.gradio_shared import css as shared_css, launch_blocks

from ...gradio_shared import css, theme
from .ui import (
    build_train_app,
)


def launch_train_app(
    visible_sections=None,
    *,
    app_title="Definers ML Studio",
    hero_label="Definers ML Studio",
    hero_heading="Train, run, inspect, and bootstrap models from one surface.",
    hero_description=(
        "The training launcher is a full ML cockpit for data-driven training, "
        "task inference, text tooling, runtime diagnostics, and bootstrap flows."
    ),
):
    launch_blocks(
        build_train_app(
            visible_sections=visible_sections,
            app_title=app_title,
            hero_label=hero_label,
            hero_heading=hero_heading,
            hero_description=hero_description,
        ),
        custom_css=css(),
        custom_theme=theme(),
    )
