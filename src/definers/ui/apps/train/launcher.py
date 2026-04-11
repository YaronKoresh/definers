from definers.ui.gradio_shared import css as shared_css, launch_blocks

from .coach_ui import train_coach_css
from .ui import (
    build_train_app,
    train_css,
    train_theme,
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
        custom_css=shared_css() + "\n" + train_css() + "\n" + train_coach_css(),
        custom_theme=train_theme(),
    )
