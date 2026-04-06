from definers.presentation.apps.train_ui import (
    build_train_app,
    train_css,
    train_theme,
)
from definers.presentation.gradio_shared import css as shared_css, launch_blocks


def launch_train_app():
    launch_blocks(
        build_train_app(),
        custom_css=shared_css() + "\n" + train_css(),
        custom_theme=train_theme(),
    )
