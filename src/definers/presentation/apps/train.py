from definers.presentation.apps.train_ui import build_train_app
from definers.presentation.gradio_shared import launch_blocks


def launch_train_app():
    launch_blocks(build_train_app())
