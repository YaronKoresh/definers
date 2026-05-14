import importlib


def test_chat_package_imports_directly():
    chat = importlib.import_module("definers.chat")

    assert chat.handlers is importlib.import_module("definers.chat.handlers")


def test_data_package_exposes_submodules_directly():
    data = importlib.import_module("definers.data")
    arrays = importlib.import_module("definers.data.arrays")

    assert data.arrays is arrays


def test_chat_package_exposes_handler_exports_directly():
    chat = importlib.import_module("definers.chat")
    handlers = importlib.import_module("definers.chat.handlers")

    assert chat.handle_chat_request is handlers.handle_chat_request


def test_text_package_exposes_translation_exports_directly():
    text = importlib.import_module("definers.text")
    translation = importlib.import_module("definers.text.translation")

    assert text.google_translate is translation.google_translate


def test_media_package_exposes_modules_and_helpers_directly():
    media = importlib.import_module("definers.media")
    web_transfer = importlib.import_module("definers.media.web_transfer")
    image_helpers = importlib.import_module("definers.image.helpers")
    video_helpers = importlib.import_module("definers.video.helpers")

    assert media.web_transfer is importlib.import_module(
        "definers.media.web_transfer"
    )
    assert media.web_transfer is web_transfer
    assert media.image_helpers is image_helpers
    assert media.video_helpers is video_helpers
