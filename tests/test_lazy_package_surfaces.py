import importlib
import sys


def test_root_package_lazy_exposes_chat_module():
    definers = importlib.import_module("definers")
    definers.__dict__.pop("chat", None)
    sys.modules.pop("definers.chat", None)

    assert "definers.chat" not in sys.modules
    assert definers.chat is importlib.import_module("definers.chat")


def test_data_package_lazy_exposes_submodules():
    data = importlib.import_module("definers.data")
    data.__dict__.pop("arrays", None)
    sys.modules.pop("definers.data.arrays", None)

    assert "definers.data.arrays" not in sys.modules
    assert data.arrays is importlib.import_module("definers.data.arrays")


def test_chat_package_lazy_exposes_handler_exports():
    chat = importlib.import_module("definers.chat")
    chat.__dict__.pop("handle_chat_request", None)
    sys.modules.pop("definers.chat.handlers", None)

    assert "definers.chat.handlers" not in sys.modules
    assert (
        chat.handle_chat_request
        is importlib.import_module("definers.chat.handlers").handle_chat_request
    )


def test_text_package_lazy_exposes_translation_exports():
    text = importlib.import_module("definers.text")
    text.__dict__.pop("google_translate", None)
    sys.modules.pop("definers.text.translation", None)

    assert "definers.text.translation" not in sys.modules
    assert (
        text.google_translate
        is importlib.import_module("definers.text.translation").google_translate
    )


def test_media_package_lazy_exposes_modules_and_helpers():
    media = importlib.import_module("definers.media")
    media.__dict__.pop("transfer", None)
    media.__dict__.pop("image_helpers", None)
    media.__dict__.pop("video_helpers", None)
    sys.modules.pop("definers.media.transfer", None)
    sys.modules.pop("definers.image.helpers", None)
    sys.modules.pop("definers.video.helpers", None)

    assert "definers.media.transfer" not in sys.modules
    assert "definers.image.helpers" not in sys.modules
    assert "definers.video.helpers" not in sys.modules
    assert media.transfer is importlib.import_module("definers.media.transfer")
    assert media.image_helpers is importlib.import_module(
        "definers.image.helpers"
    )
    assert media.video_helpers is importlib.import_module(
        "definers.video.helpers"
    )
