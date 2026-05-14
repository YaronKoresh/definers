import importlib


def test_media_web_transfer_owner_module_exports_download_surface():
    transfer_module = importlib.import_module("definers.media.web_transfer")

    assert hasattr(transfer_module, "download_file")
    assert hasattr(transfer_module, "download_and_unzip")
    assert hasattr(transfer_module, "create_http_transfer_strategy")
    assert hasattr(transfer_module, "http_transfer_policy")


def test_media_package_exposes_web_transfer_owner_module():
    media_module = importlib.import_module("definers.media")
    web_transfer_module = importlib.import_module("definers.media.web_transfer")

    assert media_module.web_transfer is web_transfer_module
