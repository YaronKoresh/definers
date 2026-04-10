import importlib


def test_media_transfer_owner_package_exports_download_surface():
    transfer_module = importlib.import_module("definers.media.transfer")

    assert hasattr(transfer_module, "download_file")
    assert hasattr(transfer_module, "download_and_unzip")
    assert hasattr(transfer_module, "create_http_transfer_strategy")
    assert hasattr(transfer_module, "http_transfer_policy")


def test_web_transfer_facade_reexports_owner_api():
    transfer_module = importlib.import_module("definers.media.transfer")
    web_transfer_module = importlib.import_module("definers.media.web_transfer")

    assert web_transfer_module.download_file is transfer_module.download_file
    assert (
        web_transfer_module.download_and_unzip
        is transfer_module.download_and_unzip
    )
    assert (
        web_transfer_module.create_http_transfer_strategy
        is transfer_module.create_http_transfer_strategy
    )
