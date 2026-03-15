from definers.media import web_transfer

__all__ = [
    "add_to_path_windows",
    "broadcast_path_change",
    "download_and_unzip",
    "download_file",
    "extract_text",
    "google_drive_download",
    "linked_url",
]

broadcast_path_change = web_transfer.broadcast_path_change
extract_text = web_transfer.extract_text
google_drive_download = web_transfer.google_drive_download
linked_url = web_transfer.linked_url


def download_file(url: str, destination: str) -> str | None:
    return web_transfer.download_file(
        url,
        destination,
        executor=web_transfer.execute_async_operation,
        orchestrator_factory=web_transfer.create_http_orchestrator,
    )


def download_and_unzip(url: str, extract_to: str) -> bool:
    return web_transfer.download_and_unzip(
        url,
        extract_to,
        executor=web_transfer.execute_async_operation,
        orchestrator_factory=web_transfer.create_zip_orchestrator,
    )


def add_to_path_windows(folder_path: str) -> None:
    web_transfer.add_to_path_windows(
        folder_path, broadcaster=web_transfer.broadcast_path_change
    )
