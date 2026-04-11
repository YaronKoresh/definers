from __future__ import annotations

import asyncio
import base64
import os
import random
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from definers.constants import MAX_INPUT_LENGTH, user_agents
from definers.media.transfer.orchestrators import (
    ResourceRetrievalOrchestrator,
    create_http_orchestrator,
    create_zip_orchestrator,
)
from definers.system import log
from definers.system.download_activity import report_download_activity

try:
    from lxml.cssselect import CSSSelector
    from lxml.html import fromstring
except ImportError:
    CSSSelector = None
    fromstring = None


def google_drive_download(id, dest, unzip=True):
    from googledrivedownloader import download_file_from_google_drive

    item_label = Path(str(dest)).name or str(dest)
    report_download_activity(
        item_label,
        detail="Downloading artifact from Google Drive.",
        phase="artifact",
    )
    try:
        download_file_from_google_drive(
            file_id=id, dest_path=dest, unzip=unzip, showsize=False
        )
        report_download_activity(
            item_label,
            detail=(
                "Downloaded and extracted the Google Drive artifact."
                if unzip
                else "Downloaded the Google Drive artifact."
            ),
            phase="extract" if unzip else "artifact",
            completed=1,
            total=1,
        )
    except Exception as error:
        from definers.media import web_transfer as web_transfer_module

        web_transfer_module.log("google_drive_download failed", error)
        return None


def linked_url(url):
    host = url.split("?")[0]
    if "?" in url:
        param = "?" + url.split("?")[1]
    else:
        param = ""
    html_string = f'''\n         <!DOCTYPE html>\n        <html>\n            <head>\n                <meta charset="UTF-8">\n                <base href="{host}" target="_top">\n                <a href="{param}"></a>\n            </head>\n            <body onload='document.querySelector("a").click()'></body>\n        </html>\n    '''
    html_bytes = html_string.encode("utf-8")
    base64_encoded_html = base64.b64encode(html_bytes).decode("utf-8")
    data_url = f"data:text/html;charset=utf-8;base64,{base64_encoded_html}"
    return data_url


def extract_text(url, selector):
    from playwright.sync_api import expect, sync_playwright

    if CSSSelector is None or fromstring is None:
        raise ImportError("lxml with cssselect is required for extract_text")

    xpath = CSSSelector(selector).path
    log("URL", url)
    html_string = None
    with sync_playwright() as playwright:
        browser_app = playwright.firefox.launch(headless=True)
        browser = browser_app.new_context(
            locale="en-US",
            timezone_id="America/New_York",
            user_agent=random.choice(user_agents["firefox"]),
            color_scheme="dark",
        )
        page = browser.new_page()
        page.goto(url, referer="https://duckduckgo.com/", timeout=18 * 1000)
        expect(page.locator(selector)).not_to_be_empty()
        page.wait_for_timeout(2000)
        html_string = page.content()
        browser.close()
        browser_app.close()
    if html_string is None:
        return None
    if not str(html_string).strip():
        return ""
    try:
        html = fromstring(html_string)
    except Exception:
        return ""
    elems = html.xpath(xpath)
    elems = [
        el.text_content().strip() for el in elems if el.text_content().strip()
    ]
    if len(elems) == 0:
        return ""
    return elems[0]


def execute_async_operation(coroutine: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    operation_outcome: dict[str, Any] = {"result": None, "error": None}

    def runner() -> None:
        event_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(event_loop)
            operation_outcome["result"] = event_loop.run_until_complete(
                coroutine
            )
            event_loop.run_until_complete(event_loop.shutdown_asyncgens())
            shutdown_default_executor = getattr(
                event_loop,
                "shutdown_default_executor",
                None,
            )
            if callable(shutdown_default_executor):
                event_loop.run_until_complete(shutdown_default_executor())
        except Exception as runner_fault:
            operation_outcome["error"] = runner_fault
        finally:
            asyncio.set_event_loop(None)
            event_loop.close()

    execution_thread = threading.Thread(target=runner, daemon=False)
    execution_thread.start()
    execution_thread.join()
    if operation_outcome["error"] is not None:
        raise operation_outcome["error"]
    return operation_outcome["result"]


def validate_network_url(url: str) -> None:
    if not isinstance(url, str):
        raise ValueError("url must be a string")
    if len(url) > MAX_INPUT_LENGTH:
        raise ValueError(f"url too long ({len(url)} > {MAX_INPUT_LENGTH})")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError(f"unsupported URL scheme: {url}")


def _execute_orchestrated_transfer(
    source_uri: str,
    target_node: str,
    *,
    executor: Callable[[Any], Any],
    orchestrator_factory: Callable[[], ResourceRetrievalOrchestrator],
) -> bool:
    async def async_runner() -> bool:
        orchestrator = orchestrator_factory()
        return await orchestrator.process(source_uri, target_node)

    return bool(executor(async_runner()))


def download_file(
    url: str,
    destination: str,
    executor: Callable[[Any], Any] = execute_async_operation,
    orchestrator_factory: Callable[
        [], ResourceRetrievalOrchestrator
    ] = create_http_orchestrator,
) -> str | None:
    validate_network_url(url)
    success = _execute_orchestrated_transfer(
        url,
        destination,
        executor=executor,
        orchestrator_factory=orchestrator_factory,
    )
    return destination if success else None


def download_and_unzip(
    url: str,
    extract_to: str,
    executor: Callable[[Any], Any] = execute_async_operation,
    orchestrator_factory: Callable[
        [], ResourceRetrievalOrchestrator
    ] = create_zip_orchestrator,
) -> bool:
    validate_network_url(url)
    return _execute_orchestrated_transfer(
        url,
        extract_to,
        executor=executor,
        orchestrator_factory=orchestrator_factory,
    )


def broadcast_path_change():
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    send_message_timeout = ctypes.windll.user32.SendMessageTimeoutW
    send_message_timeout(
        65535, 26, 0, "Environment", 2, 5000, ctypes.byref(wintypes.DWORD())
    )


def add_to_path_windows(
    folder_path: str, broadcaster: Callable[[], None] | None = None
) -> None:
    if sys.platform != "win32":
        return
    import winreg

    folder_path = os.path.normpath(folder_path).strip('"')
    if broadcaster is None:
        from definers.media import web_transfer as web_transfer_module

        path_change_broadcaster = web_transfer_module.broadcast_path_change
    else:
        path_change_broadcaster = broadcaster
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            "Environment",
            0,
            winreg.KEY_ALL_ACCESS,
        )
        try:
            current_path, _ = winreg.QueryValueEx(key, "PATH")
        except FileNotFoundError:
            current_path = ""
        parts = [
            p.strip('"') for p in str(current_path).split(";") if p.strip()
        ]
        if folder_path not in parts:
            parts.insert(0, folder_path)
            new_path = ";".join(parts)
            winreg.SetValueEx(key, "PATH", 0, winreg.REG_EXPAND_SZ, new_path)
            os.environ["PATH"] = (
                folder_path + os.pathsep + os.environ.get("PATH", "")
            )
            path_change_broadcaster()
        winreg.CloseKey(key)
    except Exception:
        return
