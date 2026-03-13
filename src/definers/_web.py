import asyncio
import base64
import io
import logging
import os
import random
import sys
import threading
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from definers._capabilities import (
    CircuitBreaker,
    CircuitBreakerOpenException,
    with_retry,
)
from definers._constants import user_agents
from definers._system import log


def google_drive_download(id, dest, unzip=True):
    from googledrivedownloader import download_file_from_google_drive

    try:
        download_file_from_google_drive(
            file_id=id, dest_path=dest, unzip=unzip, showsize=False
        )
    except Exception as e:
        log("google_drive_download failed", e)
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


def geo_new_york():
    return {
        "latitude": random.uniform(40.5, 40.9),
        "longitude": random.uniform(-74.2, -73.7),
    }


try:
    from lxml.cssselect import CSSSelector
    from lxml.html import fromstring
except ImportError:
    CSSSelector = None
    fromstring = None


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


from typing import Protocol, runtime_checkable


@runtime_checkable
class NetworkTransferStrategy(Protocol):
    async def execute_transfer(
        self, source_uri: str, target_node: Path
    ) -> bool: ...


class HttpChunkedTransferStrategy:
    def __init__(
        self, chunk_size_bytes: int = 8192, request_timeout_seconds: float = 30
    ):
        self.chunk_size_bytes = chunk_size_bytes
        self.request_timeout_seconds = request_timeout_seconds

    async def execute_transfer(
        self, source_uri: str, target_node: Path
    ) -> bool:
        try:
            import aiofiles
            import aiohttp
        except ImportError:
            await asyncio.to_thread(
                self._execute_transfer_sync, source_uri, target_node
            )
            return True
        target_node.parent.mkdir(parents=True, exist_ok=True)
        request_timeout = aiohttp.ClientTimeout(
            total=self.request_timeout_seconds
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(
                source_uri, timeout=request_timeout
            ) as network_response:
                network_response.raise_for_status()
                async with aiofiles.open(
                    target_node, "wb"
                ) as persistent_storage:
                    async for (
                        data_chunk
                    ) in network_response.content.iter_chunked(
                        self.chunk_size_bytes
                    ):
                        await persistent_storage.write(data_chunk)
        return True

    def _execute_transfer_sync(
        self, source_uri: str, target_node: Path
    ) -> None:
        target_node.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(
            source_uri, timeout=self.request_timeout_seconds
        ) as response:
            with open(target_node, "wb") as persistent_storage:
                while True:
                    data_chunk = response.read(self.chunk_size_bytes)
                    if not data_chunk:
                        break
                    persistent_storage.write(data_chunk)


class ZipExtractTransferStrategy:
    def __init__(
        self, chunk_size_bytes: int = 8192, request_timeout_seconds: float = 60
    ):
        self.chunk_size_bytes = chunk_size_bytes
        self.request_timeout_seconds = request_timeout_seconds

    async def execute_transfer(
        self, source_uri: str, target_node: Path
    ) -> bool:
        try:
            import aiohttp
        except ImportError:
            await asyncio.to_thread(
                self._execute_transfer_sync, source_uri, target_node
            )
            return True
        target_node.mkdir(parents=True, exist_ok=True)
        request_timeout = aiohttp.ClientTimeout(
            total=self.request_timeout_seconds
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(
                source_uri, timeout=request_timeout
            ) as network_response:
                network_response.raise_for_status()
                memory_buffer = io.BytesIO(await network_response.read())
                with zipfile.ZipFile(memory_buffer) as archive_context:
                    archive_context.extractall(target_node)
        return True

    def _execute_transfer_sync(
        self, source_uri: str, target_node: Path
    ) -> None:
        target_node.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(
            source_uri, timeout=self.request_timeout_seconds
        ) as response:
            payload = response.read()
        memory_buffer = io.BytesIO(payload)
        with zipfile.ZipFile(memory_buffer) as archive_context:
            archive_context.extractall(target_node)


class ResourceRetrievalOrchestrator:
    def __init__(
        self,
        strategy: NetworkTransferStrategy,
        circuit_breaker: CircuitBreaker | None = None,
        max_retries: int = 3,
        base_delay_seconds: float = 0.5,
    ):
        self.strategy = strategy
        self.circuit_breaker = circuit_breaker or CircuitBreaker(
            failure_threshold=3, recovery_timeout=30
        )
        self.max_retries = max_retries
        self.base_delay_seconds = base_delay_seconds

    async def process(self, source_uri: str, target_node: str | Path) -> bool:
        target_path_object = Path(target_node)

        @with_retry(max_retries=self.max_retries, delay=self.base_delay_seconds)
        async def transfer_operation() -> bool:
            return await self.strategy.execute_transfer(
                source_uri, target_path_object
            )

        try:
            return await self.circuit_breaker.execute_async(transfer_operation)
        except CircuitBreakerOpenException as circuit_open_fault:
            logging.getLogger(__name__).error(
                "Transfer blocked by open circuit: %s", str(circuit_open_fault)
            )
            return False
        except Exception as execution_fault:
            logging.getLogger(__name__).error(
                "Transfer fault: %s", str(execution_fault)
            )
            return False


def _execute_async_operation(coroutine: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    operation_outcome: dict[str, Any] = {"result": None, "error": None}

    def runner() -> None:
        try:
            operation_outcome["result"] = asyncio.run(coroutine)
        except Exception as runner_fault:
            operation_outcome["error"] = runner_fault

    execution_thread = threading.Thread(target=runner, daemon=False)
    execution_thread.start()
    execution_thread.join()
    if operation_outcome["error"] is not None:
        raise operation_outcome["error"]
    return operation_outcome["result"]


def download_file(url: str, destination: str) -> str | None:

    from definers._constants import MAX_INPUT_LENGTH

    if not isinstance(url, str):
        raise ValueError("url must be a string")
    if len(url) > MAX_INPUT_LENGTH:
        raise ValueError(f"url too long ({len(url)} > {MAX_INPUT_LENGTH})")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError(f"unsupported URL scheme: {url}")

    async def _async_runner() -> bool:
        orchestrator = ResourceRetrievalOrchestrator(
            HttpChunkedTransferStrategy()
        )
        return await orchestrator.process(url, destination)

    success = _execute_async_operation(_async_runner())
    return destination if success else None


def download_and_unzip(url: str, extract_to: str) -> bool:

    from definers._constants import MAX_INPUT_LENGTH

    if not isinstance(url, str):
        raise ValueError("url must be a string")
    if len(url) > MAX_INPUT_LENGTH:
        raise ValueError(f"url too long ({len(url)} > {MAX_INPUT_LENGTH})")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError(f"unsupported URL scheme: {url}")

    async def _async_runner() -> bool:
        orchestrator = ResourceRetrievalOrchestrator(
            ZipExtractTransferStrategy()
        )
        return await orchestrator.process(url, extract_to)

    return _execute_async_operation(_async_runner())


def broadcast_path_change():
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    SendMessageTimeout = ctypes.windll.user32.SendMessageTimeoutW
    SendMessageTimeout(
        65535, 26, 0, "Environment", 2, 5000, ctypes.byref(wintypes.DWORD())
    )


def add_to_path_windows(folder_path: str) -> None:
    if sys.platform != "win32":
        return
    import winreg

    folder_path = os.path.normpath(folder_path).strip('"')
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_ALL_ACCESS
        )
        try:
            (current_path, _) = winreg.QueryValueEx(key, "PATH")
        except FileNotFoundError:
            current_path = ""
        parts = [p.strip('"') for p in current_path.split(";") if p.strip()]
        if folder_path not in parts:
            parts.insert(0, folder_path)
            new_path = ";".join(parts)
            winreg.SetValueEx(key, "PATH", 0, winreg.REG_EXPAND_SZ, new_path)
            os.environ["PATH"] = folder_path + os.pathsep + os.environ["PATH"]
            broadcast_path_change()
            print(f"Added to PATH: {folder_path}")
        winreg.CloseKey(key)
    except Exception as e:
        print(f"Error updating PATH: {e}")
