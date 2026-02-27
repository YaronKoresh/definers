"""Web utilities for the definers package."""

import argparse
import asyncio
import base64
import collections
import collections.abc
import concurrent
import ctypes
import gc
import getpass
import hashlib
import importlib
import inspect
import io
import json
import logging
import math
import multiprocessing
import os
import pathlib
import platform
import queue
import random
import re
import select
import shlex
import shutil
import signal
import site
import string
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import threading
import traceback
import urllib.request
import warnings
import zipfile
from collections import Counter, OrderedDict, namedtuple
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from ctypes.util import find_library
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import lru_cache, partial
from glob import glob
from pathlib import Path
from string import ascii_letters, digits, punctuation
from time import sleep, time
from typing import Any, Optional, Union
from urllib.parse import quote

from definers._constants import user_agents
from definers._system import log, run, write


def google_drive_download(id, dest, unzip=True):
    from googledrivedownloader import download_file_from_google_drive

    download_file_from_google_drive(
        file_id=id, dest_path=dest, unzip=unzip, showsize=False
    )


def linked_url(url):

    host = url.split("?")[0]
    if "?" in url:
        param = "?" + url.split("?")[1]
    else:
        param = ""

    html_string = f"""
         <!DOCTYPE html>
        <html>
            <head>
                <meta charset="UTF-8">
                <base href="{host}" target="_top">
                <a href="{param}"></a>
            </head>
            <body onload='document.querySelector("a").click()'></body>
        </html>
    """

    html_bytes = html_string.encode("utf-8")

    base64_encoded_html = base64.b64encode(html_bytes).decode("utf-8")

    data_url = f"data:text/html;charset=utf-8;base64,{base64_encoded_html}"

    return data_url


def geo_new_york():
    return {
        "latitude": random.uniform(40.5, 40.9),
        "longitude": random.uniform(-74.2, -73.7),
    }


def extract_text(url, selector):

    from lxml.cssselect import CSSSelector
    from lxml.html import fromstring
    from playwright.sync_api import expect, sync_playwright

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

    html = fromstring(html_string)
    elems = html.xpath(xpath)
    elems = [
        el.text_content().strip() for el in elems if el.text_content().strip()
    ]
    if len(elems) == 0:
        return ""
    return elems[0]


def download_file(url, destination):
    import requests

    try:
        print(f"Downloading from {url} to {destination}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download successful.")
        return destination
    except Exception as e:
        print(f"Error downloading file: {e}")
        return None


def download_and_unzip(url, extract_to):
    import requests

    try:
        print(f"Downloading from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            print(f"Extracting to {extract_to}...")
            z.extractall(extract_to)
        print("Download and extraction successful.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return False


def add_to_path_windows(folder_path):
    print(f"Adding {folder_path} to user PATH...")
    command = f'setx PATH "{folder_path};%PATH%"'
    result = run(command)
    if result:
        print(
            f"Successfully added {folder_path} to PATH. Please restart your terminal for changes to take effect."
        )
    else:
        print(f"Failed to add {folder_path} to PATH.")
