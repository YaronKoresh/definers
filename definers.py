from enum import Enum
import getpass
import sysconfig
import base64
import select
import os
import pathlib
import importlib
from glob import glob
from pathlib import Path
import sys
import random
import string
from string import ascii_letters, digits, punctuation
import shutil
from datetime import datetime
import argparse
import multiprocessing
import threading
from time import sleep
import warnings
import logging
from time import time
import signal
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from collections import namedtuple
from dataclasses import dataclass
import re
import subprocess
import tempfile
from contextlib import contextmanager
import shlex
import json
import math
import platform
import traceback
import site
import queue
import urllib.request
import asyncio
import concurrent
from concurrent.futures import ProcessPoolExecutor
from urllib.parse import quote
import ctypes
from ctypes.util import find_library
import io
import tarfile
import hashlib

import collections
import collections.abc
collections.MutableSequence = collections.abc.MutableSequence

def init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(console_handler)
    return logger

logger = init_logger()

def init_cupy_numpy():
    import numpy as _np
    try:
        import cupy as np
    except Exception as e:
        import numpy as np
    if not hasattr(np,"float"):
        np.float = np.float64
    if not hasattr(np,"int"):
        np.int = np.int64
    return np, _np

np, _np = init_cupy_numpy()

language_codes = {'af': 'afrikaans', 'sq': 'albanian', 'am': 'amharic', 'ar': 'arabic', 'hy': 'armenian', 'as': 'assamese', 'ay': 'aymara', 'az': 'azerbaijani', 'bm': 'bambara', 'eu': 'basque', 'be': 'belarusian', 'bn': 'bengali', 'bho': 'bhojpuri', 'bs': 'bosnian', 'bg': 'bulgarian', 'ca': 'catalan', 'ceb': 'cebuano', 'ny': 'chichewa', 'zh-CN': 'chinese (simplified)', 'zh-TW': 'chinese (traditional)', 'co': 'corsican', 'hr': 'croatian', 'cs': 'czech', 'da': 'danish', 'dv': 'dhivehi', 'doi': 'dogri', 'nl': 'dutch', 'en': 'english', 'eo': 'esperanto', 'et': 'estonian', 'ee': 'ewe', 'tl': 'filipino', 'fi': 'finnish', 'fr': 'french', 'fy': 'frisian', 'gl': 'galician', 'ka': 'georgian', 'de': 'german', 'el': 'greek', 'gn': 'guarani', 'gu': 'gujarati', 'ht': 'haitian creole', 'ha': 'hausa', 'haw': 'hawaiian', 'iw': 'hebrew', 'he': 'hebrew', 'hi': 'hindi', 'hmn': 'hmong', 'hu': 'hungarian', 'is': 'icelandic', 'ig': 'igbo', 'ilo': 'ilocano', 'id': 'indonesian', 'ga': 'irish', 'it': 'italian', 'ja': 'japanese', 'jw': 'javanese', 'kn': 'kannada', 'kk': 'kazakh', 'km': 'khmer', 'rw': 'kinyarwanda', 'gom': 'konkani', 'ko': 'korean', 'kri': 'krio', 'ku': 'kurdish (kurmanji)', 'ckb': 'kurdish (sorani)', 'ky': 'kyrgyz', 'lo': 'lao', 'la': 'latin', 'lv': 'latvian', 'ln': 'lingala', 'lt': 'lithuanian', 'lg': 'luganda', 'lb': 'luxembourgish', 'mk': 'macedonian', 'mai': 'maithili', 'mg': 'malagasy', 'ms': 'malay', 'ml': 'malayalam', 'mt': 'maltese', 'mi': 'maori', 'mr': 'marathi', 'mni-Mtei': 'meiteilon (manipuri)', 'lus': 'mizo', 'mn': 'mongolian', 'my': 'myanmar', 'ne': 'nepali', 'no': 'norwegian', 'or': 'odia (oriya)', 'om': 'oromo', 'ps': 'pashto', 'fa': 'persian', 'pl': 'polish', 'pt': 'portuguese', 'pa': 'punjabi', 'qu': 'quechua', 'ro': 'romanian', 'ru': 'russian', 'sm': 'samoan', 'sa': 'sanskrit', 'gd': 'scots gaelic', 'nso': 'sepedi', 'sr': 'serbian', 'st': 'sesotho', 'sn': 'shona', 'sd': 'sindhi', 'si': 'sinhala', 'sk': 'slovak', 'sl': 'slovenian', 'so': 'somali', 'es': 'spanish', 'su': 'sundanese', 'sw': 'swahili', 'sv': 'swedish', 'tg': 'tajik', 'ta': 'tamil', 'tt': 'tatar', 'te': 'telugu', 'th': 'thai', 'ti': 'tigrinya', 'ts': 'tsonga', 'tr': 'turkish', 'tk': 'turkmen', 'ak': 'twi', 'uk': 'ukrainian', 'ur': 'urdu', 'ug': 'uyghur', 'uz': 'uzbek', 'vi': 'vietnamese', 'cy': 'welsh', 'xh': 'xhosa', 'yi': 'yiddish', 'yo': 'yoruba', 'zu': 'zulu'}

FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"

SYSTEM_MESSAGE = "You are a helpful and concise AI assistant. Provide accurate and relevant information to the user's queries in a friendly and clear manner."

tasks = {
    "video": "tencent/HunyuanVideo",
    "image": "black-forest-labs/FLUX.1-shnelkl",
    "detect": "facebook/detr-resnet-50",
    "answer": "microsoft/Phi-4-multimodal-instruct",
    "summary": "t5-large",
    "music": "facebook/musicgen-small",
    "speech-recognition": "openai/whisper-large-v3",
    "audio-classification": "MIT/ast-finetuned-audioset-10-10-0.4593",
}

MODELS = {
    "video": None,
    "image": None,
    "upscale": None,
    "detect": None,
    "answer": None,
    "summary": None,
    "music": None,
    "speech-recognition": None,
    "audio-classification": None,
    "tts": None,
}

TOKENIZERS = {
    "summary": None,
}

PROCESSORS = {
    "answer": None,
    "music": None,
}

CONFIGS = {
    "answer": None,
}

user_agents = {
    "chrome": [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    ],
    "firefox": [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:126.0) Gecko/20100101 Firefox/126.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:125.0) Gecko/20100101 Firefox/125.0',
    ],
    "safari": [
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
        'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148b Safari/604.1',
        'Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148b Safari/604.1',
    ],
    "egde": [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0',
    ],
    "opera": [
        'Opera/9.80 (Windows NT 6.0) Presto/2.12.388 Version/12.14',
        'Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.9.168 Version/11.52',
    ],
    "chrome-mobile": [
        'Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36',
        'Mozilla/5.0 (Linux; Android 13; Pixel 7 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36',
        'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/120.0.6099.119 Mobile/15E148b Safari/604.1',
    ],
    "safari-mobile": [
        'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148b Safari/604.1',
        'Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148b Safari/604.1',
    ]
}

iio_formats = ["png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif"]

common_audio_formats = [
    "wav",
    "mp3",
    "flac",
    "aac",
    "ogg",
    "opus",
    "aiff",
    "m4a",
    "wma",
]

punc = r'["!#$%&()*+,\./:;<=>?@\[\\\]^_`\{\|\}~]'

durl_empty = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAoAAAAKACAYAAAAMzckjAAAAAXNSR0IArs4c6QAAIABJREFUeF7t1kEBAAAIAjHpX9ogNxswfLBzBAgQIECAAAECKYGl0gpLgAABAgQIECBwBqAnIECAAAECBAjEBAzAWOHiEiBAgAABAgQMQD9AgAABAgQIEIgJGICxwsUlQIAAAQIECBiAfoAAAQIECBAgEBMwAGOFi0uAAAECBAgQMAD9AAECBAgQIEAgJmAAxgoXlwABAgQIECBgAPoBAgQIECBAgEBMwACMFS4uAQIECBAgQMAA9AMECBAgQIAAgZiAARgrXFwCBAgQIECAgAHoBwgQIECAAAECMQEDMFa4uAQIECBAgAABA9APECBAgAABAgRiAgZgrHBxCRAgQIAAAQIGoB8gQIAAAQIECMQEDMBY4eISIECAAAECBAxAP0CAAAECBAgQiAkYgLHCxSVAgAABAgQIGIB+gAABAgQIECAQEzAAY4WLS4AAAQIECBAwAP0AAQIECBAgQCAmYADGCheXAAECBAgQIGAA+gECBAgQIECAQEzAAIwVLi4BAgQIECBAwAD0AwQIECBAgACBmIABGCtcXAIECBAgQICAAegHCBAgQIAAAQIxAQMwVri4BAgQIECAAAED0A8QIECAAAECBGICBmCscHEJECBAgAABAgagHyBAgAABAgQIxAQMwFjh4hIgQIAAAQIEDEA/QIAAAQIECBCICRiAscLFJUCAAAECBAgYgH6AAAECBAgQIBATMABjhYtLgAABAgQIEDAA/QABAgQIECBAICZgAMYKF5cAAQIECBAgYAD6AQIECBAgQIBATMAAjBUuLgECBAgQIEDAAPQDBAgQIECAAIGYgAEYK1xcAgQIECBAgIAB6AcIECBAgAABAjEBAzBWuLgECBAgQIAAAQPQDxAgQIAAAQIEYgIGYKxwcQkQIECAAAECBqAfIECAAAECBAjEBAzAWOHiEiBAgAABAgQMQD9AgAABAgQIEIgJGICxwsUlQIAAAQIECBiAfoAAAQIECBAgEBMwAGOFi0uAAAECBAgQMAD9AAECBAgQIEAgJmAAxgoXlwABAgQIECBgAPoBAgQIECBAgEBMwACMFS4uAQIECBAgQMAA9AMECBAgQIAAgZiAARgrXFwCBAgQIECAgAHoBwgQIECAAAECMQEDMFa4uAQIECBAgAABA9APECBAgAABAgRiAgZgrHBxCRAgQIAAAQIGoB8gQIAAAQIECMQEDMBY4eISIECAAAECBAxAP0CAAAECBAgQiAkYgLHCxSVAgAABAgQIGIB+gAABAgQIECAQEzAAY4WLS4AAAQIECBAwAP0AAQIECBAgQCAmYADGCheXAAECBAgQIGAA+gECBAgQIECAQEzAAIwVLi4BAgQIECBAwAD0AwQIECBAgACBmIABGCtcXAIECBAgQICAAegHCBAgQIAAAQIxAQMwVri4BAgQIECAAAED0A8QIECAAAECBGICBmCscHEJECBAgAABAgagHyBAgAABAgQIxAQMwFjh4hIgQIAAAQIEDEA/QIAAAQIECBCICRiAscLFJUCAAAECBAgYgH6AAAECBAgQIBATMABjhYtLgAABAgQIEDAA/QABAgQIECBAICZgAMYKF5cAAQIECBAgYAD6AQIECBAgQIBATMAAjBUuLgECBAgQIEDAAPQDBAgQIECAAIGYgAEYK1xcAgQIECBAgIAB6AcIECBAgAABAjEBAzBWuLgECBAgQIAAAQPQDxAgQIAAAQIEYgIGYKxwcQkQIECAAAECBqAfIECAAAECBAjEBAzAWOHiEiBAgAABAgQMQD9AgAABAgQIEIgJGICxwsUlQIAAAQIECBiAfoAAAQIECBAgEBMwAGOFi0uAAAECBAgQMAD9AAECBAgQIEAgJmAAxgoXlwABAgQIECBgAPoBAgQIECBAgEBMwACMFS4uAQIECBAgQMAA9AMECBAgQIAAgZiAARgrXFwCBAgQIECAgAHoBwgQIECAAAECMQEDMFa4uAQIECBAgAABA9APECBAgAABAgRiAgZgrHBxCRAgQIAAAQIGoB8gQIAAAQIECMQEDMBY4eISIECAAAECBAxAP0CAAAECBAgQiAkYgLHCxSVAgAABAgQIGIB+gAABAgQIECAQEzAAY4WLS4AAAQIECBAwAP0AAQIECBAgQCAmYADGCheXAAECBAgQIGAA+gECBAgQIECAQEzAAIwVLi4BAgQIECBAwAD0AwQIECBAgACBmIABGCtcXAIECBAgQICAAegHCBAgQIAAAQIxAQMwVri4BAgQIECAAAED0A8QIECAAAECBGICBmCscHEJECBAgAABAgagHyBAgAABAgQIxAQMwFjh4hIgQIAAAQIEDEA/QIAAAQIECBCICRiAscLFJUCAAAECBAgYgH6AAAECBAgQIBATMABjhYtLgAABAgQIEDAA/QABAgQIECBAICZgAMYKF5cAAQIECBAgYAD6AQIECBAgQIBATMAAjBUuLgECBAgQIEDAAPQDBAgQIECAAIGYgAEYK1xcAgQIECBAgIAB6AcIECBAgAABAjEBAzBWuLgECBAgQIAAAQPQDxAgQIAAAQIEYgIGYKxwcQkQIECAAAECBqAfIECAAAECBAjEBAzAWOHiEiBAgAABAgQMQD9AgAABAgQIEIgJGICxwsUlQIAAAQIECBiAfoAAAQIECBAgEBMwAGOFi0uAAAECBAgQMAD9AAECBAgQIEAgJmAAxgoXlwABAgQIECBgAPoBAgQIECBAgEBMwACMFS4uAQIECBAgQMAA9AMECBAgQIAAgZiAARgrXFwCBAgQIECAgAHoBwgQIECAAAECMQEDMFa4uAQIECBAgAABA9APECBAgAABAgRiAgZgrHBxCRAgQIAAAQIGoB8gQIAAAQIECMQEDMBY4eISIECAAAECBAxAP0CAAAECBAgQiAkYgLHCxSVAgAABAgQIGIB+gAABAgQIECAQEzAAY4WLS4AAAQIECBAwAP0AAQIECBAgQCAmYADGCheXAAECBAgQIGAA+gECBAgQIECAQEzAAIwVLi4BAgQIECBAwAD0AwQIECBAgACBmIABGCtcXAIECBAgQICAAegHCBAgQIAAAQIxAQMwVri4BAgQIECAAAED0A8QIECAAAECBGICBmCscHEJECBAgAABAgagHyBAgAABAgQIxAQMwFjh4hIgQIAAAQIEDEA/QIAAAQIECBCICRiAscLFJUCAAAECBAgYgH6AAAECBAgQIBATMABjhYtLgAABAgQIEDAA/QABAgQIECBAICZgAMYKF5cAAQIECBAgYAD6AQIECBAgQIBATMAAjBUuLgECBAgQIEDAAPQDBAgQIECAAIGYgAEYK1xcAgQIECBAgIAB6AcIECBAgAABAjEBAzBWuLgECBAgQIAAAQPQDxAgQIAAAQIEYgIGYKxwcQkQIECAAAECBqAfIECAAAECBAjEBAzAWOHiEiBAgAABAgQMQD9AgAABAgQIEIgJGICxwsUlQIAAAQIECBiAfoAAAQIECBAgEBMwAGOFi0uAAAECBAgQMAD9AAECBAgQIEAgJmAAxgoXlwABAgQIECBgAPoBAgQIECBAgEBMwACMFS4uAQIECBAgQMAA9AMECBAgQIAAgZiAARgrXFwCBAgQIECAgAHoBwgQIECAAAECMQEDMFa4uAQIECBAgAABA9APECBAgAABAgRiAgZgrHBxCRAgQIAAAQIGoB8gQIAAAQIECMQEDMBY4eISIECAAAECBAxAP0CAAAECBAgQiAkYgLHCxSVAgAABAgQIGIB+gAABAgQIECAQEzAAY4WLS4AAAQIECBAwAP0AAQIECBAgQCAmYADGCheXAAECBAgQIGAA+gECBAgQIECAQEzAAIwVLi4BAgQIECBAwAD0AwQIECBAgACBmIABGCtcXAIECBAgQICAAegHCBAgQIAAAQIxAQMwVri4BAgQIECAAAED0A8QIECAAAECBGICBmCscHEJECBAgAABAgagHyBAgAABAgQIxAQMwFjh4hIgQIAAAQIEDEA/QIAAAQIECBCICRiAscLFJUCAAAECBAgYgH6AAAECBAgQIBATMABjhYtLgAABAgQIEDAA/QABAgQIECBAICZgAMYKF5cAAQIECBAgYAD6AQIECBAgQIBATMAAjBUuLgECBAgQIEDAAPQDBAgQIECAAIGYgAEYK1xcAgQIECBAgIAB6AcIECBAgAABAjEBAzBWuLgECBAgQIAAAQPQDxAgQIAAAQIEYgIGYKxwcQkQIECAAAECBqAfIECAAAECBAjEBAzAWOHiEiBAgAABAgQMQD9AgAABAgQIEIgJGICxwsUlQIAAAQIECBiAfoAAAQIECBAgEBMwAGOFi0uAAAECBAgQMAD9AAECBAgQIEAgJmAAxgoXlwABAgQIECBgAPoBAgQIECBAgEBMwACMFS4uAQIECBAgQMAA9AMECBAgQIAAgZiAARgrXFwCBAgQIECAgAHoBwgQIECAAAECMQEDMFa4uAQIECBAgAABA9APECBAgAABAgRiAgZgrHBxCRAgQIAAAQIGoB8gQIAAAQIECMQEDMBY4eISIECAAAECBAxAP0CAAAECBAgQiAkYgLHCxSVAgAABAgQIGIB+gAABAgQIECAQEzAAY4WLS4AAAQIECBAwAP0AAQIECBAgQCAmYADGCheXAAECBAgQIGAA+gECBAgQIECAQEzAAIwVLi4BAgQIECBAwAD0AwQIECBAgACBmIABGCtcXAIECBAgQICAAegHCBAgQIAAAQIxAQMwVri4BAgQIECAAAED0A8QIECAAAECBGICBmCscHEJECBAgAABAgagHyBAgAABAgQIxAQMwFjh4hIgQIAAAQIEDEA/QIAAAQIECBCICRiAscLFJUCAAAECBAgYgH6AAAECBAgQIBATMABjhYtLgAABAgQIEDAA/QABAgQIECBAICZgAMYKF5cAAQIECBAgYAD6AQIECBAgQIBATMAAjBUuLgECBAgQIEDAAPQDBAgQIECAAIGYgAEYK1xcAgQIECBAgIAB6AcIECBAgAABAjEBAzBWuLgECBAgQIAAAQPQDxAgQIAAAQIEYgIGYKxwcQkQIECAAAECBqAfIECAAAECBAjEBAzAWOHiEiBAgAABAgQMQD9AgAABAgQIEIgJGICxwsUlQIAAAQIECBiAfoAAAQIECBAgEBMwAGOFi0uAAAECBAgQMAD9AAECBAgQIEAgJmAAxgoXlwABAgQIECBgAPoBAgQIECBAgEBMwACMFS4uAQIECBAgQMAA9AMECBAgQIAAgZiAARgrXFwCBAgQIECAgAHoBwgQIECAAAECMQEDMFa4uAQIECBAgAABA9APECBAgAABAgRiAgZgrHBxCRAgQIAAAQIGoB8gQIAAAQIECMQEDMBY4eISIECAAAECBAxAP0CAAAECBAgQiAkYgLHCxSVAgAABAgQIGIB+gAABAgQIECAQEzAAY4WLS4AAAQIECBAwAP0AAQIECBAgQCAmYADGCheXAAECBAgQIGAA+gECBAgQIECAQEzAAIwVLi4BAgQIECBAwAD0AwQIECBAgACBmIABGCtcXAIECBAgQICAAegHCBAgQIAAAQIxAQMwVri4BAgQIECAAAED0A8QIECAAAECBGICBmCscHEJECBAgAABAgagHyBAgAABAgQIxAQMwFjh4hIgQIAAAQIEDEA/QIAAAQIECBCICRiAscLFJUCAAAECBAgYgH6AAAECBAgQIBATMABjhYtLgAABAgQIEDAA/QABAgQIECBAICZgAMYKF5cAAQIECBAgYAD6AQIECBAgQIBATMAAjBUuLgECBAgQIEDAAPQDBAgQIECAAIGYgAEYK1xcAgQIECBAgIAB6AcIECBAgAABAjEBAzBWuLgECBAgQIAAAQPQDxAgQIAAAQIEYgIGYKxwcQkQIECAAAECBqAfIECAAAECBAjEBAzAWOHiEiBAgAABAgQMQD9AgAABAgQIEIgJGICxwsUlQIAAAQIECBiAfoAAAQIECBAgEBMwAGOFi0uAAAECBAgQMAD9AAECBAgQIEAgJmAAxgoXlwABAgQIECBgAPoBAgQIECBAgEBMwACMFS4uAQIECBAgQMAA9AMECBAgQIAAgZiAARgrXFwCBAgQIECAgAHoBwgQIECAAAECMQEDMFa4uAQIECBAgAABA9APECBAgAABAgRiAgZgrHBxCRAgQIAAAQIGoB8gQIAAAQIECMQEDMBY4eISIECAAAECBAxAP0CAAAECBAgQiAkYgLHCxSVAgAABAgQIGIB+gAABAgQIECAQEzAAY4WLS4AAAQIECBAwAP0AAQIECBAgQCAmYADGCheXAAECBAgQIGAA+gECBAgQIECAQEzAAIwVLi4BAgQIECBAwAD0AwQIECBAgACBmIABGCtcXAIECBAgQICAAegHCBAgQIAAAQIxAQMwVri4BAgQIECAAAED0A8QIECAAAECBGICBmCscHEJECBAgAABAgagHyBAgAABAgQIxAQMwFjh4hIgQIAAAQIEDEA/QIAAAQIECBCICRiAscLFJUCAAAECBAgYgH6AAAECBAgQIBATMABjhYtLgAABAgQIEDAA/QABAgQIECBAICZgAMYKF5cAAQIECBAgYAD6AQIECBAgQIBATMAAjBUuLgECBAgQIEDAAPQDBAgQIECAAIGYgAEYK1xcAgQIECBAgIAB6AcIECBAgAABAjEBAzBWuLgECBAgQIAAAQPQDxAgQIAAAQIEYgIGYKxwcQkQIECAAAECBqAfIECAAAECBAjEBAzAWOHiEiBAgAABAgQMQD9AgAABAgQIEIgJGICxwsUlQIAAAQIECBiAfoAAAQIECBAgEBMwAGOFi0uAAAECBAgQMAD9AAECBAgQIEAgJmAAxgoXlwABAgQIECBgAPoBAgQIECBAgEBMwACMFS4uAQIECBAgQMAA9AMECBAgQIAAgZiAARgrXFwCBAgQIECAgAHoBwgQIECAAAECMQEDMFa4uAQIECBAgAABA9APECBAgAABAgRiAgZgrHBxCRAgQIAAAQIGoB8gQIAAAQIECMQEDMBY4eISIECAAAECBAxAP0CAAAECBAgQiAkYgLHCxSVAgAABAgQIGIB+gAABAgQIECAQEzAAY4WLS4AAAQIECBAwAP0AAQIECBAgQCAmYADGCheXAAECBAgQIGAA+gECBAgQIECAQEzAAIwVLi4BAgQIECBAwAD0AwQIECBAgACBmIABGCtcXAIECBAgQICAAegHCBAgQIAAAQIxAQMwVri4BAgQIECAAAED0A8QIECAAAECBGICBmCscHEJECBAgAABAgagHyBAgAABAgQIxAQMwFjh4hIgQIAAAQIEDEA/QIAAAQIECBCICRiAscLFJUCAAAECBAgYgH6AAAECBAgQIBATMABjhYtLgAABAgQIEDAA/QABAgQIECBAICZgAMYKF5cAAQIECBAgYAD6AQIECBAgQIBATMAAjBUuLgECBAgQIEDAAPQDBAgQIECAAIGYgAEYK1xcAgQIECBAgIAB6AcIECBAgAABAjEBAzBWuLgECBAgQIAAAQPQDxAgQIAAAQIEYgIGYKxwcQkQIECAAAECBqAfIECAAAECBAjEBAzAWOHiEiBAgAABAgQMQD9AgAABAgQIEIgJGICxwsUlQIAAAQIECBiAfoAAAQIECBAgEBMwAGOFi0uAAAECBAgQMAD9AAECBAgQIEAgJmAAxgoXlwABAgQIECBgAPoBAgQIECBAgEBMwACMFS4uAQIECBAgQMAA9AMECBAgQIAAgZiAARgrXFwCBAgQIECAgAHoBwgQIECAAAECMQEDMFa4uAQIECBAgAABA9APECBAgAABAgRiAgZgrHBxCRAgQIAAAQIGoB8gQIAAAQIECMQEDMBY4eISIECAAAECBAxAP0CAAAECBAgQiAkYgLHCxSVAgAABAgQIGIB+gAABAgQIECAQEzAAY4WLS4AAAQIECBAwAP0AAQIECBAgQCAmYADGCheXAAECBAgQIGAA+gECBAgQIECAQEzAAIwVLi4BAgQIECBAwAD0AwQIECBAgACBmIABGCtcXAIECBAgQICAAegHCBAgQIAAAQIxAQMwVri4BAgQIECAAAED0A8QIECAAAECBGICBmCscHEJECBAgAABAgagHyBAgAABAgQIxAQMwFjh4hIgQIAAAQIEDEA/QIAAAQIECBCICRiAscLFJUCAAAECBAgYgH6AAAECBAgQIBATMABjhYtLgAABAgQIEDAA/QABAgQIECBAICZgAMYKF5cAAQIECBAgYAD6AQIECBAgQIBATMAAjBUuLgECBAgQIEDAAPQDBAgQIECAAIGYgAEYK1xcAgQIECBAgIAB6AcIECBAgAABAjEBAzBWuLgECBAgQIAAAQPQDxAgQIAAAQIEYgIGYKxwcQkQIECAAAECBqAfIECAAAECBAjEBAzAWOHiEiBAgAABAgQMQD9AgAABAgQIEIgJGICxwsUlQIAAAQIECBiAfoAAAQIECBAgEBMwAGOFi0uAAAECBAgQMAD9AAECBAgQIEAgJmAAxgoXlwABAgQIECBgAPoBAgQIECBAgEBMwACMFS4uAQIECBAgQMAA9AMECBAgQIAAgZiAARgrXFwCBAgQIECAgAHoBwgQIECAAAECMQEDMFa4uAQIECBAgAABA9APECBAgAABAgRiAgZgrHBxCRAgQIAAAQIGoB8gQIAAAQIECMQEDMBY4eISIECAAAECBAxAP0CAAAECBAgQiAkYgLHCxSVAgAABAgQIGIB+gAABAgQIECAQEzAAY4WLS4AAAQIECBAwAP0AAQIECBAgQCAmYADGCheXAAECBAgQIGAA+gECBAgQIECAQEzAAIwVLi4BAgQIECBAwAD0AwQIECBAgACBmIABGCtcXAIECBAgQICAAegHCBAgQIAAAQIxAQMwVri4BAgQIECAAAED0A8QIECAAAECBGICBmCscHEJECBAgAABAgagHyBAgAABAgQIxAQMwFjh4hIgQIAAAQIEDEA/QIAAAQIECBCICRiAscLFJUCAAAECBAgYgH6AAAECBAgQIBATMABjhYtLgAABAgQIEDAA/QABAgQIECBAICZgAMYKF5cAAQIECBAgYAD6AQIECBAgQIBATMAAjBUuLgECBAgQIEDAAPQDBAgQIECAAIGYgAEYK1xcAgQIECBAgIAB6AcIECBAgAABAjEBAzBWuLgECBAgQIAAAQPQDxAgQIAAAQIEYgIGYKxwcQkQIECAAAECBqAfIECAAAECBAjEBAzAWOHiEiBAgAABAgQMQD9AgAABAgQIEIgJGICxwsUlQIAAAQIECBiAfoAAAQIECBAgEBMwAGOFi0uAAAECBAgQMAD9AAECBAgQIEAgJmAAxgoXlwABAgQIECBgAPoBAgQIECBAgEBMwACMFS4uAQIECBAgQMAA9AMECBAgQIAAgZiAARgrXFwCBAgQIECAgAHoBwgQIECAAAECMQEDMFa4uAQIECBAgAABA9APECBAgAABAgRiAgZgrHBxCRAgQIAAAQIGoB8gQIAAAQIECMQEDMBY4eISIECAAAECBAxAP0CAAAECBAgQiAkYgLHCxSVAgAABAgQIGIB+gAABAgQIECAQEzAAY4WLS4AAAQIECBAwAP0AAQIECBAgQCAmYADGCheXAAECBAgQIGAA+gECBAgQIECAQEzAAIwVLi4BAgQIECBAwAD0AwQIECBAgACBmIABGCtcXAIECBAgQICAAegHCBAgQIAAAQIxAQMwVri4BAgQIECAAAED0A8QIECAAAECBGICBmCscHEJECBAgAABAgagHyBAgAABAgQIxAQMwFjh4hIgQIAAAQIEDEA/QIAAAQIECBCICRiAscLFJUCAAAECBAgYgH6AAAECBAgQIBATMABjhYtLgAABAgQIEDAA/QABAgQIECBAICZgAMYKF5cAAQIECBAgYAD6AQIECBAgQIBATMAAjBUuLgECBAgQIEDAAPQDBAgQIECAAIGYgAEYK1xcAgQIECBAgIAB6AcIECBAgAABAjEBAzBWuLgECBAgQIAAAQPQDxAgQIAAAQIEYgIGYKxwcQkQIECAAAECBqAfIECAAAECBAjEBAzAWOHiEiBAgAABAgQMQD9AgAABAgQIEIgJGICxwsUlQIAAAQIECBiAfoAAAQIECBAgEBMwAGOFi0uAAAECBAgQMAD9AAECBAgQIEAgJmAAxgoXlwABAgQIECBgAPoBAgQIECBAgEBMwACMFS4uAQIECBAgQMAA9AMECBAgQIAAgZiAARgrXFwCBAgQIECAgAHoBwgQIECAAAECMQEDMFa4uAQIECBAgAABA9APECBAgAABAgRiAgZgrHBxCRAgQIAAAQIGoB8gQIAAAQIECMQEDMBY4eISIECAAAECBAxAP0CAAAECBAgQiAkYgLHCxSVAgAABAgQIGIB+gAABAgQIECAQEzAAY4WLS4AAAQIECBAwAP0AAQIECBAgQCAmYADGCheXAAECBAgQIGAA+gECBAgQIECAQEzAAIwVLi4BAgQIECBAwAD0AwQIECBAgACBmIABGCtcXAIECBAgQICAAegHCBAgQIAAAQIxAQMwVri4BAgQIECAAAED0A8QIECAAAECBGICBmCscHEJECBAgAABAgagHyBAgAABAgQIxAQMwFjh4hIgQIAAAQIEDEA/QIAAAQIECBCICRiAscLFJUCAAAECBAgYgH6AAAECBAgQIBATMABjhYtLgAABAgQIEDAA/QABAgQIECBAICZgAMYKF5cAAQIECBAgYAD6AQIECBAgQIBATMAAjBUuLgECBAgQIEDAAPQDBAgQIECAAIGYgAEYK1xcAgQIECBAgIAB6AcIECBAgAABAjEBAzBWuLgECBAgQIAAAQPQDxAgQIAAAQIEYgIGYKxwcQkQIEB6uwAFAAAAwElEQVSAAAECBqAfIECAAAECBAjEBAzAWOHiEiBAgAABAgQMQD9AgAABAgQIEIgJGICxwsUlQIAAAQIECBiAfoAAAQIECBAgEBMwAGOFi0uAAAECBAgQMAD9AAECBAgQIEAgJmAAxgoXlwABAgQIECBgAPoBAgQIECBAgEBMwACMFS4uAQIECBAgQMAA9AMECBAgQIAAgZiAARgrXFwCBAgQIECAgAHoBwgQIECAAAECMQEDMFa4uAQIECBAgACBB3r2AoFVBl6GAAAAAElFTkSuQmCC"

negative_keywords = [
    # Quality & Style
    "low quality", "worst quality", "lowres", "jpeg artifacts", "blurry", "noisy", "pixelated",
    "watermark", "signature", "username", "text", "error", "out of frame", "cropped",
    "ugly", "disgusting", "horrific", "scary", "creepy",
    # Anatomy & Body
    "malformed", "disfigured", "deformed", "mutated", "mutation", "extra limbs", "missing limbs",
    "extra fingers", "missing fingers", "fewer digits", "bad anatomy", "poorly drawn hands",
    "poorly drawn face", "mangled", "cloned face", "bad proportions", "fused fingers",
    # Composition & Scene
    "static position", "same frames", "boring", "uninteresting", "illogical", "unreasonable scenario",
    "weird scenario", "disconnected", "disjointed", "tiling", "asymmetrical",
    # Duplication & Repetition
    "duplicate", "cloned", "multiple views", "grid", "collage", "split screen"
]
random.shuffle(negative_keywords)
_negative_prompt_ = ", ".join(negative_keywords)

_base_prompt_ = "cinematic masterpiece, ultra realistic, 8k, best quality, sharp focus, professional color grading"

def get_os_name():
    return platform.system().lower()

def is_admin_windows():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def _install_ffmpeg_windows():
    import requests
    import zipfile

    print("[INFO] Running FFmpeg installer for Windows...")

    if not is_admin_windows():
        print("[ERROR] This script requires Administrator privileges to run on Windows.")
        print("[INFO] Please re-run this script from a terminal with Administrator rights.")
        sys.exit(1)

    print("\n[INFO] Attempting to install using Winget (Windows Package Manager)...")
    try:
        result = subprocess.run(
            [
                "winget", "install", "--id=Gyan.FFmpeg.Essentials", "-e",
                "--accept-source-agreements", "--accept-package-agreements"
            ],
            check=True,
            capture_output=True,
            text=True
        )
        print("[SUCCESS] FFmpeg has been installed via Winget.")
        print("[INFO] You may need to restart your terminal for the PATH changes to take effect.")
        return
    except FileNotFoundError:
        print("[WARN] Winget command not found. It might not be installed or in the PATH.")
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Winget installation failed with exit code {e.returncode}.")
        print(f"[DEBUG] Winget stderr: {e.stderr}")

    print("\n[INFO] Winget installation failed or was not available. Attempting manual download...")

    temp_dir = tempfile.gettempdir()
    zip_path = os.path.join(temp_dir, "ffmpeg.zip")
    extract_path = os.path.join(temp_dir, "ffmpeg_extracted")
    program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
    ffmpeg_install_dir = os.path.join(program_files, "ffmpeg")

    try:
        print(f"[INFO] Downloading latest FFmpeg essentials build from {FFMPEG_URL}...")
        with requests.get(FFMPEG_URL, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("[SUCCESS] Download complete.")

        print(f"[INFO] Extracting FFmpeg to {extract_path}...")
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("[SUCCESS] Extraction complete.")

        extracted_files = os.listdir(extract_path)
        if not extracted_files:
            raise IOError("Extraction failed, no files found in temporary directory.")
        
        ffmpeg_build_dir = os.path.join(extract_path, extracted_files[0])
        ffmpeg_bin_dir = os.path.join(ffmpeg_build_dir, "bin")

        print(f"[INFO] Moving FFmpeg binaries to {ffmpeg_install_dir}...")
        if os.path.exists(ffmpeg_install_dir):
            shutil.rmtree(ffmpeg_install_dir)
        shutil.move(ffmpeg_bin_dir, ffmpeg_install_dir)
        print("[SUCCESS] Binaries moved.")

        print("[INFO] Adding FFmpeg to the system PATH...")
        subprocess.run(["setx", "/M", "PATH", f"%PATH%;{ffmpeg_install_dir}"], check=True)
        print("[SUCCESS] FFmpeg added to system PATH.")
        print("[INFO] IMPORTANT: You must restart your terminal or PC for the new PATH to be recognized.")

    except Exception as e:
        print(f"\n[ERROR] An error occurred during manual installation: {e}")
        sys.exit(1)
    finally:
        print("[INFO] Cleaning up temporary files...")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
        print("[INFO] Cleanup complete.")

def _install_ffmpeg_linux():
    print("[INFO] Running FFmpeg installer for Linux...")
    
    if os.geteuid() != 0:
        print("[WARN] This script needs sudo privileges to install packages.")
        print("[INFO] It will likely prompt you for your password.")

    package_managers = {
        'apt': {
            'update_cmd': ['apt-get', 'update'],
            'install_cmd': ['apt-get', 'install', 'ffmpeg', '-y']
        },
        'dnf': {
            'install_cmd': ['dnf', 'install', 'ffmpeg', '-y']
        },
        'pacman': {
            'install_cmd': ['pacman', '-S', 'ffmpeg', '--noconfirm']
        }
    }

    selected_pm = None
    for pm in package_managers:
        if shutil.which(pm):
            selected_pm = pm
            break

    if not selected_pm:
        print("[ERROR] Could not detect a supported package manager (apt, dnf, pacman).")
        print("[INFO] Please install FFmpeg manually.")
        sys.exit(1)

    print(f"[INFO] Detected package manager: {selected_pm}")

    try:
        pm_cmds = package_managers[selected_pm]
        
        if 'update_cmd' in pm_cmds:
            print(f"[INFO] Running package list update ({selected_pm})...")
            subprocess.run(pm_cmds['update_cmd'], check=True)

        print(f"[INFO] Installing FFmpeg using {selected_pm}...")
        subprocess.run(pm_cmds['install_cmd'], check=True)
        
        print("\n[SUCCESS] FFmpeg installed successfully.")

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] The installation command failed with exit code {e.returncode}.")
        print("[INFO] Please check the output above for errors from the package manager.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        sys.exit(1)

def install_ffmpeg():
    if installed("ffmpeg"):
        return True
    system = get_os_name()
    if system == "windows":
        _install_ffmpeg_windows()
        return True
    elif system == "linux":
        _install_ffmpeg_linux()
        return True
    else:
        print(f"[ERROR] Unsupported operating system: {system}.")
        print("[INFO] This script only supports Windows and Linux.")
        sys.exit(1)

def install_audio_effects():
    os_name = get_os_name()
    install_dir = os.path.join(os.path.expanduser("~"), "app_dependencies")
    os.makedirs(install_dir, exist_ok=True)
    if os_name == "linux":
        print("Detected Linux. Installing system dependencies with apt-get...")
        dependencies_apt = ["rubberband-cli", "fluidsynth", "fluid-soundfont-gm", "build-essential"]
        run("apt-get update -y")
        run(f"apt-get install -y {' '.join(dependencies_apt)}")
    elif os_name == "windows":
        print("Detected Windows. Automating dependency installation...")
        print(f"Dependencies will be installed in: {install_dir}")
        rubberband_url = "https://breakfastquay.com/files/releases/rubberband-3.3.0-gpl-executable-windows.zip"
        fluidsynth_url = "https://github.com/FluidSynth/fluidsynth/releases/download/v2.3.5/fluidsynth-2.3.5-win64.zip"
        soundfont_url = "https://github.com/FluidSynth/fluidsynth/raw/master/sf2/FluidR3_GM.sf2"
        soundfont_path = os.path.join(install_dir, "soundfonts", "FluidR3_GM.sf2")
        rubberband_extract_path = os.path.join(install_dir, "rubberband")
        if not any("rubberband" in s for s in os.environ["PATH"]):
            if download_and_unzip(rubberband_url, rubberband_extract_path):
                extracted_dirs = [d for d in os.listdir(rubberband_extract_path) if os.path.isdir(os.path.join(rubberband_extract_path, d))]
                if extracted_dirs:
                    rubberband_bin_path = os.path.join(rubberband_extract_path, extracted_dirs[0])
                    add_to_path_windows(rubberband_bin_path)
        fluidsynth_extract_path = os.path.join(install_dir, "fluidsynth")
        if not any("fluidsynth" in s for s in os.environ["PATH"]):
            if download_and_unzip(fluidsynth_url, fluidsynth_extract_path):
                fluidsynth_bin_path = os.path.join(fluidsynth_extract_path, "bin")
                add_to_path_windows(fluidsynth_bin_path)
        if not os.path.exists(soundfont_path):
            os.makedirs(os.path.dirname(soundfont_path), exist_ok=True)
            print("Downloading SoundFont for MIDI playback...")
            download_file(soundfont_url, soundfont_path)
    else:
        print(f"Unsupported OS: {os_name}. Manual installation of system dependencies may be required.")
    print("\nInstalling Python packages with pip...")

def merge_system_message(data):
    text = ""
    for key, value in data.items():
        if isinstance(value, str):
            text += f"<|system|>\n{value}\n<|end|>\n"
        elif isinstance(value, list):
            for item in value:
                text += "<|system|>\n"
                for k, v in item.items():
                    if isinstance(v, dict):
                        for lang, content in v.items():
                            text += f"{lang}: {content}\n"
                    else:
                        text += f"{k}: {v}\n"
                text += "<|end|>\n"
    return text

def set_system_message(
    name: Optional[str] = None,
    role: Optional[str] = None,
    tone: Optional[str] = None,
    goals: Optional[List[str]] = None,
    chattiness: Optional[str] = None,
    persona_data: Optional[Dict[str, str]] = None,
    task_rules: Optional[List[str]] = None,
    interaction_style: Optional[str] = None,
    output_format: Optional[str] = None
):
    """
    Constructs and sets a comprehensive system message for the AI model globally.

    This function allows for detailed customization of the AI's persona, behavior,
    and output format by dynamically building a system prompt from the provided arguments.

    Args:
        name (Optional[str]): The name the AI should use for itself (e.g., 'Definer').
        role (Optional[str]): The primary role of the AI (e.g., 'a code assistant', 'a travel guide').
        tone (Optional[str]): The desired tone of voice (e.g., 'friendly and encouraging', 'formal and professional').
        goals (Optional[List[str]]): A list of the goals of the AI in the conversations (e.g., ['answer users questions', 'guide users with the application usage']).
        chattiness (Optional[str]): The desired level of verbosity (e.g., 'be concise', 'provide detailed explanations').
        persona_data (Optional[Dict[str, str]]): A dictionary of facts about the AI's persona (e.g., {"your creator": "John Doe"}).
        task_rules (Optional[List[str]]): A list of specific rules the AI must follow (e.g., ["Do not mention you are an AI."]).
        interaction_style (Optional[str]): Instructions on how to interact (e.g., 'ask clarifying questions before answering').
        output_format (Optional[str]): Specific instructions for the output structure (e.g., 'Respond only in JSON format').
    
    Returns:
        str: The newly constructed system message.
    """
    global _system_message

    message_parts = []

    if role:
        message_parts.append(f"You are {role}.")
    else:
        message_parts.append("You are a helpful AI assistant.")

    if name:
        message_parts.append(f"Your name is {name}.")

    style_instructions = []
    if tone:
        style_instructions.append(f"Your tone should be {tone}.")
    if chattiness:
        style_instructions.append(f"In terms of verbosity, {chattiness}.")
    if interaction_style:
        style_instructions.append(f"When interacting, {interaction_style}.")

    if style_instructions:
        message_parts.append(" ".join(style_instructions))

    if persona_data:
        persona_str = "Here is some information for you to learn and remember: "
        persona_facts = [f"{key} is {value}" for key, value in persona_data.items()]
        persona_str += "; ".join(persona_facts) + "."
        message_parts.append(persona_str)

    if goals:
        goals_str = "; ".join(goals) + "."
        message_parts.append(goals_str)

    if task_rules or output_format:
        rules_header = "You must strictly follow these rules:"
        rules_list = []
        if task_rules:
            rules_list.extend(task_rules)
        if output_format:
            rules_list.append(f"Your final output must be exclusively in the following format: {output_format}.")

        formatted_rules = "\n".join(f"{i+1}. {rule}" for i, rule in enumerate(rules_list))
        message_parts.append(f"{rules_header}\n{formatted_rules}")

    _system_message = "\n\n".join(message_parts)
    
    log("System Message Updated", _system_message)
    
    return _system_message

def answer(history: list):

    from PIL import Image
    import soundfile as sf

    internal = '<|system|>'
    human = '<|user|>'
    ai = '<|assistant|>'
    end = '<|end|>'
    img = '<|image_X|>'
    snd = '<|audio_X|>'

    messages = [merge_system_message(SYSTEM_MESSAGE)]

    img_list = []
    snd_list = []

    for h in history:

        if h["role"] == "assistant":
            messages.append(end)
            messages.append(ai)
        elif h["role"] == "user":
            messages.append(human)

        content = h["content"]
        if isinstance(content,dict) or isinstance(content,tuple):
            ps = []
            if isinstance(content,dict):
                ps = [content["path"]]
            else:
                ps = [c["path"] for c in content if isinstance(c,dict)]
            for p in ps:
                ext = p.split(".")[-1]
                if ext in common_audio_formats:
                    audio, samplerate = sf.read(audio_url)
                    snd_list.append( (audio, samplerate) )
                    messages.append(snd.replace( "X", str(len(snd_list)) ))
                if ext in iio_formats:
                    img_list.append( Image.open(p) )
                    messages.append(img.replace( "X", str(len(img_list)) ))
        else:
            messages.append(content.replace("|"," or "))

        messages.append(end)

    messages.append(ai)
    prompt = "".join(messages)

    log("Chat history",{
        "prompt": prompt,
        "audios": snd_list,
        "images": img_list,
    },status="")

    lsts = {}
    if len(snd_list) > 0:
        lsts["audios"]=snd_list
    if len(img_list) > 0:
        lsts["images"]=img_list

    response = MODELS["answer"].generate(prompt=prompt, max_length=200, beam_width=16, **lsts)
    return response

def linear_regression(X, y, learning_rate=0.01, epochs=50):
    m, n = X.shape

    weights = np.zeros(n)
    bias = 0

    for _ in range(epochs):
        y_pred = X @ weights + bias

        error = y_pred - y
        dw = (2/m) * X.T @ error
        db = (2/m) * np.sum(error)

        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias

def initialize_linear_regression(input_dim, model_path):
    import torch
    if os.path.exists(model_path):
        model_torch = LinearRegressionTorch(input_dim)
        model_torch.load_state_dict(torch.load(model_path))
        print("Loaded existing model.")
    else:
        model_torch = LinearRegressionTorch(input_dim)
        print("Created new model.")

    model_torch.to(device())
    return model_torch

def train_linear_regression(X, y, model_path, learning_rate=0.01):
    import torch
    model_torch = initialize_linear_regression(X.shape[1], model_path)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model_torch.parameters(), lr=learning_rate)

    d = device()
    X_torch = torch.tensor(X, dtype=torch.float32, device=d)
    y_torch = torch.tensor(y, dtype=torch.float32, device=d)

    y_pred = model_torch(X_torch).squeeze()
    loss = criterion(y_pred, y_torch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    torch.save(model_torch.state_dict(), model_path)
    print("Model saved.")

    return model_torch

def fetch_dataset(src, url_type=None, revision=None):
    import PIL
    PIL.__spec__ = PIL.Image.__spec__

    from datasets import load_dataset
    try:
        if revision:
            dataset = load_dataset(src, revision=revision, split="train")
        else:
            dataset = load_dataset(src, split="train")
    except FileNotFoundError:
        logging.error(f"Dataset {src} not found.")
        return None
    except ConnectionError:
        logging.error(f"Connection error while loading dataset {src}.")
        return None
    except Exception as e:
        logging.error(f"Error loading dataset {src}: {e}")
        if url_type:
            try:
                if revision:
                    dataset = load_dataset(url_type, data_files={"train": src}, revision=revision, split="train")
                else:
                    dataset = load_dataset(url_type, data_files={"train": src}, split="train")
            except FileNotFoundError:
                logging.error(f"Dataset {url_type} with data_files {src} not found.")
                return None
            except ConnectionError:
                logging.error(f"Connection error while loading dataset {url_type} with data_files {src}.")
                return None
            except Exception as e2:
                logging.error(f"Error loading dataset {url_type} with data_files {src}: {e2}")
                return None
        else:
            return None

    return dataset

def drop_columns(dataset, drop_list):
    if not check_parameter(drop_list):
        return dataset
    columns_to_delete = [col for col in dataset.column_names if col in drop_list]
    return dataset.remove_columns(columns_to_delete)

def select_columns(dataset, cols):
    if not check_parameter(cols):
        return dataset
    all_cols = dataset.column_names
    cols_to_drop = [c for c in all_cols if c not in cols]
    return drop_columns(dataset, cols_to_drop)

def select_rows(dataset, start_index, end_index):
    import PIL
    PIL.__spec__ = PIL.Image.__spec__

    from datasets import Dataset

    subset_data = {}
    for column_name in dataset.column_names:
        column_data = dataset[column_name]
        subset_data[column_name] = column_data[start_index:end_index]
    subset = Dataset.from_dict(subset_data)
    return subset

def split_columns(data, labels, is_batch=False):
    if not check_parameter(labels):
        X, y = data
        return X, y

    if is_batch:
        X_batch = []
        y_batch = []

        batch_size = 0
        for value in data.values():
            if isinstance(value, list) or isinstance(value, np.ndarray):
                batch_size = len(value)
                break

        if batch_size == 0:
            return [], []

        for i in range(batch_size):
            X = {}
            y = {}
            for key, value in data.items():
                if key not in labels:
                    X[key] = value[i]
                else:
                    y[key] = value[i]
            X_batch.append(X)
            y_batch.append(y)

        return X_batch, y_batch

    else:
        features = drop_columns(data, labels)
        labels_data = select_columns(data, labels)
        return features, labels_data

def tokenize_and_pad(rows, tokenizer=None):

    if not tokenizer:
        tokenizer = init_tokenizer()

    features_list = []
    for row in rows:
        if isinstance(row, dict):  # Check if it's a dictionary (important!)
            features_strings = []
            for key, value in row.items():
                if isinstance(value, (list, np.ndarray)):
                    features_strings.extend(map(str, value)) #Convert list or numpy array elements to string
                elif value is not None:  # Corrected condition: if value is not None
                    features_strings.append(str(value)) #Convert other values to string
            features_list.append(" ".join(features_strings)) #Join the string values to one string for each row
        elif isinstance(row, str): #Handle if it is a string
            features_list.append(row)
        else:
            return rows

    tokenized_inputs = tokenizer(features_list, padding=True, truncation=True, return_tensors="pt")
    return two_dim_numpy(tokenized_inputs['input_ids'])

def init_tokenizer(mod="google-bert/bert-base-multilingual-cased"):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(mod)

def init_custom_model(model_type, model_path=None):
    import onnx
    import pickle

    try:
        model = None

        if model_type.lower() not in ["onnx", "pkl"]:
            print('Model type must be one of ["onnx", "pkl"]')
            return None

        if model_path and model_type.lower() == "onnx":
            try:
                with open(model_path, 'rb') as f:
                    model = onnx.load(f)
            except Exception as e_onnx_load:
                print(f"Error loading ONNX model: {e_onnx_load}")
                return None
        elif model_path and model_type.lower() == "pkl":
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            except Exception as e_pkl_load:
                print(f"Error loading Pickle model: {e_pkl_load}")
                return None
        else:
            model = None

        return model

    except:
        catch("Error initializing model")
        return None

def files_to_dataset(features_paths: list, labels_paths: list = None):
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    features = []
    labels = []

    try:
        for feature_path in features_paths:
            loaded = load_as_numpy(feature_path,training=True)

            if isinstance(loaded,list):
                for l in loaded:
                    feature = cupy_to_numpy(l)
                    if feature is None:
                        print(f"Error loading feature file: {feature_path}")
                        return None
                    features.append( feature )
            else:
                feature = cupy_to_numpy(loaded)
                if feature is None:
                    print(f"Error loading feature file: {feature_path}")
                    return None
                features.append( feature )

        if labels_paths:
            for label_path in labels_paths:
                loaded = load_as_numpy(label_path,training=True)

                if isinstance(loaded,list):
                    for l in loaded:
                        label = cupy_to_numpy(l)
                        if label is None:
                            print(f"Error loading label file: {label_path}")
                            return None
                        labels.append( label )
                else:
                    label = cupy_to_numpy(loaded)
                    if label is None:
                        print(f"Error loading label file: {label_path}")
                        return None
                    labels.append( label )

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    max_lens = get_max_shapes(*features,*labels)

    try:
        features = convert_tensor_dtype(torch.stack([
            torch.tensor(reshape_numpy(_,lengths=max_lens)) for _ in features
        ]))

        if labels:
            labels = convert_tensor_dtype(torch.stack([
                torch.tensor(reshape_numpy(_,lengths=max_lens)) for _ in labels
            ]))

        if labels is not None and len(labels) > 0:
            dataset = TensorDataset(features, labels)
        elif features is not None and len(features) > 0:
            dataset = TensorDataset(features)
        else:
            print("No features or labels loaded.")
            return None

        return dataset

    except Exception as e_label:
        catch(f"{type(e_label)}")
        catch(e_label)
        return None

def merge_columns(X, y=None):
    from torch.utils.data import TensorDataset, DataLoader
    if y:
        return TensorDataset(X, y)
    return X

def to_loader(dataset, batch_size=1):
    from torch.utils.data import TensorDataset, DataLoader
    return DataLoader(dataset, pin_memory=False, num_workers=0, batch_size=batch_size, shuffle=True, drop_last=False)

def pad_sequences(X):
    import torch
    X = three_dim_numpy(X)
    X = torch.from_numpy(cupy_to_numpy(X))
    return torch.nn.utils.rnn.pad_sequence(X, batch_first=True)

def kmeans_k_suggestions(X, k_range=range(2,20), random_state=None):
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    wcss_values = {}
    silhouette_scores = {}
    davies_bouldin_indices = {}
    calinski_harabasz_indices = {}

    suggested_k_elbow = None
    suggested_k_silhouette = None
    suggested_k_davies_bouldin = None
    suggested_k_calinski_harabasz = None
    final_suggestion_k = None

    kmeans_lib = None
    is_cupy_available = True

    try:
        import cupy
    except ImportError:
        is_cupy_available = False

    from cuml.cluster import KMeans
    kmeans_lib = KMeans

    X_array = np.asarray(X)

    if len(k_range) < 2:
        return {
            'wcss': wcss_values,
            'silhouette_scores': silhouette_scores,
            'davies_bouldin_indices': davies_bouldin_indices,
            'calinski_harabasz_indices': calinski_harabasz_indices,
            'suggested_k_elbow': suggested_k_elbow,
            'suggested_k_silhouette': suggested_k_silhouette,
            'suggested_k_davies_bouldin': suggested_k_davies_bouldin,
            'suggested_k_calinski_harabasz': suggested_k_calinski_harabasz,
            'final_suggestion': final_suggestion_k,
            'notes': "K-range too small to provide meaningful suggestions. Try a range with at least 2 different k values."
        }

    for k in k_range:
        if k <= 1:
            wcss_values[k] = 0
            silhouette_scores[k] = np.nan
            davies_bouldin_indices[k] = np.nan
            calinski_harabasz_indices[k] = np.nan
            continue

        kmeans = kmeans_lib(n_clusters=int(k), random_state=random_state, init='k-means++')
        labels = kmeans.fit_predict(X_array)

        numpy_labels = np.asnumpy(labels) if is_cupy_available else labels
        numpy_X = np.asnumpy(X_array) if is_cupy_available else X_array

        wcss_values[k] = kmeans.inertia_

        silhouette_scores[k] = silhouette_score(numpy_X, numpy_labels)
        davies_bouldin_indices[k] = davies_bouldin_score(numpy_X, numpy_labels)
        calinski_harabasz_indices[k] = calinski_harabasz_score(numpy_X, numpy_labels)

    wcss_ratios = {}
    if len(k_range) > 2:
        for i in range(len(k_range) - 1):
            k1 = k_range[i]
            k2 = k_range[i+1]
            if wcss_values[k1] > 0:
                ratio = wcss_values[k2] / wcss_values[k1]
                wcss_ratios[k2] = ratio

        if wcss_ratios:
            suggested_k_elbow = min(wcss_ratios, key=wcss_ratios.get)


    suggested_k_silhouette = max(silhouette_scores, key=silhouette_scores.get)

    suggested_k_davies_bouldin = min(davies_bouldin_indices, key=davies_bouldin_indices.get)

    suggested_k_calinski_harabasz = max(calinski_harabasz_indices, key=calinski_harabasz_indices.get)

    if suggested_k_elbow is not None:
        final_suggestion_k = suggested_k_elbow
    elif suggested_k_silhouette is not None and silhouette_scores[suggested_k_silhouette] > 0.5:
        final_suggestion_k = suggested_k_silhouette
    elif suggested_k_calinski_harabasz is not None:
        final_suggestion_k = suggested_k_calinski_harabasz
    else:
        final_suggestion_k = None


    return {
        'wcss': wcss_values,
        'silhouette_scores': silhouette_scores,
        'davies_bouldin_indices': davies_bouldin_indices,
        'calinski_harabasz_indices': calinski_harabasz_indices,
        'suggested_k_elbow': suggested_k_elbow,
        'suggested_k_silhouette': suggested_k_silhouette,
        'suggested_k_davies_bouldin': suggested_k_davies_bouldin,
        'suggested_k_calinski_harabasz': suggested_k_calinski_harabasz,
        'final_suggestion': final_suggestion_k,
        'random_state': random_state,
        'notes': "Suggestions are based on heuristics. Visualize metrics and use domain knowledge for final k selection. GPU acceleration is automatically used if available."
    }

def fit(model):

    log("Features",model.X_all)

    if hasattr(model,"y_all"):
        log("Labels",model.y_all)
        max_lens = get_max_shapes(model.X_all,model.y_all)
        try:
            model.X_all = numpy_to_cupy(reshape_numpy(cupy_to_numpy(model.X_all),lengths=max_lens))
            model.y_all = numpy_to_cupy(reshape_numpy(cupy_to_numpy(model.y_all),lengths=max_lens))
            log("Fitting Supervised", model.X_all.shape[0])

            model.fit(model.X_all, model.y_all)

        except Exception as e:
            catch(e)
    else:
        max_lens = get_max_shapes(model.X_all)
        try:
            model.X_all = numpy_to_cupy(reshape_numpy(cupy_to_numpy(model.X_all),lengths=max_lens))
            log("Fitting Unsupervised", model.X_all.shape[0])

            model.fit(model.X_all)

        except Exception as e:
            catch(e)

    return model

def feed(model, X_new, y_new=None, epochs=1):

    if model is None:
        model = HybridModel()

    if y_new is None:

        for epoch in range(epochs):
            log(f"Feeding epoch {epoch+1} X",one_dim_numpy(X_new))
            if not hasattr(model, 'X_all'):
                model.X_all = one_dim_numpy(X_new)
            else:
                model.X_all = one_dim_numpy(np.concatenate((model.X_all, one_dim_numpy(X_new)), axis=0))

    else:

        for epoch in range(epochs):
            log(f"Feeding epoch {epoch+1} X",one_dim_numpy(X_new))
            log(f"Feeding epoch {epoch+1} y",one_dim_numpy(y_new))
            if not hasattr(model, 'X_all'):
                model.X_all = one_dim_numpy(X_new)
                model.y_all = one_dim_numpy(y_new)
            else:
                model.X_all = one_dim_numpy(np.concatenate((model.X_all, one_dim_numpy(X_new)), axis=0))
                model.y_all = one_dim_numpy(np.concatenate((model.y_all, one_dim_numpy(y_new)), axis=0))

    return model

def train(
    model_path=None,
    remote_src=None, revision=None, url_type="parquet",
    features=None, labels=None,
    dataset_label_columns=None, drop_list=None,
    selected_rows=None
):
    import joblib

    tokenizer = init_tokenizer()

    got_inp = check_parameter(features) or check_parameter(remote_src)
    is_supv = check_parameter(dataset_label_columns) or check_parameter(labels)

    model = None

    if check_parameter(model_path):
        model = joblib.load(model_path)
        print(f"cuML model loaded from {model_path}")
        if model is None:
            logging.error(f"Could not load model from {model_path}")
            return None

    model_path = f'model_{random_string()}.joblib'

    if not got_inp:
        return None

    if check_parameter(remote_src):
        dataset = fetch_dataset(remote_src, url_type, revision)
    else:
        dataset = files_to_dataset(features, labels)

    dataset = drop_columns(dataset, drop_list)

    log("Full dataset length",len(dataset))

    loaders = []
    if check_parameter(selected_rows):
        selected_rows = simple_text(selected_rows).split()
        for part in selected_rows:
            if "-" in part:
                start_end = part.split("-")
                loaders.append(to_loader(
                    select_rows( dataset, int(start_end[0])-1, int(start_end[-1]) )
                ))
            else:
                loaders.append(to_loader(
                    select_rows( dataset, int(part)-1, int(part) )
                ))
    else:
        loaders.append( to_loader(dataset) )

    if is_supv:

        for l,loader in enumerate(loaders):
            print(f"Loader {l+1}")
            for i,b in enumerate(loader):
                print(f"Batch {i+1}: {b}")

                X, y = split_columns(b, dataset_label_columns, is_batch=True)

                X = tokenize_and_pad(X, tokenizer)
                y = tokenize_and_pad(y, tokenizer)

                #  X = process_rows(X)

                X = pad_sequences(X)

                X = numpy_to_cupy(X)
                y = numpy_to_cupy(y)

                print("Feeding model")
                model = feed(model, X, y)

    else:

        for l,loader in enumerate(loaders):
            print(f"Loader {l+1}")
            for i,b in enumerate(loader):
                print(f"Batch {i+1}: {b}")

                X = tokenize_and_pad(b, tokenizer)

                #  X = process_rows(X)

                X = pad_sequences(X)

                X = numpy_to_cupy(X)

                print("Feeding model")
                model = feed(model, X)

    print("Fitting model")
    fit(model)

    try:
        joblib.dump(model, model_path)
        log("Trained model path",model_path,status=True)
        return model_path
    except Exception as e:
        print(f"Error saving cuML model: {e}")
        return None


def create_vectorizer(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)
    return vectorizer

def vectorize(vectorizer, texts):

    if vectorizer is None or texts is None:
        return None

    X_tfidf = vectorizer.transform(texts)
    return np.array(X_tfidf.toarray())

def unvectorize(vectorizer, vectorized_data):
    if vectorizer is None or vectorized_data is None:
        return None

    vocabulary = vectorizer.vocabulary_
    index_to_word = {v: k for k, v in vocabulary.items()}  # Reverse mapping

    unvectorized_texts = []
    for row in vectorized_data:
        words = []
        for i, value in enumerate(row):
            if value > 0:  # Consider non-zero TF-IDF values
                if i in index_to_word:
                    words.append(index_to_word[i])
        unvectorized_texts.append(" ".join(words))  # Reconstruct text

    return unvectorized_texts


def extract_video_features(video_path, frame_interval=10):
    import cv2
    import skimage.feature as skf

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error opening video file.")

        frame_count = 0
        all_frame_features = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Color Histograms
                hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256]).flatten()
                hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256]).flatten()
                hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256]).flatten()
                color_hist = _np.concatenate((hist_b, hist_g, hist_r)).astype(_np.float32)

                radius = 1
                n_points = 8 * radius
                lbp = skf.local_binary_pattern(frame_gray, n_points, radius, method='uniform').flatten().astype(_np.float32)

                edges = cv2.Canny(frame_gray, 100, 200).flatten().astype(_np.float32)

                frame_features = _np.concatenate((color_hist, lbp, edges))
                all_frame_features.append(frame_features)

            frame_count += 1

        cap.release()

        if not all_frame_features:
            return None

        return np.array(all_frame_features)

    except Exception as e:
        catch(e)
        return None

def extract_text_features(text, vectorizer=None):
    from sklearn.feature_extraction.text import TfidfVectorizer

    try:
        vectorizer = vectorizer or TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text])  # Input must be a list of strings
        features = tfidf_matrix.toarray().flatten().astype(_np.float32) #convert to float 32 for cuml
        return features

    except Exception as e:
        print(f"Error extracting text features: {e}")
        return None

def extract_image_features(image_path):
    import cv2
    import skimage.feature as skf

    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image could not be read.")

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Color Histograms (3 channels: B, G, R)
        hist_b = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        hist_g = cv2.calcHist([img], [1], None, [256], [0, 256]).flatten()
        hist_r = cv2.calcHist([img], [2], None, [256], [0, 256]).flatten()
        color_hist = _np.concatenate((hist_b, hist_g, hist_r)).astype(_np.float32)

        # Local Binary Patterns (LBP)
        radius = 1
        n_points = 8 * radius
        lbp = skf.local_binary_pattern(img_gray, n_points, radius, method='uniform').flatten().astype(_np.float32)

        # Canny Edge Detection
        edges = cv2.Canny(img_gray, 100, 200).flatten().astype(_np.float32)

        # Combine all features
        all_features = _np.concatenate((color_hist, lbp, edges))

        return all_features

    except Exception as e:
        print(f"Error extracting image features: {e}")
        return None

def extract_audio_features(file_path, n_mfcc=20):
    import librosa

    try:
        y, sr = librosa.load(file_path, sr=None)  # Load with original sample rate
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None

    try:
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, n_mels=80).flatten()

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).flatten()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).flatten()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).flatten()

        spectral_features = _np.concatenate((spectral_centroid, spectral_bandwidth, spectral_rolloff))

        # Zero-crossing rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y).flatten()

        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr).flatten()

        # Combine all features
        all_features = _np.concatenate((mfccs, spectral_features, zero_crossing_rate, chroma)).astype(_np.float32) #convert to float 32 for cuml
        return all_features

    except Exception as e:
        catch(e)
        return None

def load_as_numpy(path, training=False):

    import imageio as iio
    from scipy.io import wavfile
    import sox
    import pandas

    try:
        parts = path.split(".")
        if len(parts) >= 2:
            last = parts[-1].strip().lower()
            if last in ["wav","mp3"]:
                try:
                    tfm = sox.Transformer()
                    tfm.rate(32000)

                    if training:

                        temp_name = tmp("wav")
                        tfm.build_file(path, temp_name)

                        temp_2 = tmp("mp3")
                        remove_silence(temp_name,temp_2)
                        dir, num = split_mp3( temp_2, 5 )
                        files = read(dir)
                        x = []
                        for _f in files:
                            _x = numpy_to_cupy(extract_audio_features(f'{dir}/{_f}'))
                            x.append(_x)
                        delete(temp_name)
                        delete(temp_2)

                    else:
                        temp_name = tmp("mp3")
                        tfm.build_file(path, temp_name)

                        x = numpy_to_cupy(extract_audio_features(temp_name))
                        delete(temp_name)

                    return x

                except Exception as e:
                    catch(e)
                    return None
            elif last in ["csv", "xlsx", "json"]:  # Return NumPy arrays for consistency
                try:
                    if last == "csv":
                        df = pandas.read_csv(path)
                    elif last == "xlsx":
                        df = pandas.read_excel(path)
                    elif last == "json":
                        df = pandas.read_json(path)
                    return df.values  # Convert DataFrame to NumPy array
                except Exception as e_data:
                    catch(e_data)
                    return None
            elif last == "txt":
                try:
                    txt = read(path)
                    return numpy_to_cupy(extract_text_features(txt))
                except Exception as e_txt:
                    catch(e_txt)
                    return None
            elif last in iio_formats:
                try:
                    image_data = iio.imread(path)
                    data = resize_image(image_data, 1024, 1024)
                    path_resized = save_image(data)
                    return numpy_to_cupy(extract_image_features(path_resized))
                except Exception as e_image:
                    catch(e_image)
                    return None
            else:
                try:
                    resized_video_file = resize_video(path, 1024, 1024)
                    new_fps_video_file = convert_video_fps(resized_video_file, 24)
                    return numpy_to_cupy(extract_video_features(new_fps_video_file))
                except Exception as e_video:
                    catch(e_video)
                    return None
        else:
            print(f"Invalid path format: {path}")
            return None
    except Exception as e_overall:
        catch(e_overall)
        return None

def read_as_numpy(path:str):
    return load_as_numpy(path)

def get_prediction_file_extension(pred_type):
    """Returns the correct file extension for the prediction type."""
    if pred_type == "video":
        return "mp4"
    elif pred_type == "image":
        return "png"
    elif pred_type == "audio":
        return "wav"
    elif pred_type == "text":
        return "txt"
    else:
        return "data"


def process_rows(batch):
    try:
        from cuml.preprocessing import StandardScaler, Normalizer, SimpleImputer
    except Exception as e:
        catch(e)
        print("Falling back to sklearn (CPU)")
        from sklearn.preprocessing import StandardScaler, Normalizer
        from sklearn.impute import SimpleImputer

    lst = []
    for i, row in enumerate(batch):
        r = two_dim_numpy(row)
        log(f"Scaling {i+1}",r)
        scaler = StandardScaler()
        r = scaler.fit_transform(r)
        log(f"Normalizing {i+1}",r)
        normalizer = Normalizer()
        r = normalizer.fit_transform(r)
        log(f"Imputing {i+1}",r)
        imputer = SimpleImputer()
        r = imputer.fit_transform(r)
        log(f"Reshaping {i+1}",r)
        lst.append(reshape_numpy(r))

    return two_dim_numpy(lst)

def predict_linear_regression(X_new, model_path):
    import torch
    try:
        input_dim = X_new.shape[1]
        model_torch = LinearRegressionTorch(input_dim)
        model_torch.load_state_dict(torch.load(model_path))

        model_torch.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_torch.to(device)

        X_new_torch = torch.tensor(X_new, dtype=torch.float32, device=device)

        with torch.no_grad():
            predictions_torch = model_torch(X_new_torch).squeeze()

        predictions_numpy = predictions_torch.cpu().numpy()

        return predictions_numpy

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def features_to_audio(
    predicted_features,
    sr=None,
    n_mfcc=None,
    n_mels=None,
    n_fft=None,
    hop_length=512,
):
    import librosa

    sr = sr if sr is not None else 32000
    n_mfcc = n_mfcc if n_mfcc is not None else 20
    n_mels = n_mels if n_mels is not None else 80
    n_fft = n_fft if n_fft is not None else 2048

    if sr is None or n_mfcc is None or n_mels is None or n_fft is None:
         print("Error: Audio parameters (sr, n_mfcc, n_mels, n_fft) are not provided and global defaults are not set.")
         return None

    expected_freq_bins = n_fft // 2 + 1

    try:
        predicted_features = _np.asarray(predicted_features)
        remainder = predicted_features.size % n_mfcc
        if remainder != 0:
            padding_needed = n_mfcc - remainder
            print(f"Padding with {padding_needed} zeros to make the predicted features ({predicted_features.size}) a multiple of n_mfcc ({n_mfcc}).")
            predicted_features = _np.pad(predicted_features, (0, padding_needed), mode='constant', constant_values=0)

        mfccs = predicted_features.reshape((n_mfcc, -1))

        if mfccs.shape[1] == 0:
            print("Error: Reshaped MFCCs have zero frames. Cannot proceed with audio reconstruction.")
            return None

        mel_spectrogram_db = librosa.feature.inverse.mfcc_to_mel(mfccs, n_mels=n_mels)

        mel_spectrogram = librosa.db_to_amplitude(mel_spectrogram_db)
        mel_spectrogram = _np.nan_to_num(mel_spectrogram, nan=0.0, posinf=_np.finfo(_np.float16).max, neginf=_np.finfo(_np.float16).min)
        mel_spectrogram = _np.maximum(0, mel_spectrogram)

        magnitude_spectrogram = librosa.feature.inverse.mel_to_stft(M = mel_spectrogram, sr = sr, n_fft = n_fft)
        magnitude_spectrogram = _np.nan_to_num(magnitude_spectrogram, nan=0.0, posinf=_np.finfo(_np.float16).max, neginf=_np.finfo(_np.float16).min)
        magnitude_spectrogram = _np.maximum(0, magnitude_spectrogram)
        magnitude_spectrogram = _np.nan_to_num(magnitude_spectrogram, nan=0.0, posinf=_np.finfo(_np.float16).max, neginf=_np.finfo(_np.float16).min)

        if magnitude_spectrogram.shape[0] != expected_freq_bins:
            print(f"Error: Magnitude spectrogram has incorrect frequency bin count ({magnitude_spectrogram.shape[0]}) for n_fft ({n_fft}).\nExpected {expected_freq_bins}.\nCannot perform Griffin-Lim.")
            return None

        if magnitude_spectrogram.shape[1] == 0:
            print("Error: Magnitude spectrogram has zero frames. Skipping Griffin-Lim.")
            return None

        griffin_lim_iterations = [12, 32]

        for n_iter in griffin_lim_iterations:
            try:
                audio_waveform = librosa.griffinlim(magnitude_spectrogram, n_fft=n_fft, hop_length=hop_length, n_iter=n_iter)

                if audio_waveform.size > 0:
                    print(f"Griffin-Lim finished {n_iter} iterations")

                    audio_waveform = _np.nan_to_num( audio_waveform, nan=0.0, posinf=_np.finfo(_np.float16).max, neginf=_np.finfo(_np.float16).min )                        
                    audio_waveform = _np.clip(audio_waveform, -1.0, 1.0)

                    if not _np.all(_np.isfinite(audio_waveform)):
                         print("Warning: Audio waveform contains non-finite values after clipping.\nThis is unexpected.\nReturning None.")
                         return None

                    return audio_waveform

                else:
                    print(f"Griffin-Lim with n_iter={n_iter} produced an empty output.")

            except Exception as e:
                print(f"Griffin-Lim with n_iter={n_iter} failed!")
                catch(e)
                if n_iter == griffin_lim_iterations[-1]:
                    print("Griffin-Lim failed. Returning None.")
                    return None
                else:
                    print("Trying again with more iterations...")

        return None

    except Exception as e:
        catch(e)
        return None

def predict_audio(model, audio_file):
    import librosa
    import soundfile as sf

    try:
        audio_data, sr = librosa.load(audio_file, sr=32000, mono=True)

        timeline = get_active_audio_timeline(audio_file)

        log("Audio shape", audio_data.shape)
        log("Active audio timeline", timeline)

        predicted_audio = _np.zeros_like(audio_data) # Use _np for standard numpy operation

        if not timeline:
             log("Silent timeline", "No active audio segments found.")


        for i, (start_time, end_time) in enumerate(timeline):
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)

            active_audio_part_np = audio_data[start_sample:end_sample]

            if active_audio_part_np.size == 0:
                 log("Segment skipped", f"Skipping empty audio segment from {start_time:.2f}s to {end_time:.2f}s")
                 continue

            active_audio_part_model_input = numpy_to_cupy(active_audio_part_np)

            log("Predicting segment", f"Predicting audio segment {i+1}/{len(timeline)} with shape {active_audio_part_model_input.shape}")
            prediction = model.predict(active_audio_part_model_input)

            if is_clusters_model(model):
                 log("Getting prediction cluster content", f"Predicted cluster for segment {i+1}: {int(prediction[0])}")
                 part_feat = cupy_to_numpy(get_cluster_content(model, int(prediction[0])))
            else:
                 part_feat = cupy_to_numpy(prediction)

            log("Prediction shape", f"Predicted features shape for segment {i+1}: {part_feat.shape}")

            part_aud = features_to_audio(part_feat)

            if part_aud is None:
                 log("Segment failure", f"Failed to convert features to audio for segment {i+1}. Skipping this segment.")
                 continue

            part_length = end_sample - start_sample

            min_len = min(part_aud.shape[0], part_length)
            predicted_audio[start_sample : start_sample + min_len] = part_aud[:min_len]


        output_file = tmp("wav")
        sf.write(output_file, predicted_audio, sr)

        log("Audio output", f"Predicted audio saved to: {output_file}")
        return output_file

    except Exception as e:
        catch(e)
        return None

def features_to_image(predicted_features):
    import cv2

    image_shape = (1024, 1024, 3)

    try:
        height, width, channels = image_shape
        hist_size = 256 * 3  # Color histograms (B, G, R)
        lbp_size = height * width  # LBP features
        edge_size = height * width # Canny edge features

        # Split the predicted features
        color_hist = predicted_features[:hist_size].reshape(3, 256)
        lbp_features = predicted_features[hist_size:hist_size + lbp_size].reshape(height, width)
        edge_features = predicted_features[hist_size + lbp_size:].reshape(height, width)

        # Create a blank image
        reconstructed_image = np.zeros(image_shape, dtype=np.uint8)

        # Reconstruct color channels (simplified)
        for c in range(channels):
          for i in range(256):
            if c == 0:
              reconstructed_image[:,:,0] += np.uint8(color_hist[0][i]/np.max(color_hist[0]) * 255)
            elif c == 1:
              reconstructed_image[:,:,1] += np.uint8(color_hist[1][i]/np.max(color_hist[1]) * 255)
            else:
              reconstructed_image[:,:,2] += np.uint8(color_hist[2][i]/np.max(color_hist[2]) * 255)

        # Reconstruct LBP and Edge (simplified)
        lbp_scaled = cv2.normalize(lbp_features, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        edge_scaled = cv2.normalize(edge_features, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        reconstructed_image_gray = cv2.addWeighted(lbp_scaled, 0.5, edge_scaled, 0.5, 0)
        reconstructed_image = cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2GRAY)
        reconstructed_image = cv2.addWeighted(reconstructed_image, 0.5, reconstructed_image_gray, 0.5, 0)
        reconstructed_image = cv2.cvtColor(reconstructed_image, cv2.COLOR_GRAY2BGR)

        return reconstructed_image

    except Exception as e:
        print(f"Error generating image from features: {e}")
        return None

def features_to_video(predicted_features, frame_interval=10, fps=24):
    """
    Generates a video from predicted video features, assuming the features
    were extracted using the provided 'extract_video_features' function.

    Args:
        predicted_features (numpy.ndarray): 2D NumPy array, where each row represents features from a frame.
        frame_interval (int): Frame interval used during feature extraction.
        fps (int): Frames per second of the output video.

    Returns:
        bool: True if video generation was successful, False otherwise.
    """

    import cv2

    output_path = tmp("mp4")

    video_shape = (1024, 1024, 3)

    try:
        height, width, channels = video_shape
        hist_size = 256 * 3
        lbp_size = height * width
        edge_size = height * width

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or another suitable codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame_features in predicted_features:
            color_hist = frame_features[:hist_size].reshape(3, 256)
            lbp_features = frame_features[hist_size:hist_size + lbp_size].reshape(height, width)
            edge_features = frame_features[hist_size + lbp_size:].reshape(height, width)

            reconstructed_frame = np.zeros(video_shape, dtype=np.uint8)

            # Reconstruct color channels (simplified)
            for c in range(channels):
                for i in range(256):
                    if c == 0:
                        reconstructed_frame[:, :, 0] += np.uint8(color_hist[0][i] / np.max(color_hist[0]) * 255)
                    elif c == 1:
                        reconstructed_frame[:, :, 1] += np.uint8(color_hist[1][i] / np.max(color_hist[1]) * 255)
                    else:
                        reconstructed_frame[:, :, 2] += np.uint8(color_hist[2][i] / np.max(color_hist[2]) * 255)

            # Reconstruct LBP and Edge (simplified)
            lbp_scaled = cv2.normalize(lbp_features, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            edge_scaled = cv2.normalize(edge_features, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            reconstructed_frame_gray = cv2.addWeighted(lbp_scaled, 0.5, edge_scaled, 0.5, 0)
            reconstructed_frame = cv2.cvtColor(reconstructed_frame, cv2.COLOR_BGR2GRAY)
            reconstructed_frame = cv2.addWeighted(reconstructed_frame, 0.5, reconstructed_frame_gray, 0.5, 0)
            reconstructed_frame = cv2.cvtColor(reconstructed_frame, cv2.COLOR_GRAY2BGR)

            out.write(reconstructed_frame)

        out.release()
        return output_path

    except Exception as e:
        print(f"Error generating video from features: {e}")
        return False

def features_to_text(predicted_features, vectorizer=None, vocabulary=None):
    """
    Generates text from predicted TF-IDF features, assuming the features
    were extracted using the provided 'extract_text_features' function.

    Args:
        predicted_features (numpy.ndarray): 1D NumPy array of predicted TF-IDF features.
        vectorizer (TfidfVectorizer, optional): The trained TfidfVectorizer used to extract features.
                                                  If None, a vocabulary must be provided.
        vocabulary (list, optional): The vocabulary (list of words) used to create the features.
                                      Required if vectorizer is None.

    Returns:
        str: Reconstructed text string, or None if an error occurs.
    """

    from sklearn.feature_extraction.text import TfidfVectorizer

    try:
        if vectorizer is None and vocabulary is None:
            raise ValueError("Either a vectorizer or a vocabulary must be provided.")

        if vectorizer is None:
            vectorizer = TfidfVectorizer(vocabulary=vocabulary)
            vectorizer.fit(vocabulary)  # Need to fit with vocabulary to get correct mapping

        # Reconstruct the TF-IDF matrix (1 sample)
        tfidf_matrix = predicted_features.reshape(1, -1)

        # Inverse transform to get word indices
        word_indices = tfidf_matrix.nonzero()[1]

        # Get feature names (words)
        feature_names = vectorizer.get_feature_names_out()

        # Reconstruct the text
        reconstructed_words = [feature_names[i] for i in word_indices]
        reconstructed_text = " ".join(reconstructed_words)

        return reconstructed_text

    except Exception as e:
        print(f"Error generating text from features: {e}")
        return None

def predict(prediction_file: str, model_path: str):

    import imageio as iio
    from scipy.io import wavfile
    import joblib

    vec = None
    input_data = None

    mod = joblib.load(model_path)
    print(f"cuML model loaded from {model_path}")
    if mod == None:
        logging.error(f"Could not load model from {model_path}")
        return None

    if prediction_file.split(".")[-1].strip().lower() == "txt":
        txt = read(prediction_file)
        if isinstance(txt, tuple) or isinstance(txt, list):
            txt = "".join(txt)
        vec = create_vectorizer([txt])
        input_data = numpy_to_cupy(extract_text_features(txt,vec))
    elif prediction_file.split(".")[-1].strip().lower() in common_audio_formats:
        out = predict_audio(mod,prediction_file)
        print(f"Prediction saved to {out}")
        return out
    else:
        input_data = numpy_to_cupy(load_as_numpy(prediction_file))

    if input_data == None:
        log("Could not load input data",prediction_file,status=False)
        return None

    input_data = one_dim_numpy(input_data)
    pred = mod.predict(input_data)

    if pred == None:
        logging.error("Model prediction failed.")
        return None

    if is_clusters_model(mod):
        pred = one_dim_numpy(get_cluster_content( mod, int(pred[0]) ))

    pred_type = guess_numpy_type(pred)
    output_filename = f"{random_string()}.{get_prediction_file_extension(pred_type)}"

    if vec != None:
        pred = features_to_text(cupy_to_numpy(pred))
    elif pred_type == "text":
        vec = create_vectorizer([""])
        pred = features_to_text(cupy_to_numpy(pred))
    elif pred_type == "audio":
        pred = features_to_audio(cupy_to_numpy(pred))
    elif pred_type == "image":
        pred = features_to_image(cupy_to_numpy(pred))
    elif pred_type == "video":
        pred = features_to_video(cupy_to_numpy(pred))

    handlers = {
        "video": lambda: write_video(pred, 24),
        "image": lambda: iio.imwrite(output_filename, (cupy_to_numpy(pred) * 255).astype(np.uint8)),
        "audio": lambda: wavfile.write(output_filename, 32000, cupy_to_numpy(pred)),
        "text": lambda: open(output_filename, "w").write(pred),
    }

    if pred_type in handlers:
        try:
            handlers[pred_type]()
        except Exception as e:
            catch(e)
            return None
    else:
        logging.error(f"Unsupported prediction type: {pred_type}")
        return None

    print(f"Prediction saved to {output_filename}")
    return output_filename

def lang_code_to_name(code):
    return language_codes[code]

def write_on_image(image_path,top_title=None,middle_title=None,bottom_title=None):
    from PIL import Image, ImageDraw, ImageFont

    if "Alef-Bold.ttf" not in read("."):
        google_drive_download("1C48KkYWQDYu7ypbNtSXAUJ6kuzoZ42sI","./Alef-Bold.ttf")

    img = Image.open(image_path)

    w, h = img.size
    
    draw = ImageDraw.Draw(img)

    labels_distance = 1/3

    if top_title:
        rows = len(top_title.split("\n"))
        textheight=min(math.ceil( w / 10 ), math.ceil( h / 5 ))
        font = ImageFont.truetype("Alef-Bold.ttf", textheight)
        textwidth = draw.textlength(top_title,font)
        x = math.ceil((w - textwidth) / 2)
        y = h - (textheight * rows / 2) - (h / 2)
        y = math.ceil(y - (h / 2 * labels_distance))
        draw.text((x, y), top_title, (255,255,255), font=font, spacing=2, stroke_width=math.ceil(textheight/20), stroke_fill=(0,0,0))

    if middle_title:
        rows = len(middle_title.split("\n"))
        textheight=min(math.ceil( w / 12 ), math.ceil( h / 6 ))
        font = ImageFont.truetype("Alef-Bold.ttf", textheight)
        textwidth = draw.textlength(middle_title,font)
        x = math.ceil((w - textwidth) / 2)
        y = h - (textheight * rows / 2) - (h / 2)
        draw.text((x, y), middle_title, (255,255,255), font=font, spacing=4, stroke_width=math.ceil(textheight/40), stroke_fill=(64,64,64))

    if bottom_title:
        rows = len(bottom_title.split("\n"))
        textheight=min(math.ceil( w / 10 ), math.ceil( h / 5 ))
        font = ImageFont.truetype("Alef-Bold.ttf", textheight)
        textwidth = draw.textlength(bottom_title,font)
        x = math.ceil((w - textwidth) / 2)
        y = h - (textheight * rows / 2) - (h / 2)
        y = math.ceil(y + (h / 2 * labels_distance))
        draw.text((x, y), bottom_title, (0,0,0), font=font, spacing=2, stroke_width=math.ceil(textheight/20), stroke_fill=(255,255,255))

    return save_image(img)

def init_upscale():
    import torch
    import numpy as np
    from torch import nn
    import pillow_heif
    from PIL import Image
    from refiners.foundationals.latent_diffusion.stable_diffusion_1.multi_upscaler import (
        MultiUpscaler,
        UpscalerCheckpoints,
    )
    try:
        import cupy.typing as npt
    except Exception as e:
        import numpy.typing as npt

    Tile = tuple[int, int, Image.Image]
    Tiles = list[tuple[int, int, list[Tile]]]
    
    def conv_block(in_nc: int, out_nc: int) -> nn.Sequential:
        
        return nn.Sequential(
            nn.Conv2d(in_nc, out_nc, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
    
    class ResidualDenseBlock_5C(nn.Module):

        def __init__(self, nf: int = 64, gc: int = 32) -> None:
            
            super().__init__()
    
            self.conv1 = conv_block(nf, gc)
            self.conv2 = conv_block(nf + gc, gc)
            self.conv3 = conv_block(nf + 2 * gc, gc)
            self.conv4 = conv_block(nf + 3 * gc, gc)
            self.conv5 = nn.Sequential(nn.Conv2d(nf + 4 * gc, nf, kernel_size=3, padding=1))
    
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x1 = self.conv1(x)
            x2 = self.conv2(torch.cat((x, x1), 1))
            x3 = self.conv3(torch.cat((x, x1, x2), 1))
            x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):

        def __init__(self, nf: int) -> None:
            super().__init__()
            self.RDB1 = ResidualDenseBlock_5C(nf)
            self.RDB2 = ResidualDenseBlock_5C(nf)
            self.RDB3 = ResidualDenseBlock_5C(nf)
    
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.RDB1(x)
            out = self.RDB2(out)
            out = self.RDB3(out)
            return out * 0.2 + x
    
    
    class Upsample2x(nn.Module):
        """Upsample 2x."""
    
        def __init__(self) -> None:
            super().__init__()  # type: ignore[reportUnknownMemberType]
    
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return nn.functional.interpolate(x, scale_factor=2.0)  # type: ignore
    
    
    class ShortcutBlock(nn.Module):
        """Elementwise sum the output of a submodule to its input"""
    
        def __init__(self, submodule: nn.Module) -> None:
            super().__init__()  # type: ignore[reportUnknownMemberType]
            self.sub = submodule
    
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.sub(x)
    
    
    class RRDBNet(nn.Module):
        def __init__(self, in_nc: int, out_nc: int, nf: int, nb: int) -> None:
            super().__init__()  # type: ignore[reportUnknownMemberType]
            assert in_nc % 4 != 0  # in_nc is 3
    
            self.model = nn.Sequential(
                nn.Conv2d(in_nc, nf, kernel_size=3, padding=1),
                ShortcutBlock(
                    nn.Sequential(
                        *(RRDB(nf) for _ in range(nb)),
                        nn.Conv2d(nf, nf, kernel_size=3, padding=1),
                    )
                ),
                Upsample2x(),
                nn.Conv2d(nf, nf, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                Upsample2x(),
                nn.Conv2d(nf, nf, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(nf, nf, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(nf, out_nc, kernel_size=3, padding=1),
            )
    
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)
    
    
    def infer_params(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, int, int, int]:
        # this code is adapted from https://github.com/victorca25/iNNfer
        scale2x = 0
        scalemin = 6
        n_uplayer = 0
        out_nc = 0
        nb = 0
    
        for block in list(state_dict):
            parts = block.split(".")
            n_parts = len(parts)
            if n_parts == 5 and parts[2] == "sub":
                nb = int(parts[3])
            elif n_parts == 3:
                part_num = int(parts[1])
                if part_num > scalemin and parts[0] == "model" and parts[2] == "weight":
                    scale2x += 1
                if part_num > n_uplayer:
                    n_uplayer = part_num
                    out_nc = state_dict[block].shape[0]
            assert "conv1x1" not in block  # no ESRGANPlus
    
        nf = state_dict["model.0.weight"].shape[0]
        in_nc = state_dict["model.0.weight"].shape[1]
        scale = 2**scale2x
    
        assert out_nc > 0
        assert nb > 0
    
        return in_nc, out_nc, nf, nb, scale  # 3, 3, 64, 23, 4
    
    # https://github.com/philz1337x/clarity-upscaler/blob/e0cd797198d1e0e745400c04d8d1b98ae508c73b/modules/images.py#L64
    Grid = namedtuple("Grid", ["tiles", "tile_w", "tile_h", "image_w", "image_h", "overlap"])

    # adapted from https://github.com/philz1337x/clarity-upscaler/blob/e0cd797198d1e0e745400c04d8d1b98ae508c73b/modules/images.py#L67
    def split_grid(image: Image.Image, tile_w: int = 512, tile_h: int = 512, overlap: int = 64) -> Grid:
        w = image.width
        h = image.height
    
        non_overlap_width = tile_w - overlap
        non_overlap_height = tile_h - overlap
    
        cols = max(1, math.ceil((w - overlap) / non_overlap_width))
        rows = max(1, math.ceil((h - overlap) / non_overlap_height))
    
        dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
        dy = (h - tile_h) / (rows - 1) if rows > 1 else 0
    
        grid = Grid([], tile_w, tile_h, w, h, overlap)
        for row in range(rows):
            row_images: list[Tile] = []
            y1 = max(min(int(row * dy), h - tile_h), 0)
            y2 = min(y1 + tile_h, h)
            for col in range(cols):
                x1 = max(min(int(col * dx), w - tile_w), 0)
                x2 = min(x1 + tile_w, w)
                tile = image.crop((x1, y1, x2, y2))
                row_images.append((x1, tile_w, tile))
            grid.tiles.append((y1, tile_h, row_images))
    
        return grid

    # https://github.com/philz1337x/clarity-upscaler/blob/e0cd797198d1e0e745400c04d8d1b98ae508c73b/modules/images.py#L104
    def combine_grid(grid: Grid):
        def make_mask_image(r: npt.NDArray[np.float32]) -> Image.Image:
            r = r * 255 / grid.overlap
            return Image.fromarray(r.astype(np.uint8), "L")
    
        mask_w = make_mask_image(
            np.arange(grid.overlap, dtype=np.float32).reshape((1, grid.overlap)).repeat(grid.tile_h, axis=0)
        )
        mask_h = make_mask_image(
            np.arange(grid.overlap, dtype=np.float32).reshape((grid.overlap, 1)).repeat(grid.image_w, axis=1)
        )
    
        combined_image = Image.new("RGB", (grid.image_w, grid.image_h))
        for y, h, row in grid.tiles:
            combined_row = Image.new("RGB", (grid.image_w, h))
            for x, w, tile in row:
                if x == 0:
                    combined_row.paste(tile, (0, 0))
                    continue
    
                combined_row.paste(tile.crop((0, 0, grid.overlap, h)), (x, 0), mask=mask_w)
                combined_row.paste(tile.crop((grid.overlap, 0, w, h)), (x + grid.overlap, 0))
    
            if y == 0:
                combined_image.paste(combined_row, (0, 0))
                continue
    
            combined_image.paste(
                combined_row.crop((0, 0, combined_row.width, grid.overlap)),
                (0, y),
                mask=mask_h,
            )
            combined_image.paste(
                combined_row.crop((0, grid.overlap, combined_row.width, h)),
                (0, y + grid.overlap),
            )
    
        return combined_image
    
    
    class UpscalerESRGAN:
        def __init__(self, model_path: Path, device: torch.device, dtype: torch.dtype):
            self.model_path = model_path
            self.device = device
            self.model = self.load_model(model_path)
            self.to(device, dtype)
    
        def __call__(self, img: Image.Image) -> Image.Image:
            return self.upscale_without_tiling(img)
    
        def to(self, device: torch.device, dtype: torch.dtype):
            self.device = device
            self.dtype = dtype
            self.model.to(device=device, dtype=dtype)
    
        def load_model(self, path: Path) -> RRDBNet:
            filename = path
            state_dict: dict[str, torch.Tensor] = torch.load(filename, weights_only=True, map_location=self.device)  # type: ignore
            in_nc, out_nc, nf, nb, upscale = infer_params(state_dict)
            assert upscale == 4, "Only 4x upscaling is supported"
            model = RRDBNet(in_nc=in_nc, out_nc=out_nc, nf=nf, nb=nb)
            model.load_state_dict(state_dict)
            model.eval()
    
            return model
    
        def upscale_without_tiling(self, img: Image.Image) -> Image.Image:
            img_np = np.array(img)
            img_np = img_np[:, :, ::-1]
            img_np = np.ascontiguousarray(np.transpose(img_np, (2, 0, 1))) / 255
            img_t = torch.from_numpy(img_np).float()  # type: ignore
            img_t = img_t.unsqueeze(0).to(device=self.device, dtype=self.dtype)
            with torch.no_grad():
                output = self.model(img_t)
            output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = 255.0 * np.moveaxis(output, 0, 2)
            output = output.astype(np.uint8)
            output = output[:, :, ::-1]
            return Image.fromarray(output, "RGB")
    
        # https://github.com/philz1337x/clarity-upscaler/blob/e0cd797198d1e0e745400c04d8d1b98ae508c73b/modules/esrgan_model.py#L208
        def upscale_with_tiling(self, img: Image.Image) -> Image.Image:
            img = img.convert("RGB")
            grid = split_grid(img)
            newtiles: Tiles = []
            scale_factor: int = 1
    
            for y, h, row in grid.tiles:
                newrow: list[Tile] = []
                for tiledata in row:
                    x, w, tile = tiledata
                    output = self.upscale_without_tiling(tile)
                    scale_factor = output.width // tile.width
                    newrow.append((x * scale_factor, w * scale_factor, output))
                newtiles.append((y * scale_factor, h * scale_factor, newrow))
    
            newgrid = Grid(
                newtiles,
                grid.tile_w * scale_factor,
                grid.tile_h * scale_factor,
                grid.image_w * scale_factor,
                grid.image_h * scale_factor,
                grid.overlap * scale_factor,
            )
            output = combine_grid(newgrid)
            return output

    @dataclass(kw_only=True)
    class ESRGANUpscalerCheckpoints(UpscalerCheckpoints):
        esrgan: Path

    class ESRGANUpscaler(MultiUpscaler):
        def __init__(
            self,
            checkpoints: ESRGANUpscalerCheckpoints,
            device: torch.device,
            dtype: torch.dtype,
        ) -> None:
            super().__init__(checkpoints=checkpoints, device=device, dtype=dtype)
            self.esrgan = UpscalerESRGAN(checkpoints.esrgan, device=self.device, dtype=self.dtype)
    
        def to(self, device: torch.device, dtype: torch.dtype):
            self.esrgan.to(device=device, dtype=dtype)
            self.sd = self.sd.to(device=device, dtype=dtype)
            self.device = device
            self.dtype = dtype
    
        def pre_upscale(self, image: Image.Image, upscale_factor: float, **_: Any) -> Image.Image:
            image = self.esrgan.upscale_with_tiling(image)
            return super().pre_upscale(image=image, upscale_factor=upscale_factor / 4)

    pillow_heif.register_heif_opener()

    def _rescale_checkpoints():
    
        from huggingface_hub import hf_hub_download
    
        CHECKPOINTS = ESRGANUpscalerCheckpoints(
            unet=Path(
                hf_hub_download(
                    repo_id="refiners/juggernaut.reborn.sd1_5.unet",
                    filename="model.safetensors",
                    revision="347d14c3c782c4959cc4d1bb1e336d19f7dda4d2",
                )
            ),
            clip_text_encoder=Path(
                hf_hub_download(
                    repo_id="refiners/juggernaut.reborn.sd1_5.text_encoder",
                    filename="model.safetensors",
                    revision="744ad6a5c0437ec02ad826df9f6ede102bb27481",
                )
            ),
            lda=Path(
                hf_hub_download(
                    repo_id="refiners/juggernaut.reborn.sd1_5.autoencoder",
                    filename="model.safetensors",
                    revision="3c1aae3fc3e03e4a2b7e0fa42b62ebb64f1a4c19",
                )
            ),
            controlnet_tile=Path(
                hf_hub_download(
                    repo_id="refiners/controlnet.sd1_5.tile",
                    filename="model.safetensors",
                    revision="48ced6ff8bfa873a8976fa467c3629a240643387",
                )
            ),
            esrgan=Path(
                hf_hub_download(
                    repo_id="philz1337x/upscaler",
                    filename="4x-UltraSharp.pth",
                    revision="011deacac8270114eb7d2eeff4fe6fa9a837be70",
                )
            ),
            negative_embedding=Path(
                hf_hub_download(
                    repo_id="philz1337x/embeddings",
                    filename="JuggernautNegative-neg.pt",
                    revision="203caa7e9cc2bc225031a4021f6ab1ded283454a",
                )
            ),
            negative_embedding_key="string_to_param.*",
            loras={
                "more_details": Path(
                    hf_hub_download(
                        repo_id="philz1337x/loras",
                        filename="more_details.safetensors",
                        revision="a3802c0280c0d00c2ab18d37454a8744c44e474e",
                    )
                ),
                "sdxl_render": Path(
                    hf_hub_download(
                        repo_id="philz1337x/loras",
                        filename="SDXLrender_v2.0.safetensors",
                        revision="a3802c0280c0d00c2ab18d37454a8744c44e474e",
                    )
                )
            }
        )
    
        return CHECKPOINTS

    upscaler = ESRGANUpscaler(checkpoints=_rescale_checkpoints(), device=device(), dtype=dtype())
    upscaler.to(device=device(), dtype=dtype())

    MODELS["upscale"] = upscaler

def upscale(
    path,
    upscale_factor: int = 2,
    prompt: str = "Reasonable, Accurate, Natural, Real, Convincing.",
    negative_prompt: str = "Fake, Polished, Shiny, Blurry, Painted, Anime.",
    seed: int = None,
    controlnet_scale: float = 0.5,
    controlnet_decay: float = 0.8,
    condition_scale: float = 8.0,
    tile_width: int = 1024,
    tile_height: int = 1024,
    denoise_strength: float = 0.2,
    num_inference_steps: int = 25,
    solver: str = "DDIM",
):
    from PIL import Image
    from refiners.fluxion.utils import manual_seed
    from refiners.foundationals.latent_diffusion import Solver, solvers

    if upscale_factor < 2 or upscale_factor > 4:
        return

    if not seed:
        seed = random.randint(0, big_number())

    manual_seed(seed)

    solver_type: type[Solver] = getattr(solvers, solver)

    input_image = Image.open(path)

    upscaled_image = MODELS["upscale"].upscale(
        image=input_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        upscale_factor=upscale_factor,
        controlnet_scale=controlnet_scale,
        controlnet_scale_decay=controlnet_decay,
        condition_scale=condition_scale,
        tile_size=(tile_height, tile_width),
        denoise_strength=denoise_strength,
        num_inference_steps=num_inference_steps,
        loras_scale={"more_details": 0.3, "sdxl_render": 1.0},
        solver_type=solver_type,
    )

    return save_image( upscaled_image )

def find_latest_rvc_checkpoint(folder_path: str, model_name: str) -> str | None:
    logger.info(f"Searching for latest checkpoint in '{folder_path}' with model name '{model_name}'")
    if not os.path.isdir(folder_path):
        logger.error(f"Error: Folder not found at {folder_path}")
        return None

    pattern = re.compile(rf"^{re.escape(model_name)}_e(\d+)_s(\d+)\.pth$")

    latest_checkpoint = None
    latest_epoch = -1
    latest_global_step = -1

    try:
        for filename in os.listdir(folder_path):
            match = pattern.match(filename)
            if match:
                epoch = int(match.group(1))
                global_step = int(match.group(2))

                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_global_step = global_step
                    latest_checkpoint = filename
                elif epoch == latest_epoch and global_step > latest_global_step:
                    latest_global_step = global_step
                    latest_checkpoint = filename

    except Exception as e:
        logger.error(f"An error occurred while scanning the folder for checkpoints: {e}")
        return None

    if latest_checkpoint:
        logger.info(f"Latest checkpoint found: {latest_checkpoint}")
    else:
        logger.warning(f"No checkpoint found matching the pattern in '{folder_path}'")

    return latest_checkpoint

def get_max_resolution(width, height, mega_pixels = 0.25, factor = 16):
    max_pixels = mega_pixels * 1000 * 1000
    ratio = width / height
    new_height = (max_pixels / ratio) ** 0.5
    new_width = ratio * new_height
    new_height = int( int(new_height) - (int(new_height) % factor) )
    new_width = int( int(new_width) - (int(new_width) % factor) )
    return new_width, new_height

def master(source_path, strength, format_choice):
    import matchering as mg
    import pydub

    output_stem = Path(source_path).with_name(f"{Path(source_path).stem}_mastered")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "reference.wav"
            google_drive_download("1UF_FIuq4vbCdDfCVLHvD_9fXzJDoredh", str(reference_path))
            def _master(current_source_path):
                result_wav_path = tmp("wav", keep=False)
                mg.process(target=str(current_source_path), reference=str(reference_path), results=[mg.pcm24(str(result_wav_path))], config=mg.Config(max_length=15*60, threshold=0.99/strength, internal_sample_rate=44100))
                return result_wav_path
            processed_path = source_path
            for _ in range(math.floor(strength)): processed_path = _master(processed_path)
            final_sound = pydub.AudioSegment.from_file(processed_path) + (strength - 1.0) * 6
            output_path = export_audio(final_sound, output_stem, format_choice)
            delete(processed_path)
            return output_path
    except Exception as e:
        catch(e)
        return None

def get_cluster_content(model, cluster_index):

    if not hasattr(model, 'labels_'):
        raise ValueError("Model must be a trained KMeans model.")

    cluster_labels = model.labels_

    cluster_contents = {}
    for i, label in enumerate(cluster_labels):
        if label not in cluster_contents:
            cluster_contents[label] = []
        cluster_contents[label].append(model.x_all[i])

    if cluster_index in cluster_contents:
        return cluster_contents[cluster_index]
    return None

def is_clusters_model(model):
    return hasattr(model,"cluster_centers_")

def install_faiss():
    if importable("faiss"):
        return False
    faiss_repo_url = "https://github.com/facebookresearch/faiss.git"
    faiss_dir = "_faiss_"
    build_dir = os.path.join(faiss_dir, "build")
    python_dir = os.path.join(build_dir, "faiss", "python")
    try:
        subprocess.run(["git", "clone", faiss_repo_url, faiss_dir], check=True)
        with cwd(faiss_dir):
            cmake_command = [
                "cmake", "-B", build_dir, "-DBUILD_TESTING=OFF", "-DCMAKE_BUILD_TYPE=Release",
                "-DFAISS_ENABLE_C_API=ON", "-DFAISS_ENABLE_GPU=ON", "-DFAISS_ENABLE_PYTHON=ON",
                f"-DPython_EXECUTABLE={sys.executable}",
                f"-DPython_INCLUDE_DIR={sys.prefix}/include/python{sys.version_info.major}.{sys.version_info.minor}",
                f"-DPython_LIBRARY={sys.prefix}/lib/libpython{sys.version_info.major}.{sys.version_info.minor}.so",
                f"-DPython_NumPy_INCLUDE_DIRS={sys.prefix}/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/numpy/core/include",
                "."
            ]
            subprocess.run(cmake_command, check=True)
            subprocess.run(["make", "-C", build_dir, "-j16", "faiss"], check=True)
            subprocess.run(["make", "-C", build_dir, "-j16", "swigfaiss"], check=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "."], cwd=python_dir, check=True)
        print("Faiss installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error during installation: {e}")
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def simple_text(prompt):
    prompt = re.sub("[\t]", ' ', prompt)
    prompt = re.sub("( ){2,}", ' ', prompt)
    prompt = re.sub("(\n){2,}", "\n", prompt)
    prompt = re.sub("(-){2,}", "-", prompt)
    prompt = re.sub(punc, '', prompt)
    prompt = prompt.lower().strip()
    prompt = prompt.replace(" -", "-").replace("- ", "-")
    return prompt

def exist(path):
    path = os.path.abspath(os.path.expanduser(path.strip()))
    ret = run(f'ls -1 "{ path }"', silent=True)
    if not ret:
        return False
    return True

def add_path(path):

    if path not in sys.path:
        permit(path)
        sys.path.append(path)
        site.addsitedir(path)

def paths(*patterns):

    patterns = [os.path.abspath(os.path.expanduser(p)) for p in patterns]

    path_list = []
    for p in patterns:
        try:
            lst = list(glob(p,recursive=True))
            path_list = [*path_list,*lst]
        except Exception as e:
            pass

    return list(set(path_list))

def copy(src,dst):
    if os.path.isdir(src) or Path(src).is_symlink() and os.path.isdir( str(Path(src).resolve()) ):
        shutil.copytree(
            src, dst,
            symlinks=False,
            ignore_dangling_symlinks=True
        )
    else:
        shutil.copy(src, dst)

def big_number(zeros=10):
    return int("1" + ("0" * zeros))

def find_package_paths(package_name):
    package_paths_found = []
    package_dir_name = package_name.replace('-', '_')

    site_packages_dirs = site.getsitepackages()
    for site_packages_dir in site_packages_dirs:
        package_path = os.path.join(site_packages_dir, package_dir_name)
        if os.path.exists(package_path) and os.path.isdir(package_path):
            package_paths_found.append(package_path)

    for path in sys.path:
        if path:
            potential_package_path = os.path.join(path, package_dir_name)
            if os.path.exists(potential_package_path) and os.path.isdir(potential_package_path):
                package_paths_found.append(potential_package_path)

    for site_packages_dir in site_packages_dirs:
        dist_packages_dir = site_packages_dir.replace('site-packages', 'dist-packages')
        if dist_packages_dir != site_packages_dir:
            package_path = os.path.join(dist_packages_dir, package_dir_name)
            if os.path.exists(package_path) and os.path.isdir(package_path):
                package_paths_found.append(package_path)

    unique_paths = list(set(package_paths_found))
    return unique_paths

def tmp(suffix:str=".data", keep:bool=True, dir=False):
    if dir:
        with tempfile.TemporaryDirectory() as temp:
            if not keep:
                delete(temp)
            return temp
    if not suffix.startswith("."):
        if len(suffix.split(".")) > 1:
            suffix = suffix.split(".")
            suffix = suffix[len(suffix)-1]
            if len(suffix) < 1:
                suffix = "tmp"
        suffix = "." + suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        if not keep:
            delete(temp.name)
        return temp.name

def get_process_pid(process_name):
    try:
        pid = int(subprocess.check_output(["pidof", process_name]).strip())
        return pid
    except subprocess.CalledProcessError:
        return None
    except ValueError:
        return None

def send_signal_to_process(pid, signal_number):
    try:
        os.kill(pid, signal_number)
        return True
    except OSError as e:
        print(f"Error sending signal: {e}")
        return False

@contextmanager
def cwd(dir=None):
    if not dir:
        dir = os.path.dirname(__file__)
    owd = os.getcwd()
    try:
        os.chdir(dir)
        yield dir
    finally:
        os.chdir(owd)

def log(subject,data,status=None):

    if status is True:
            print(f"\n >>> { datetime.now().time() } <<< \nOK OK OK OK OK OK OK\n{ str(data) }\nOK OK OK OK OK OK OK\n >>> {subject} <<< \n")
    elif status is False:
            print(f"\n >>> { datetime.now().time() } <<< \nx ERR x ERR x ERR x\n{ str(data) }\nx ERR x ERR x ERR x\n >>> {subject} <<< \n")
    elif status is None:
            print(f"\n >>> { datetime.now().time() } <<< \n===================\n{ str(data) }\n===================\n >>> {subject} <<< \n")
    elif isinstance(status,str) and status.strip() != "":
            print(f"\n >>> { datetime.now().time() } <<< \n{status}\n{ str(data) }\n{status}\n >>> {subject} <<< \n")
    else:
            print(f"\n{ datetime.now().time() }\n{ str(data) }\n{subject}\n")

def catch(e):
    logger.exception(e)

def directory(dir):
    dir = os.path.realpath( str(dir) )
    os.makedirs(dir, exist_ok=True)

def move(src,dest):
    if os.path.isdir(src) or Path(src).is_symlink() and os.path.isdir( str(Path(src).resolve()) ):
        shutil.copytree(
            src, dest,
            symlinks=False,
            ignore_dangling_symlinks=True,
            copy_function=shutil.move
        )
        shutil.rmtree(src)
    else:
        shutil.move(src, dest)

def delete(path):
    obj = Path(path)
    if not exist(path):
        return
    if os.path.isdir(path) and not obj.is_symlink():
        shutil.rmtree(path)
    else:
        obj.unlink(missing_ok=True)

def remove(path):
    delete(path)

def load(path):
        path = os.path.realpath( str(path) )
        permit(path)
        if not os.path.exists(path):
                return None
        if os.path.isdir(path):
                return os.listdir(path)
        else:
                try:
                        with open(path, encoding="utf8") as file:
                                return file.read()
                except:
                        with open(path, "rb") as file:
                                return file.read()

def read(path):
    return load(path)

def write(path,txt=""):
    return save(path,txt)

def save(path, text=""):
    path = os.path.realpath( str(path) )
    os.makedirs( str(Path(path).parent), exist_ok=True)
    with open(path, "w+", encoding="utf8") as file:
        file.write( str(text) )

def save_temp_text(text_content):
    if text_content is None:
        return None
    temp_path = tmp()
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(text_content)
    return temp_path

def run_linux(command, silent=False, env={}):
    import pty

    original_env = os.environ.copy()
    modified_env = {**original_env, **env}

    if isinstance(command, list):
        command = "\n".join(command)

    in_lines = command.strip().splitlines()
    cmds = [i.strip() for i in in_lines if i.strip() != ""]

    if len(cmds) > 0:

        script = "\n".join(cmds)

        name = tmp(".sh")
        try:
            write(name,"#!/bin/bash --login\n"+script)
            permit(name)
            master, slave = pty.openpty()
            pid = os.fork()
            if pid == 0:
                os.setsid()
                try:
                    with open(os.devnull, "r") as stdin:
                        os.dup2(stdin.fileno(), 0)
                    os.dup2(slave, 1)  # Redirect stdout to the slave pty
                    os.dup2(slave, 2)  # Redirect stderr to the slave pty
                    os.close(master)  # Close the master in the child
                    os.close(slave)  # Close the slave in the child (important!)
                    os.environ.update(modified_env)
                    os.execl("/bin/bash", "/bin/bash", "--login", "-c", name, "&")
                except Exception as e:
                    print(f"Execution Error: {e}")
                finally:
                    delete(name)
                    os.environ.update(original_env)
                    os._exit(0)  # Child MUST exit

            else:  # Parent process

                os.close(slave)
                output_bytes = b""
                output=""
                while True:
                    rlist, _, _ = select.select([master], [], [])  # Wait for output
                    if master in rlist:
                        try:
                            chunk = os.read(master, 1024)
                            if not chunk:  # Process finished
                                break
                            output_bytes += chunk
                            try:
                                chunk_utf = chunk.decode('utf-8', errors='replace')
                                if not silent:
                                    print(chunk_utf, end="", flush=True)
                                output+=chunk_utf
                            except UnicodeDecodeError:
                                continue
                        except OSError: # Handle pty closing
                            break
                os.close(master)
                returncode = os.waitpid(pid, 0)[1] >> 8  # Get the return code
                if returncode != 0:
                    if not silent:
                        log(f'Script failed [{returncode}]',script)
                    return False
                if not silent:
                    log('Script completed',script)
                out_lines = output.strip().splitlines()
                ret_lines = [o.strip() for o in out_lines if o.strip() != ""]
                return ret_lines

        except OSError as e:
            catch(e)
            return False

def run_windows(
    command, 
    silent = False, 
    env = {}
):
    try:
        if isinstance(command, list):
            cmds = command
        else:
            cmds = command.strip().splitlines()
            if len(cmds) > 1:
                 command_to_run = " && ".join([c.strip() for c in cmds if c.strip()])
            else:
                 command_to_run = command

        modified_env = {**os.environ.copy(), **env}

        process = subprocess.Popen(
            command_to_run,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=modified_env,
            universal_newlines=True
        )

        stdout, stderr = process.communicate()

        returncode = process.returncode

        if not silent:
            if stdout:
                print(stdout, end="", flush=True)
            if stderr:
                print(stderr, end="", flush=True)

        if returncode != 0:
            if not silent:
                log(f'Script failed [{returncode}]', command_to_run)
                log(f'Stderr: {stderr.strip()}', "")
            return False
        else:
            if not silent:
                log('Script completed', command_to_run)
            
            out_lines = stdout.strip().splitlines()
            ret_lines = [o.strip() for o in out_lines if o.strip()]
            return ret_lines

    except Exception as e:
        catch(e)
        return False

def run(command, silent=False, env={}):
    if sys.platform.startswith('win'):
        return run_windows(command, silent, env)
    else:
        return run_linux(command, silent, env)

def thread(func, *args, **kwargs):
    try:
        t = threading.Thread(target=func, args=args, kwargs=kwargs)
        t.start()
        return t
    except Exception as e:
        catch(e)

def wait(*threads):
    for t in threads:
        t.join()

def permit(path):
    try:
        subprocess.run(["chmod", "-R", "a+xrw", path], check=True)
        return True
    except Exception as e:
        return False

def check_version_wildcard(version_spec, version_actual):
    version_spec = version_spec.replace(".","\\.").replace("*", ".*")
    pattern = re.compile(f"^{version_spec}$")
    return bool(pattern.match(version_actual))

def installed(pack, version=None):

    pack_lower = pack.lower().strip()

    version_lower = None
    if version:
        version_lower = version.lower().strip()

    system = get_os_name()

    if system == "windows":
        cmd = 'powershell.exe -Command "Get-ItemProperty HKLM:\\Software\\Wow6432Node\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*, HKLM:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*, HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\* | Select-Object DisplayName, DisplayVersion | Format-Table -HideTableHeaders'
        try:
            lines = run(cmd, silent=True)
            for line in lines:
                parts = re.split(r'\s{2,}', line.strip())
                if not parts or not parts[0]:
                    continue

                name = parts[0].lower().strip()
                ver = parts[1].strip() if len(parts) > 1 else ""

                if pack_lower in name:
                    if version_lower is None or (ver and ver.startswith(version_lower)) or (ver and "*" in version_lower and check_version_wildcard(version_lower, ver)):
                        return True
        except Exception:
            return False

    elif system == "linux":
        if not shutil.which(pack):
            return False

        if version_lower is None:
            return True

        try:
            lines = run(f'{pack} --version', silent=True)
            if not lines:
                 lines = run(f'{pack} -v', silent=True)
            
            if lines:
                match = re.search(r'(\d+\.\d+(\.\d+)*)', lines[0])
                if match:
                    actual_version = match.group(0)
                    if actual_version.startswith(version_lower) or ("*" in version_lower and check_version_wildcard(version_lower, actual_version)):
                        return True
        except Exception:
            return False 

    try:
        lines = run(f'pip list', silent=True)
        if lines:
            for line in lines:
                parts = re.sub( r"( ){2,}", ";", line).split(";")
                if len(parts) == 2:
                    n = parts[0].lower().strip()
                    v = parts[1].lower().strip()
                    if n == pack_lower and (
                        version_lower == None or v.startswith(version_lower) or (
                            "*" in version_lower and check_version_wildcard(version_lower, v)
                        )
                    ):
                        return True
                else:
                    continue

        return False

    except subprocess.CalledProcessError as e:
        catch(e)
        return False

    except FileNotFoundError:
        return False

def importable(name):
    res = run(f'python -c "import {name}"', silent=True)
    if res == False:
        return False
    return True

def runnable(cmd):
    if get_os_name() == "windows" and run(f"powershell.exe -Command {repr(cmd)} -WhatIf", silent=True):
        return True
    if get_os_name() == "linux" and run(f"which {repr(cmd.split()[0])}", silent=True):
        return True
    return False

def is_package_path(package_path,package_name=None):
    if exist(package_path) and os.path.isdir(package_path) and (
        os.path.exists(os.path.join(package_path, "__init__.py")) or (
            os.path.exists(os.path.join(package_path, os.path.basename(package_path)))
        ) or (
            os.path.exists(os.path.join(package_path, "src"))
        )
    ) and (
        package_name is None or package_name == os.path.basename(package_path)
    ):
        return True
    return False

def cuda_toolkit():

    if get_os_name() != "linux":
        return None

    directory("/usr/share/keyrings/")
    directory("/etc/modprobe.d/")
    permit("/tmp")
    permit("/usr/bin")
    permit("/usr/lib")
    permit("/usr/local")

    run("apt-get update")
    run(f"apt-get install -y curl")

    run("""
        export PATH=/sbin:$PATH
        apt-get update
        apt-get purge nvidia-*
        echo "blacklist nouveau" > /etc/modprobe.d/blacklist-nouveau.conf
        echo "options nouveau modeset=0" >> /etc/modprobe.d/blacklist-nouveau.conf
        apt-get install -y --reinstall dkms
        apt-get install -f
        curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb > /usr/share/keyrings/cuda.deb
        cd /usr/share/keyrings/
        ar vx cuda.deb
        tar xvf data.tar.xz
        mv /usr/share/keyrings/usr/share/keyrings/cuda-archive-keyring.gpg /usr/share/keyrings/cuda-archive-keyring.gpg
        rm -r /usr/share/keyrings/usr/
        rm -r /usr/share/keyrings/etc/
        echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /" > /etc/apt/sources.list.d/CUDA.list
    """)

    permit("/usr/share/keyrings/cuda-archive-keyring.gpg")
    permit("/etc/apt/sources.list.d/CUDA.list")

    run(f"""
        apt-get update
        apt-get install -y cuda-toolkit
    """)

def cuda_version():
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, check=True)
        output = result.stdout
        match = re.search(r"Build cuda_([\d\.]+)", output)
        if match:
            cuda_version = match.group(1)
            return cuda_version
        else:
            return False

    except Exception as e:
        return False

def set_cuda_env():

    if get_os_name() != "linux":
        return None

    cu_path = paths(
        "/opt/cuda*/",
        "/usr/local/cuda*/",
    )
    ld_path = paths(
        "/opt/cuda*/lib",
        "/usr/local/cuda*/lib",
        "/opt/cuda*/lib64",
        "/usr/local/cuda*/lib64",
    )
    if len(cu_path) > 0 and len(ld_path) > 0:
        cu = cu_path[0]
        ld = ld_path[0]
        log("CUDA_PATH",cu,status=True)
        log("LD_LIBRARY_PATH",ld,status=True)
        os.environ["CUDA_PATH"] = cu
        os.environ["LD_LIBRARY_PATH"] = ld
        return

    log("Cuda not found", "Failed setting CUDA environment",status=False)
    return

def free():
    import torch
    try:
        torch.cuda.empty_cache()
    except Exception as e:
        catch(e)
    run("rm -rf ~/.cache/huggingface/*", silent=True)
    run("rm -rf /data-nvme/zerogpu-offload/*", silent=True)
    run("rm -rf /opt/ml/checkpoints/*", silent=True)
    run(f'pip cache purge', silent=True)

    mamba_path = os.path.expanduser("~/miniconda3/bin/mamba")
    if os.path.exists(mamba_path):
        run(f'{mamba_path} clean --all', silent=True)

def post_install():

    free()

    import torch
    from torch.fx.experimental import proxy_tensor
    def get_proxy_mode(): # -> Optional[ProxyTorchDispatchMode]
        pre_dispatch_mode = torch._ops._get_dispatch_mode_pre_dispatch(
            torch._C._TorchDispatchModeKey.PROXY
        )
        mode = torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY)
        assert (
            pre_dispatch_mode is None or mode is None
        ), f"pre_dispatch_mode={pre_dispatch_mode}, mode={mode}"
        return pre_dispatch_mode or mode
    proxy_tensor.get_proxy_mode = getattr(proxy_tensor,"get_proxy_mode",get_proxy_mode)

    import numpy as np
    def dummy_npwarn_decorator_factory():
        def npwarn_decorator(x):
            return x
        return npwarn_decorator
    np._no_nep50_warning = getattr(np, '_no_nep50_warning', dummy_npwarn_decorator_factory)

def pre_install():

    os.environ['TRANSFORMERS_CACHE'] = '/opt/ml/checkpoints/'
    os.environ['HF_DATASETS_CACHE'] = '/opt/ml/checkpoints/'
    os.environ["GRADIO_ALLOW_FLAGGING"] = "never"
    os.environ["OMP_NUM_THREADS"] = "4"
    if sys.platform == "darwin":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["DISPLAY"] = ":0.0"
    os.environ["NUMBA_CACHE_DIR"] = f'{os.environ["HOME"]}/.tmp'
    os.environ["DISABLE_FLASH_ATTENTION"] = "True"

def apt_install():

    basic_apt="build-essential gcc cmake swig gdebi git git-lfs wget curl libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev initramfs-tools libgirepository1.0-dev libdbus-1-dev libdbus-glib-1-dev libsecret-1-0 libmanette-0.2-0 libharfbuzz0b libharfbuzz-icu0 libenchant-2-2 libhyphen0 libwoff1 libgraphene-1.0-0 libxml2-dev libxmlsec1-dev"
    audio_apt="libportaudio2 libasound2-dev sox libsox-fmt-all praat ffmpeg libavcodec-extra libavif-dev"
    visual_apt="libopenblas-dev libgflags-dev libgles2 libgtk-3-0 libgtk-4-1 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxcomposite1 libxdamage1 libatspi2.0-0 libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-gl"

    pre_install()

    run("apt-get update")
    run(f"apt-get install -y { basic_apt } { audio_apt } { visual_apt }")

    post_install()

def device():
    from accelerate import Accelerator
    acc = Accelerator()
    return str(acc.device)

def get_python_version():
    try:
        version_info = sys.version_info
        version_str = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
        return version_str
    except Exception as e:
        print(f"Error getting Python version: {e}")
        return None

def get_linux_distribution():
    try:
        try:
            subprocess.run(['apt-get', 'update'], check=True)
            subprocess.run(['apt-get', 'install', '-y', 'lsb_release'], check=True)
            result = subprocess.run(['lsb_release', '-a'], capture_output=True, text=True, check=True)
            output = result.stdout

            distro_match = re.search(r"Distributor ID:\s*([^\n]+)", output)
            release_match = re.search(r"Release:\s*([^\n]+)", output)

            if distro_match and release_match:
                distro = distro_match.group(1).strip().lower().split(" ")[0]
                release = release_match.group(1).strip()
                return distro, release
            else:
                return None, None

        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        try:
            with open("/etc/os-release", "r") as f:
                os_release_content = f.read()

            name_match = re.search(r'NAME="([^"]+)"', os_release_content)
            version_match = re.search(r'VERSION_ID="([^"]+)"', os_release_content)

            if name_match and version_match:
              distro = name_match.group(1).strip()
              release = version_match.group(1).strip()
              return distro, release
            else:
              return None, None

        except FileNotFoundError:
            return None, None

        return None, None

    except Exception as e:
        print(f"Error getting distribution info: {e}")
        return None, None

def split_mp3( path:str, chunk_seconds:float ):

    from pydub import AudioSegment

    sound = AudioSegment.from_mp3(path)

    chunk_ms = chunk_seconds * 1000
    chunks = [sound[(chunk_ms * i):(chunk_ms * (i+1))] for i in range(math.ceil( len(sound)/(chunk_seconds*1000) ))]

    export_path = f'{os.getcwd()}/mp3_segments_{str(random.random()).split(".")[1]}'

    Path(export_path).mkdir(parents=True, exist_ok=True)

    i = 0
    for chunk_idx in range(len(chunks)):

        chunk = chunks[chunk_idx]
        chunk.export(export_path+f'/{str(chunk_idx)}.mp3', format="mp3")
        i = chunk_idx

    i = i + 1
    return export_path, i

def remove_silence( input_file:str, output_file:str ):
    try:
        subprocess.run(['ffmpeg', '-y', '-i', input_file, '-ac', '2', '-af', 'silenceremove=stop_duration=0.1:stop_threshold=-32dB', output_file], check=True)
        return output_file
        
    except subprocess.CalledProcessError as e:
        catch(e)

def compact_audio( input_file:str, output_file:str ):
    try:
        subprocess.run(['ffmpeg', '-y', '-i', input_file, '-ar', '16000', '-ab', "320k", '-ac', '1', output_file], check=True)
        return output_file
        
    except subprocess.CalledProcessError as e:
        catch(e)

def google_drive_download(id,dest):
    from googledrivedownloader import download_file_from_google_drive
    download_file_from_google_drive( file_id=id, dest_path=dest, unzip=True, showsize=False )

def save_image(img,path="."):
    name = os.path.join( path, "img_"+random_string()+".png" )
    img.save(name)
    return name

def tensor_length(tensor):
    from torch import tensor

    nums = list(tensor.size())

    ret = 1
    for num in nums:
        ret = ret * num
    return ret

def dtype():
    import torch
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def language(text):
    from langdetect import detect
    return detect(text)

def linked_url(url):

    host = url.split("?")[0]
    if "?" in url:
        param = "?"+url.split("?")[1]
    else:
        param = ""

    html_string = f'''
         <!DOCTYPE html>
        <html>
            <head>
                <meta charset="UTF-8">
                <base href="{host}" target="_top">
                <a href="{param}"></a>
            </head>
            <body onload='document.querySelector("a").click()'></body>
        </html>
    '''

    html_bytes = html_string.encode('utf-8')

    base64_encoded_html = base64.b64encode(html_bytes).decode('utf-8')

    data_url = f"data:text/html;charset=utf-8;base64,{base64_encoded_html}"

    return data_url

def geo_new_york():
    return {
        "latitude": random.uniform(40.5,40.9),
        "longitude": random.uniform(-74.2,-73.7)
    }

def extract_text(url,selector):

    from lxml.html import fromstring
    from playwright.sync_api import sync_playwright, expect
    from lxml.cssselect import CSSSelector

    xpath = CSSSelector(selector).path

    log("URL", url)

    html_string = None

    with sync_playwright() as playwright:
        browser = playwright.firefox.launch(headless=True).new_context( locale="en-US", timezone_id="America/New_York", user_agent=random.choice(user_agents["firefox"]), color_scheme="dark")
        page = browser.new_page()
        page.goto( url, referer="https://duckduckgo.com/", timeout=18*1000 )
        expect(page.locator(selector)).not_to_be_empty()
        page.wait_for_timeout(2000)
        html_string = page.content()
        browser.close()

    if html_string == None:
        return None

    html = fromstring(html_string)
    elems = html.xpath(xpath)
    elems = [el.text_content().strip() for el in elems if el.text_content().strip()]
    if len(elems)==0:
        return ""
    return elems[0]

def ai_translate(text, lang="en"):

    if text == None or lang == None:
        return ""

    if text.strip() == "":
        return ""

    lang = simple_text(lang)

    to_lang = language_codes[lang]
    to_lang = to_lang[0].upper() + to_lang[1:]

    from_lang = language_codes[language(text)]
    from_lang = from_lang[0].upper() + from_lang[1:]

    if from_lang == to_lang:
        return simple_text(text)

    text = f"translate {from_lang} to {to_lang}: {simple_text(text)}" 

    log("Exec T5 translation",text,status="")

    TOKENIZERS["summary"].src_lang = from_lang

    encoded = TOKENIZERS["summary"](text, return_tensors="pt")
    encoded = {key: tensor.to(device()) for key, tensor in encoded.items()}

    generated_tokens = MODELS["summary"].generate(**encoded)

    translated_text = TOKENIZERS["summary"].batch_decode(generated_tokens, skip_special_tokens=True)[0]

    log("T5 translated text",translated_text,status="")

    return simple_text(translated_text)

def google_translate(text,lang="en"):

    import requests

    if text == None or lang == None:
        return ""

    if text.strip() == "":
        return ""

    lang = simple_text(lang)
    text = simple_text(text)

    url = f'https://translate.googleapis.com/translate_a/single?client=gtx&dt=t&q={text}&sl={language(text)}&tl={lang}'
    r = requests.get(url)

    ret = r.text.split('"')[1]
    ret = simple_text(ret)
    print(ret)
    return ret

def duck_translate(text,lang="en"):

    if text == None or lang == None:
        return ""

    if text.strip() == "":
        return ""

    lang = simple_text(lang)
    lang = language_codes[lang]

    text = simple_text(text)

    url = f'https://duckduckgo.com/?q={lang} translate: {text}&ia=web'
    scraped = extract_text(url,f".module--translations-translatedtext.js-module--translations-translatedtext")
    if scraped is None or scraped == "":
        print(f'Translation Warning: Failed To Translate!')
    else:
        text = scraped
    text = simple_text(text)
    print(text)
    return text

def css():
    return """

    * {
        scrollbar-width: none;
    }

    input, textarea, input::placeholder, textarea::placeholder {
        text-align: center !important;
    }

        *, *::placeholder {
            font-family: Suez One !important;
        }

        h1,h2,h3,h4,h5,h6 {
            width: 100% !important;
            text-align: center;
        }

        footer {
            display: none !important;
        }

        .dropdown-arrow {
            display: none !important;
        }

        div:not(.hide):has(>button) {
            display: flex !important;
            justify-content: space-evenly !important;
            align-items: center !important;
        }

    button {
        margin: 10px 0 !important; /* Add some vertical margin to buttons */
        border-radius: 2mm !important; /* Rounded corners for buttons */
        border: none !important;
        cursor: pointer !important;
        transition: background-color 0.3s ease !important; /* Smooth hover effect */
    }

    * > img {
        max-width: 100% !important; /* Make sure image scales within the container */
        height: auto !important;    /* Maintain aspect ratio */
        display: block !important; /* Prevents a small space below the image */
    }

    textarea {
        border: 1px solid #ccc !important;
        border-radius: 5px !important;
        padding: 8px !important;
        height: auto !important;
        margin-bottom: 10px !important;
    }

    textarea:focus{
        border-color: #4CAF50 !important;
        outline: none !important;
        box-shadow: 0 0 5px rgba(76, 175, 80, 0.5) !important;
    }

    h1 {
        color: #333 !important;
    }

    h2 {
        color: #444 !important;
    }

    h3{
        color: #555 !important;
    }

    .block {
        gap: 20px !important;
        padding: 10px !important;
    }

    .column{
        padding: 10px !important;
    }

    .gradio-container {
        padding: 20px !important;
    }

    """

def random_string(min_len=50, max_len=60):
    characters = string.ascii_letters + string.digits + "_"
    length = random.randint(min_len,max_len)
    return ''.join(random.choice(characters) for _ in range(length))

def random_number(size):
	return int.from_bytes(os.urandom(size), sys.byteorder)

def number_to_hex(num):
	return int(num).encode('hex')

def string_to_bytes(str):
	return bytes(f"{str}", encoding="utf-8")

def file_to_sha3_512(path,salt_num=None):
	content = read(path)
	if content != None:
		return string_to_sha3_512(content,salt_num)

def string_to_sha3_512(str,salt_num=None):
	if salt_num == None:
		salt_num = random_number(16)
	salt = number_to_hex(salt_num)

	m = hashlib.sha3_512()
	m.update(bytes( str , encoding="utf-8"))
	m.update(bytes( salt ))
	return [ m.hexdigest(), salt_num ]

class Database:
    def __init__(self, path):
        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def _get_history(self, db, filters={}, days=None):
        db_path = os.path.join(self.path, db)
        if not os.path.exists(db_path):
            return []

        start_timestamp = 0
        if days is not None and isinstance(days, (int, float)):
            start_timestamp = time() - (days * 86400)

        try:
            timestamp_dirs = [d for d in os.listdir(db_path) if int(d) >= start_timestamp]
        except (ValueError, FileNotFoundError):
            return []

        results = []
        for ts_string in timestamp_dirs:
            record_path = os.path.join(db_path, ts_string)
            if not os.path.isdir(record_path):
                continue

            item_data = {}
            for key_file in os.listdir(record_path):
                with open(os.path.join(record_path, key_file), 'r') as f:
                    item_data[key_file] = f.read()
            
            all_filters_match = True
            for key, value in filters.items():
                if item_data.get(key) != str(value):
                    all_filters_match = False
                    break
            
            if all_filters_match:
                ts_int = int(ts_string)
                record = {
                    "timestamp": ts_int,
                    "time": datetime.fromtimestamp(ts_int),
                    "data": item_data
                }
                results.append(record)

        return sorted(results, key=lambda x: x['timestamp'], reverse=True)

    def history(self, db, filters={}, days=None):
        full_history = self._get_history(db, filters, days)
        return [item['data'] for item in full_history]

    def push(self, db, data, timestamp=None):
        if timestamp is None:
            timestamp = int(time())
        elif not isinstance(timestamp, int):
            try:
                timestamp = int(timestamp)
            except (ValueError, TypeError):
                timestamp = int(time())

        record_path = os.path.join(self.path, db, str(timestamp))
        os.makedirs(record_path, exist_ok=True)

        for key, value in data.items():
            file_path = os.path.join(record_path, key)
            with open(file_path, 'w') as f:
                f.write(str(value))

    def latest(self, db="*", filters={}, days=None, identifierKey="id"):
        if db == "*":
            return {db_name: self.latest(db_name, filters, days, identifierKey) for db_name in os.listdir(self.path)}
        if isinstance(db, list):
            return {db_name: self.latest(db_name, filters, days, identifierKey) for db_name in db}

        full_history = self._get_history(db)

        latest_items = {}
        for item in full_history:
            item_id = item['data'].get(identifierKey)
            if item_id is None:
                continue

            if item_id not in latest_items or item['timestamp'] > latest_items[item_id]['timestamp']:
                latest_items[item_id] = item

        filtered_results = list(latest_items.values())

        if days is not None:
            start_timestamp = time() - (days * 86400)
            filtered_results = [item for item in filtered_results if item['timestamp'] >= start_timestamp]

        if filters:
            final_results = []
            for item in filtered_results:
                all_filters_match = True
                for key, value in filters.items():
                    if item['data'].get(key) != str(value):
                        all_filters_match = False
                        break
                if all_filters_match:
                    final_results.append(item)
            filtered_results = final_results

        sorted_results = sorted(filtered_results, key=lambda x: x['timestamp'], reverse=True)
        return [item['data'] for item in sorted_results]

    def clean(self, db="*", identifierKey="id"):
        if db == "*":
            dbs = os.listdir(self.path)
            for db_name in dbs:
                self.clean(db_name, identifierKey)
            return
        if isinstance(db, list):
            for db_name in db:
                self.clean(db_name, identifierKey)
            return

        full_history = self._get_history(db)
        latest_items = {}
        for item in full_history:
            item_id = item['data'].get(identifierKey)
            if item_id is None:
                continue
            if item_id not in latest_items or item['timestamp'] > latest_items[item_id]['timestamp']:
                latest_items[item_id] = item
        
        records_to_keep = list(latest_items.values())

        db_path = os.path.join(self.path, db)
        if os.path.isdir(db_path):
            shutil.rmtree(db_path)

        for item in records_to_keep:
            self.push(db, item['data'], item['timestamp'])

def _summarize(text_to_summarize, is_chunk=False):
    prefix = "summarize: "
    encoded = TOKENIZERS["summary"](prefix + text_to_summarize, return_tensors="pt", truncation=True, max_length=512)
    encoded = {key: tensor.to(device()) for key, tensor in encoded.items()}
    
    gen_kwargs = {
        "max_length": 512,
        "repetition_penalty": 1.2,
        "length_penalty": 2.0 if is_chunk else 1.0,
        "no_repeat_ngram_size": 3,
        "num_beams": 8,
        "early_stopping": True
    }
    
    if is_chunk:
        gen_kwargs["min_length"] = 40

    gen = MODELS["summary"].generate(**encoded, **gen_kwargs)
    return simple_text(TOKENIZERS["summary"].decode(gen[0], skip_special_tokens=True))

def map_reduce_summary(text, max_words=50):
    words = text.split()
    chunk_size = 350 
    overlap = 50

    chunk_summaries = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_text = " ".join(words[i:i + chunk_size])
        chunk_summary = _summarize(chunk_text, is_chunk=True)
        chunk_summaries.append(chunk_summary)

    combined_summary = " ".join(chunk_summaries)
    
    if len(combined_summary.split()) > max_words:
        final_summary = _summarize(combined_summary, is_chunk=False)
    else:
        final_summary = combined_summary
        
    return final_summary

def summary(text, max_words=50):
    word_count = len(text.split())
    
    if word_count > 350:
        return map_reduce_summary(text, max_words)
    else:
        return _summarize(text, is_chunk=False)

def prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    **kwargs,
):
    if past_key_values is not None:
        input_ids = input_ids[:, -1:]

    return {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "attention_mask": attention_mask,
        **kwargs,
    }

def init_pretrained_model(task:str,turbo:bool=False):

    free()

    global MODELS
    global TOKENIZERS

    if task in MODELS and MODELS[task]:
        return

    import torch

    model = None

    if task in ["tts"]:

        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device=device())

    elif task in ["svc"]:

        logger.info("Initializing RVC by downloading necessary files.")
        file_ids = {
            "configs": "1dIWJ9iP-nLOUw8eflcHFH3RwFWKNepVW",
            "assets": "1THxR2rRnTx1qv21TZUuCew0G6XlJZJeY",
            "docs": "1LSq2gJJpLMwjkjtDo0urgrIC2eJ9QVDF",
            "i18n": "1bQ4pIZxxpnDknpG63hRwoY10b2jx_RwY",
            "infer": "1kqMYQskvVKwKglcWQsK2Q5G3yPahnbtH",
            "logs": "1fNMl60ga8OMb4aUvnXzrFIExphWBbHXR",
            "tools": "1neqVUNipdXukEpImwZUU8O3aQdXR1vDg",
        }
        for name, file_id in file_ids.items():
            dest_path = f"./{name}.zip"
            logger.info(f"Downloading {name} ({file_id}) to {dest_path}")
            try:
                google_drive_download(id=file_id, dest=dest_path)

            except Exception as e:
                logger.error(f"Failed to download {name}: {e}")
                catch(e)
        logger.info("RVC initialization complete.")

    elif task in ["speech-recognition"]:

        from transformers import pipeline
        model = pipeline("automatic-speech-recognition", model=tasks["speech-recognition"], device=device())

    elif task in ["audio-classification"]:

        from transformers import pipeline
        model = pipeline("audio-classification", model=tasks["audio-classification"], device=device())

    elif task in ["detect"]:

        from transformers import pipeline, AutoProcessor, GenerationConfig, AutoConfig, AutoModel, TFAutoModel, T5ForConditionalGeneration, T5Tokenizer, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        config = AutoConfig.from_pretrained(tasks[task])
        try:
            model = AutoModel.from_pretrained(tasks[task], config=config, trust_remote_code=True, torch_dtype=dtype()).to(device())
        except:
            model = TFAutoModel.from_pretrained(tasks[task], config=config, trust_remote_code=True, torch_dtype=dtype()).to(device())

    elif task in ["music"]:

        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        PROCESSORS[task] = AutoProcessor.from_pretrained("facebook/musicgen-small")
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(device())

    elif task in ["answer"]:

        import torch
        from transformers import TRANSFORMERS_CACHE, AutoConfig, AutoProcessor, AutoModelForCausalLM, AutoTokenizer

        config = AutoConfig.from_pretrained(tasks[task], trust_remote_code=True)
        module_name, class_name = config.auto_map["AutoModelForCausalLM"].rsplit(".", 1)
        model_cache_path = Path(TRANSFORMERS_CACHE) / f"models--{module_name.replace('/', '--')}"
        snapshot_dir = next(model_cache_path.glob("snapshots/*"))
        sys.path.append(str(snapshot_dir))
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        if not hasattr(cls, "prepare_inputs_for_generation"):
            cls.prepare_inputs_for_generation = prepare_inputs_for_generation
            print(f"✅ Successfully patched '{class_name}' with 'prepare_inputs_for_generation'.")
        else:
            print(f"✅ Method 'prepare_inputs_for_generation' already exists on '{class_name}'. No patch needed.")

        tok = AutoTokenizer.from_pretrained(tasks[task])
        prc = AutoProcessor.from_pretrained(tasks[task], trust_remote_code=True)
        mod = AutoModelForCausalLM.from_pretrained(
            tasks[task],
            torch_dtype=dtype(),
            trust_remote_code=True,
            _attn_implementation="eager",
        ).to(device())

        model = BeamSearch(mod, tok, prc, device(), length_penalty=2.0, repetition_penalty=1.2, no_repeat_ngram_size=3)
    elif task in ["summary"]:

        from transformers import T5ForConditionalGeneration, T5Tokenizer

        TOKENIZERS[task] = T5Tokenizer.from_pretrained(tasks[task])
        free()
        model = T5ForConditionalGeneration.from_pretrained(tasks[task], torch_dtype=dtype()).to(device())

    elif task in ["video"]:

        from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel

        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            tasks[task], subfolder="transformer", torch_dtype=dtype(), revision='refs/pr/18'
        )
        model = HunyuanVideoPipeline.from_pretrained(tasks[task], transformer=transformer, revision='refs/pr/18', torch_dtype=dtype()).to(device())

    elif task in ["image"]:

        from diffusers import DiffusionPipeline
        model = FluxPipeline.from_pretrained(tasks[task], torch_dtype=dtype()).to(device())

    try:
        try:
            model.enable_vae_slicing()
            model.enable_vae_tiling()
        except Exception as e:
            pass
        else:
            try:
                model.enable_model_cpu_offload()
            except Exception as e:
                pass
            if turbo is False:
                model.enable_sequential_cpu_offload()
                model.enable_attention_slicing(1)
    except Exception as e:
        pass

    MODELS[task] = model

    free()

def choose_random_words(word_list, num_words=10):
    if not word_list:
        return []

    list_length = len(word_list)

    if num_words > list_length:
        num_words = list_length - 1

    if num_words == 0:
        num_words = 1
    elif num_words == -1:
        return []

    chosen_words = random.sample(word_list, num_words)
    return chosen_words

def optimize_prompt_realism(prompt):
    prompt = preprocess_prompt(prompt)
    prompt = "reasonable accurate natural convincing real recorded scenario of " + prompt
    return prompt

def preprocess_prompt(prompt):

    if len(prompt) > 0:
        lines = prompt.splitlines()
        prompt = " and ".join(lines)
        prompt = simple_text(prompt)

    return prompt

def pipe(task:str, *a, prompt:str="", path:str="", resolution:str="640x640", length:int=3, fps:int=24):

    import torch
    import cv2
    from PIL import Image
    from diffusers.utils import export_to_video

    params1 = []
    params2 = {}
    if task in ["image","video"]:
        log(f"Pipe activated",prompt,status="")
        width, height = resolution.split("x")
        width, height = int(width), int(height)
        if task == "video":
            length = length*fps
        else:
            length = 1
        params2["prompt"] = prompt
        params2["height"] = height
        params2["width"] = width
        params2["guidance_scale"] = 4.5
        if task == "video":
            params2["num_videos_per_prompt"] = 1
            params2["num_frames"] = length
        else:
            params2["negative_prompt"] = _negative_prompt_
            params2["max_sequence_length"]=512
        params2["num_inference_steps"]=60
        params2["generator"]=torch.Generator(device()).manual_seed(random.randint(0, big_number()))
    elif task == "detect":
        image = Image.open(path)
        params1.append(image)

    from transformers import AutoTokenizer
    if task in ["detect"]:
        tokenizer = AutoTokenizer.from_pretrained(tasks[task])
        inputs = tokenizer(*params1, **params2, return_tensors="tf")
    elif task in ["image","video"]:
        inputs = params2

    try:
        outputs = MODELS[task](**inputs)
    except Exception as e:
        catch(e)
        if task == "image":
            outputs = MODELS["video"](**inputs)
        elif task == "video":
            outputs = MODELS["image"](**inputs)

    if task in ["image","video"]:
        if task == "video":
            sample = outputs.frames[0]
            path = tmp("mp4")
            export_to_video(sample,path,fps=24)
            return path
        else:
            # import imageio as iio
            # if isinstance(sample[0], np.ndarray):
            #    sample = (sample[0] * 255).astype(np.uint8)
            # else:
            #     sample = np.array(sample[0])
            # path = tmp("png")
            # iio.imwrite(path, sample)
            sample = outputs.images[0]
            return save_image(sample)
    elif task == "answer":
        return outputs
    elif task == "detect":
        preds = {}
        if not preds[ pred["label"] ]:
            preds[ pred["label"] ] = []
        for pred in outputs:
            preds[ pred["label"] ].append( pred["box"] )
        return preds

def check_parameter(p):
    return p is not None and not (
        isinstance(p,list) and (
            len(p) == 0 or isinstance(p[0],str) and p[0].strip() == ""
        ) or isinstance(p,str) and p.strip() == ""
    )

def read_mp3(file, normalized = False):
    import pydub

    audio_segment = pydub.AudioSegment.from_mp3(file)
    samples = np.array(audio_segment.get_array_of_samples())
    
    if audio_segment.channels == 2:
        audio_data = samples.reshape((-1, 2)).T
    else:
        audio_data = samples.reshape((1, -1))
        
    if normalized:
        return audio_segment.frame_rate, np.float32(audio_data) / 32768.0
    else:
        return audio_segment.frame_rate, audio_data

def write_mp3(file_path, sr, audio_data):
    if audio_data.ndim == 1:
        channels = 1
    else:
        channels = audio_data.shape[0]

    y = np.int8(
        (audio_data * 128.0 / 2 + 128.0)
        + (audio_data * 128.0 / 2 - 128.0)
    )
    interleaved_data = np.ascontiguousarray(y.T)

    song = pydub.AudioSegment(
        interleaved_data.tobytes(),
        frame_rate=sr,
        sample_width=1,
        channels=channels
    )
    song.export(file_path, format="mp3", bitrate="320k")

def export_to_pkl(model, pkl_path):
    import pickle
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
import numpy as np

def process_audio_chunks(fn, data, chunk_size, overlap=0):
    """
    A stereo-aware function to process audio in overlapping chunks.
    """
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    # Ensure data is float32 for processing
    data = data.astype(np.float32)

    # Handle both mono (1D) and stereo (2D) arrays correctly
    if data.ndim == 1:
        # It's mono, add a channel axis to make it consistent: (n_samples,) -> (1, n_samples)
        data = data[np.newaxis, :]
    
    num_channels, audio_length = data.shape
    step = chunk_size - overlap

    # Create stereo-aware buffers for the final result and window sum
    final_result = np.zeros_like(data, dtype=np.float32)
    window_sum = np.zeros_like(data, dtype=np.float32)

    window = np.hanning(chunk_size)
    # Reshape window for broadcasting with stereo data: (chunk_size,) -> (1, chunk_size)
    window = window[np.newaxis, :]

    start = 0
    while start < audio_length:
        end = min(start + chunk_size, audio_length)
        current_chunk_size = end - start
        
        # Slice all channels for the current time window
        chunk = data[:, start:end]
        
        # Pad the last chunk if it's shorter than chunk_size
        if current_chunk_size < chunk_size:
            padding_size = chunk_size - current_chunk_size
            # Pad only the second axis (the samples), not the channels
            chunk = np.pad(chunk, ((0, 0), (0, padding_size)), 'constant')
        
        # The callback function `fn` receives the raw chunk
        processed_chunk = fn(chunk)

        # Ensure the processed chunk is 2D for consistency
        if processed_chunk.ndim == 1:
            processed_chunk = processed_chunk[np.newaxis, :]
        
        # Apply windowing during the reconstruction (overlap-add)
        final_result[:, start:end] += processed_chunk[:, :current_chunk_size] * window[:, :current_chunk_size]
        window_sum[:, start:end] += window[:, :current_chunk_size]**2
        
        if end == audio_length:
            break
        start += step

    # Avoid division by zero for silent parts
    window_sum[window_sum == 0] = 1.0
    final_result /= window_sum

    return final_result

def str_to_numpy(txt):
    if isinstance(txt, tuple) or isinstance(txt, list):
        txt = "".join(txt)
    vec = create_vectorizer([txt])
    return numpy_to_cupy(vectorize(vec,[txt]))

def one_dim_numpy(v):
    return two_dim_numpy(v).flatten()

def two_dim_numpy(v):
    import torch
    if isinstance(v, torch.Tensor):
        v = v.cpu().numpy()
    elif isinstance(v, str):
        v = str_to_numpy(v)
    elif isinstance(v, np.ndarray):
        if _np.issubdtype(v.dtype, _np.str_):
            v = numpy_to_str(v)
            v = str_to_numpy(v)
        elif not np.issubdtype(v.dtype, np.number):
            raise TypeError(f"CuPy array of dtype {v.dtype} is not supported.")
    elif isinstance(v, (list, tuple)):
        v = np.array(v)
    elif not np.issubdtype(type(v), _np.number):
        try:
            v = np.array(v).astype(float)
        except Exception as e:
            raise TypeError(f"Input of type {type(v)} is not supported: {e}")
    else:
        v = np.array([v])

    if v.ndim == 0:
        return v.reshape(1, 1)
    elif v.ndim == 1:
        return v.reshape(-1, 1)
    elif v.ndim == 2:
        return v
    else:
        try:
            new_shape = (-1, v.shape[-1])
            return v.reshape(new_shape)
        except ValueError as e:
            raise ValueError(f"Cannot reshape array of shape {v.shape} to 2D: {e}")

def three_dim_numpy(v):
    import torch
    if isinstance(v, torch.Tensor):
        v = v.cpu().numpy()
    elif isinstance(v, str):
        v = str_to_numpy(v)
    elif isinstance(v, np.ndarray):
        if _np.issubdtype(v.dtype, _np.str_):
            v = numpy_to_str(v)
            v = str_to_numpy(v)
        elif not np.issubdtype(v.dtype, np.number):
            raise TypeError(f"CuPy array of dtype {v.dtype} is not supported.")
    elif isinstance(v, (list, tuple)):
        v = np.array(v)
    elif not np.issubdtype(type(v), _np.number):
        try:
            v = np.array(v).astype(float)
        except Exception as e:
            raise TypeError(f"Input of type {type(v)} is not supported: {e}")
    else:
        v = np.array([v])

    if v.ndim <= 2:
        return v.reshape(-1, 1, 1)
    elif v.ndim == 3:
        return v
    else:
        try:
            new_shape = (-1, v.shape[-2], v.shape[-1])
            return v.reshape(new_shape)
        except ValueError as e:
            raise ValueError(f"Cannot reshape array of shape {v.shape} to 3D: {e}")

def resize_video(input_video_path, target_height, target_width, anti_aliasing=True):
    """
    Resizes a video using skimage.transform.resize.

    Args:
        input_video_path (str): Path to the input video file.
        target_height (int): The desired height in pixels.
        target_width (int): The desired width in pixels.
        anti_aliasing (bool, optional): Whether to apply anti-aliasing. Defaults to True.
    """

    output_video_path = tmp("mp4")

    try:
        reader = iio.imiter(input_video_path)
        metadata = reader.metadata()
        fps = metadata['fps']

        writer = iio.imwriter(output_video_path, fps=fps)

        for frame in reader:
            resized_frame = resize(frame, (target_height, target_width), anti_aliasing=anti_aliasing)
            writer.append_data((resized_frame * 255).astype(np.uint8)) #Save the frame as uint8.

        writer.close()
        reader.close()

        return output_video_path

    except FileNotFoundError:
        print(f"Error: Video file not found at {input_video_path}")
    except Exception as e:
        print(f"An error occurred during video resizing: {e}")

def resize_image(image_data, target_height, target_width, anti_aliasing=True):
    """
    Resizes an image using skimage.transform.resize.

    Args:
        image_data (np.ndarray): The image data as a NumPy array (e.g., from iio.imread).
        target_height (int): The desired height in pixels.
        target_width (int): The desired width in pixels.
        anti_aliasing (bool, optional): Whether to apply anti-aliasing. Defaults to True.

    Returns:
        np.ndarray: The resized image data.
    """

    import imageio as iio
    from skimage.transform import resize

    try:
        if image_data.ndim < 2:
            raise ValueError("Input image must have at least 2 dimensions (height, width).")

        resized_image = resize(image_data, (target_height, target_width), anti_aliasing=anti_aliasing)
        return resized_image

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None

    except Exception as e:
        print(f"An error occurred during resizing: {e}")
        return None

def numpy_to_list(np_arr):
    return np.concatenate(np_arr, axis=None).ravel().tolist()

def guess_numpy_sample_rate(audio_data, possible_sample_rates=None, 
                         window_type='hann', window_size=None, 
                         peak_prominence=0.01, peak_distance=10,
                         frequency_threshold=0.05):
    """
    מעריכה את קצב הדגימה של אודיו באמצעות ניתוח תדרים.

    Args:
        audio_data (np.ndarray): מערך NumPy של נתוני האודיו.
        possible_sample_rates (list, optional): רשימה של קצבי דגימה אפשריים. 
            אם None, ינסה למצוא התאמה מתוך טווח רחב.
        window_type (str, optional): סוג חלון לשימוש (hann, hamming, etc.).
        window_size (int, optional): גודל החלון. אם None, ייבחר אוטומטית.
        peak_prominence (float, optional): מינימום prominence לזיהוי פסגות.
        peak_distance (int, optional): מינימום מרחק בין פסגות.
        frequency_threshold (float, optional): סף להתאמה בין תדרים (יחסית לתדר נייקוויסט).

    Returns:
        int: קצב הדגימה המשוער, או None אם לא נמצאה התאמה.
    """

    from scipy import signal

    audio_data = cupy_to_numpy(audio_data)

    # 1. הכנת חלון וחישוב ספקטרום
    if window_size is None:
        window_size = len(audio_data)
    window = signal.get_window(window_type, window_size)
    frequencies = _np.fft.fftfreq(window_size, d=1.0) # d=1.0 כי אנחנו עובדים עם דגימות ולא עם זמן
    spectrum = _np.abs(_np.fft.fft(audio_data[:window_size] * window)) # לוקחים רק חלק מהמידע אם קיים

    # 2. זיהוי תדרים דומיננטיים
    peak_indices = signal.find_peaks(spectrum, prominence=peak_prominence, distance=peak_distance)[0]
    dominant_frequencies = frequencies[peak_indices]

    # 3. הערכת קצב הדגימה
    if possible_sample_rates is None:
        possible_sample_rates = [22050, 44100, 48000, 88200, 96000, 192000]
    
    for sr in possible_sample_rates:
        nyquist_frequency = sr / 2
        for freq in dominant_frequencies:
            if abs(freq) < nyquist_frequency and abs(freq - round(freq)) / nyquist_frequency < frequency_threshold:
                return sr
    return None

def guess_numpy_type(data):

    np_list = numpy_to_list(data)
    mean = np.mean(data)
    std = np.std(data)
    ratio = std / mean if mean != 0 else float('inf')
    if data.shape and len(data.shape) > 3:
        return "video"
    elif data.shape and len(data.shape) > 2:
        return "image"
    elif str(data.dtype)[1] in ["U","S"]:
        return "text"
    elif data.ndim > 1 or str(data.dtype)[1] in ["f"] or ratio > 1:
        return "audio"
    else:
        return "text"

def cupy_to_numpy(v: Any) -> Any:
    try:
        import cupy as cp
        return cp.asnumpy(v)
    except Exception:
        return v

def numpy_to_cupy(v: Any) -> Any:
    try:
        import cupy as cp
        return cp.array(v)
    except Exception:
        return v

def get_max_shapes(*data):

        lengths = []

        shapes = [np_arr.shape for np_arr in data]

        for sh in shapes:
            l = len(lengths)
            while l < len(sh):
                lengths.append(0)
                l = len(lengths)
            for i,dim in enumerate(sh):
                lengths[i] = max(lengths[i],dim)

        return lengths

def pad_nested(nested_data, lengths, fill_value=0):
    if isinstance(nested_data, _np.ndarray):
        nested_data = nested_data.tolist()
    elif isinstance(nested_data, tuple):
        nested_data = list(nested_data)

    if not nested_data:
        return [fill_value] * lengths[0]

    if not isinstance(nested_data[0], list):
        data_len = len(nested_data)
        diff = lengths[0] - data_len
        if diff > 0:
            nested_data.extend([fill_value] * diff)
        return nested_data

    ret = []
    for arr in nested_data:
        ret.append(pad_nested(arr, lengths[1:], fill_value))

    data_len = len(ret)
    diff = lengths[0] - data_len
    if diff > 0:
        ret.extend([pad_nested([], lengths[1:], fill_value)] * diff)

    return ret

def reshape_numpy(data, fill_value = 0, lengths = None):

    if isinstance(data, _np.ndarray):
        data = data.tolist()

    if not data:
        return _np.array([])

    try:
        if lengths is None:
            lengths = get_max_shapes(data)

        log("Reshaping data",lengths)
        reshaped_data = pad_nested(data, lengths)
        log("Reshaped data",lengths)

        return _np.array(reshaped_data)

    except (TypeError, IndexError) as e:
        catch(e)
        return _np.array([])
    except Exception as e2:
        catch(e2)
        return _np.array([])

def convert_tensor_dtype(tensor):
    import torch

    if tensor.is_floating_point():
        if tensor.dtype == torch.float64:
            return tensor.to(torch.float32)
        else:
            return tensor
    elif torch.is_floating_point(tensor) == False:
        max_val = tensor.max()
        min_val = tensor.min()

        if min_val >= 0: #unsigned int
            if max_val <= 255:
                return tensor.to(torch.uint8)
            elif max_val <= 65535:
                return tensor.to(torch.uint16)
            elif max_val <= 4294967295:
                return tensor.to(torch.uint32)
            else:
                return tensor.to(torch.uint64)
        else: #signed int
            if min_val >= -128 and max_val <= 127:
                return tensor.to(torch.int8)
            elif min_val >= -32768 and max_val <= 32767:
                return tensor.to(torch.int16)
            elif min_val >= -2147483648 and max_val <= 2147483647:
                return tensor.to(torch.int32)
            else:
                return tensor.to(torch.int64)

    else:
        return tensor

def get_active_audio_timeline(audio_file, threshold_db=-16, min_silence_len=0.1):
    """
    Gets the start and end times of each non-silence audio part.

    Args:
        audio_file (str): Path to the audio file.
        threshold_db (float): Silence threshold in dB.
        min_silence_len (float): Minimum silence length in seconds.

    Returns:
        list: A list of tuples, where each tuple contains (start_time, end_time) of active audio.
    """

    import librosa

    audio_data, sample_rate = librosa.load(audio_file, sr=32000)
    silence_mask = detect_silence_mask(audio_data, sample_rate, threshold_db, min_silence_len)

    # Find active audio regions
    active_regions = librosa.effects.split(_np.logical_not(silence_mask).astype(float), frame_length=1, hop_length=1)

    # Convert sample indices to time
    timeline = [(
        start.item() / int(sample_rate),
        end.item() / int(sample_rate)
    ) for start, end in active_regions]
    return timeline

def detect_silence_mask(audio_data, sample_rate, threshold_db=-16, min_silence_len=0.1):
    """Detects silence in an audio signal and creates a silence mask."""

    import librosa

    threshold_amplitude = librosa.db_to_amplitude(threshold_db)
    frame_length = int(0.02 * sample_rate)
    hop_length = frame_length // 4
    rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
    silence_mask_rms = rms < threshold_amplitude
    silence_mask = np.repeat(silence_mask_rms, hop_length)
    if len(silence_mask) > len(audio_data):
        silence_mask = silence_mask[:len(audio_data)]
    elif len(silence_mask) < len(audio_data):
        padding = np.ones(len(audio_data) - len(silence_mask), dtype=bool)
        silence_mask = np.concatenate((silence_mask, padding))
    min_silence_samples = int(min_silence_len * sample_rate)
    silence_mask_filtered = silence_mask.copy()
    silence_regions = librosa.effects.split(silence_mask.astype(float), top_db=0.5)
    for start, end in silence_regions:
        if end - start < min_silence_samples:
            silence_mask_filtered[start:end] = False
    return silence_mask_filtered

def convert_video_fps(input_video_path, target_fps):
    """
    Converts a video's 24 to a target 24.

    Args:
        input_video_path (str): Path to the input video file.
        target_fps (float): The desired target 24.
    """

    output_video_path = tmp("mp4")

    try:
        reader = iio.imiter(input_video_path)
        metadata = reader.metadata()
        original_fps = metadata['fps']
        frames = list(reader)
        reader.close()

        if original_fps == target_fps:
            # No conversion needed
            iio.imwrite(output_video_path, frames, fps=target_fps)
            return

        ratio = target_fps / original_fps
        new_frames = []
        for i in np.arange(0, len(frames), 1 / ratio):
            index = int(i)
            if index < len(frames):
                new_frames.append(frames[index])

        iio.imwrite(output_video_path, new_frames, fps=target_fps)

        return output_video_path

    except FileNotFoundError:
        print(f"Error: Video file not found at {input_video_path}")
    except Exception as e:
        print(f"An error occurred during 24 conversion: {e}")

def write_video(video_data, fps):
    """
    Writes a video file using imageio.

    Args:
        video_data (list): A list of NumPy arrays, where each array represents a frame.
        fps (int, optional): Frames per second.
    """

    output_path = tmp("mp4")

    try:
        writer = iio.imwriter(output_path, fps=fps)
        for frame in video_data:
            writer.append_data(frame)
        writer.close()
        return output_path
    except Exception as e:
        print(f"An error occurred during video writing: {e}")

def read_video(video_path):
    """
    Reads a video file using imageio.

    Args:
        video_path (str): Path to the video file.

    Returns:
        tuple: A tuple containing the video data as a NumPy array and the video metadata.
               Returns (None, None) if an error occurs.
    """
    try:
        reader = iio.imiter(video_path)
        metadata = reader.metadata()
        video_data = list(reader)  # Convert to a list of frames
        reader.close()
        return metadata, video_data
    except FileNotFoundError:
        print(f"Error: Video file not found at {video_path}")
        return None, None
    except Exception as e:
        print(f"An error occurred during video reading: {e}")
        return None, None

def is_gpu():
    import torch
    return torch.cuda.is_available()

def check_onnx(path):
    import onnx
    model = onnx.load(path)
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError:
        return False
    return True

def pytorch_to_onnx(model_torch, input_dim, onnx_path="model.onnx"):
    import torch
    dummy_input = torch.randn(1, input_dim).cuda()
    torch.onnx.export(model_torch, dummy_input, onnx_path, verbose=True)
    print("ONNX export complete!")

def compress(dir:str, format:str="zip", keep_name:bool=True):
    if keep_name:
        target = str(Path(dir).parent) + "/" + str(Path(dir).name)
    else:
        target = str(Path(dir).parent) + "/" + random_string()
    shutil.make_archive( target , format, str(Path(dir).parent), str(Path(dir).name) )
    return target + "." + format

def extract(arcv, dest = None, format = None):
    if not dest:
        dest = str(Path(arcv).parent)
    if format:
        shutil.unpack_archive( arcv, dest, format )
    else:
        shutil.unpack_archive( arcv, dest )

class HybridModel:
    def __init__(self):
        self.model = None

    def fit(self, X, y=None):
        if y is not None: #Supervised
            from cuml.linear_model import LinearRegression as cuLinearRegression

            if self.model is None:
                self.model = cuLinearRegression()

            # Training
            start_train = time()
            self.model.fit(X, y)
            np.cuda.runtime.deviceSynchronize()
            end_train = time()
            train_time = end_train - start_train

            print(f"Train Time: {train_time:.4f} seconds")

        else: #Unsupervised
            from cuml.cluster import KMeans as cuKMeans

            if self.model is None:
                self.model = cuKMeans(n_clusters=32768)

            # Training
            start_train = time()
            self.model.fit(X)
            np.cuda.runtime.deviceSynchronize()
            end_train = time()
            train_time = end_train - start_train

            print(f"Train Time: {train_time:.4f} seconds")

    def predict(self, X):
        """
        Predicts using a trained hybrid model.
        """

        if self.model is None:
            raise ValueError("Model must be trained before prediction.")

        start_predict = time()

        predictions = self.model.predict(X)

        np.cuda.runtime.deviceSynchronize()
        end_predict = time()
        predict_time = end_predict - start_predict
        predictions = cupy_to_numpy(predictions)

        print(f"Predict Time: {predict_time:.4f} seconds")

        return predictions

class BeamSearch:
    import torch
    import torch.nn.functional as F

    def __init__(self, model, tokenizer, processor, device, length_penalty: float = 1.0, score_function = None):

        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device
        self.eos_token_id = tokenizer.eos_token_id
        self.length_penalty = length_penalty
        self.score_function = score_function or self._default_score_function

    def _default_score_function(self, _arg) -> float:
        beam, total_score = _arg
        seq, _ = beam[-1]
        seq_len = seq.shape[1]
        return total_score / (seq_len ** self.length_penalty)

    def search(self, input_ids: torch.Tensor, max_length: int, beam_width: int) -> torch.Tensor:

        input_ids = input_ids.to(self.device)
        beams = [([(input_ids, 0.0)], 0.0)]

        for _ in range(max_length - input_ids.shape[1]):
            new_beams = []
            for beam, total_score in beams:
                seq, score = beam[-1]
                if self.eos_token_id is not None and seq[0, -1].item() == self.eos_token_id:
                    new_beams.append((beam, total_score))
                    continue

                with torch.no_grad():
                    outputs = self.model(seq)
                    logits = outputs.logits[:, -1, :]
                    probs = F.log_softmax(logits, dim=-1)

                topk_probs, topk_indices = torch.topk(probs, beam_width)
                for i in range(beam_width):
                    new_seq = torch.cat([seq, topk_indices[:, i].unsqueeze(-1)], dim=-1)
                    new_score = score + topk_probs[:, i].item()
                    new_beams.append((beam + [(new_seq, new_score)], total_score + topk_probs[:, i].item()))

            beams = sorted(new_beams, key=self.score_function, reverse=True)[:beam_width]
            if self.eos_token_id is not None and all(beam[-1][0][0, -1].item() == self.eos_token_id for beam, _ in beams):
                break

        best_beam, _ = beams[0]
        best_seq, _ = best_beam[-1]
        return best_seq.cpu()

    def generate(self, prompt: str, max_length: int, beam_width: int, **kw) -> str:

        import torch.nn.modules.module as module

        inputs = self.processor(prompt, return_tensors="pt", **kw).to(self.device)

        input_ids = inputs["input_ids"]
        beam_ids = self.search(input_ids, max_length, beam_width)
        beam_ids = two_dim_numpy(beam_ids)
        beam_ids = torch.from_numpy(beam_ids).to(self.device)

        inputs["input_ids"] = beam_ids

        original_requires_grad_ = module.Module.requires_grad_

        def no_grad_requires_grad_(self, requires_grad=True):
            pass

        module.Module.requires_grad_ = no_grad_requires_grad_

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_logits_to_keep=0,
            )

        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]

        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        module.Module.requires_grad_ = original_requires_grad_

        log("Response",response)

        return response

def LinearRegressionTorch(input_dim):
    import torch
    class _LinearRegressionTorch(torch.nn.Module):
        def __init__(self, input_dim):
            super(LinearRegressionTorch, self).__init__()
            self.linear = torch.nn.Linear(input_dim, 1)
        def forward(self, x):
            return self.linear(x)
    return _LinearRegressionTorch(input_dim)

def SklearnWrapper(sklearn_model, is_classification=False):
    import torch

    class _SklearnWrapper(torch.nn.Module):
        def __init__(self, sklearn_model, is_classification=False):
            super().__init__()
            self.sklearn_model = sklearn_model
            self.is_classification = is_classification

        def forward(self, x, y=None, y_mask=None):
            x_numpy = self._to_numpy(x)
            if hasattr(self.sklearn_model, "predict_proba") and self.is_classification:
                predictions = self.sklearn_model.predict_proba(x_numpy)
            elif hasattr(self.sklearn_model, "decision_function") and self.is_classification:
                predictions = self.sklearn_model.decision_function(x_numpy)
            else:
                predictions = self.sklearn_model.predict(x_numpy)
            return torch.tensor(predictions, dtype=torch.float32, device=x.device)

        def fit(self, x, y=None):
            x_numpy = self._to_numpy(x)
            y_numpy = self._to_numpy(y) if y is not None else None

            if y_numpy is not None:
                self.sklearn_model.fit(x_numpy, y_numpy)
            else:
                if len(x_numpy.shape) > 2:
                    logging.warning("Fitting model on 3D input without labels. Fitting on each sequence independently.")
                    for i in range(x_numpy.shape[0]):
                        self.sklearn_model.fit(x_numpy[i])
                else:
                    self.sklearn_model.fit(x_numpy)

        def _to_numpy(self, tensor_or_array):
            if tensor_or_array is None:
                return None
            if isinstance(tensor_or_array, np.ndarray):
                return tensor_or_array
            if isinstance(tensor_or_array, torch.Tensor):
                return tensor_or_array.cpu().numpy()
            raise ValueError(f"Expected torch.Tensor or numpy.ndarray, got {type(tensor_or_array)}")

    return _SklearnWrapper(sklearn_model, is_classification)

def add_chat_message(history, message):
    for x in message["files"]:
        history.append({"role": "user", "content": {"path": x}})
    if message["text"] is not None:
        txt = message["text"]
        history.append({"role": "user", "content": txt})
    return history

def get_chat_response(message, history: list):
    history = add_chat_message(history, message)
    response = answer(history)
    return response

def init_chat( title:str, high_performance:bool = True ):
    import gradio as gr

    if not MODELS["answer"]:
        init_pretrained_model( "answer", high_performance )

    if not MODELS["summary"]:
        init_pretrained_model( "summary", high_performance )

    chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False, type="messages")
    return gr.ChatInterface(fn=get_chat_response, type="messages", chatbot=chatbot, multimodal=True, theme=gr.themes.Citrus(), title=title, css=css(), save_history=True, show_progress="full")

def download_file(url, destination):
    import requests
    try:
        print(f"Downloading from {url} to {destination}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download successful.")
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

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
        print(f"Successfully added {folder_path} to PATH. Please restart your terminal for changes to take effect.")
    else:
        print(f"Failed to add {folder_path} to PATH.")

def rvc_to_onnx(model_path):
    if not os.path.exists("infer") and not os.path.exists("infer/"):
        logger.info("Infer module not found, downloading...")
        google_drive_download( id="1kqMYQskvVKwKglcWQsK2Q5G3yPahnbtH", dest='./infer.zip' )

    try:
        from .infer.modules.onnx.export import export_onnx as eo
        eo( model_path, model_path.replace( ".pth" , "" )+".onnx" )
        logger.info("ONNX export complete.")
        return model_path.replace( ".pth" , "" )+".onnx"
    except ImportError:
        logger.error("Failed to import ONNX export module. Ensure 'infer' directory is correctly set up.")
        catch(ImportError("Failed to import ONNX export module."))
    except Exception as e:
        logger.error(f"An error occurred during ONNX export!")
        catch(e)

def export_files_rvc(experiment: str):
    logger.info(f"Exporting files for experiment: {experiment}")
    now_dir = os.getcwd()
    weight_root = os.path.join(now_dir, "assets", "weights")
    index_root = os.path.join(now_dir, "logs")
    exp_path = os.path.join(index_root, experiment)

    latest_checkpoint_filename = find_latest_checkpoint(weight_root, experiment)

    if latest_checkpoint_filename is None:
        error_message = f"Error: No latest checkpoint found for experiment '{experiment}' in '{exp_path}'. Cannot export."
        logger.error(error_message)
        return []

    pth_path = os.path.join(weight_root, latest_checkpoint_filename)
    logger.info(f"Found latest checkpoint: {pth_path}")

    index_file = ""
    for root, dirs, files in os.walk(exp_path, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_file = os.path.join(root, name)
                logger.info(f"Found index file: {index_file}")
                break
        if index_file:
            break

    onnx_path = rvc_to_onnx(pth_path)

    exported_files = [pth_path]
    if os.path.exists(onnx_path):
        exported_files.append(onnx_path)
        logger.info(f"Added ONNX file to exported list: {onnx_path}")
    else:
        logger.warning(f"ONNX file not found after export attempt: {onnx_path}")

    if os.path.exists(index_file):
        exported_files.append(index_file)
        logger.info(f"Added index file to exported list: {index_file}")
    else:
         logger.warning(f"Index file not found: {index_file}")


    logger.info(f"Exported files: {exported_files}")
    return exported_files


def find_latest_checkpoint(folder_path: str, model_name: str) -> str | None:
    logger.info(f"Searching for latest checkpoint in '{folder_path}' with model name '{model_name}'")
    if not os.path.isdir(folder_path):
        logger.error(f"Error: Folder not found at {folder_path}")
        return None

    pattern = re.compile(rf"^{re.escape(model_name)}_e(\d+)_s(\d+)\.pth$")

    latest_checkpoint = None
    latest_epoch = -1
    latest_global_step = -1

    try:
        for filename in os.listdir(folder_path):
            match = pattern.match(filename)
            if match:
                epoch = int(match.group(1))
                global_step = int(match.group(2))

                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_global_step = global_step
                    latest_checkpoint = filename
                elif epoch == latest_epoch and global_step > latest_global_step:
                    latest_global_step = global_step
                    latest_checkpoint = filename

    except Exception as e:
        logger.error(f"An error occurred while scanning the folder for checkpoints: {e}")
        return None

    if latest_checkpoint:
        logger.info(f"Latest checkpoint found: {latest_checkpoint}")
    else:
        logger.warning(f"No checkpoint found matching the pattern in '{folder_path}'")

    return latest_checkpoint


def train_model_rvc(experiment: str, path: str, lvl: int = 1):
    logger.info(f"Starting RVC training for experiment: {experiment}")

    import torch

    from .configs.config import Config
    from .i18n.i18n import I18nAuto

    now_dir = os.getcwd()
    index_root = os.path.join(now_dir, "logs")

    config = Config()

    gpus = "-".join([str(i) for i in range(torch.cuda.device_count())]) if torch.cuda.is_available() else ""
    gpu_memories = [int(torch.cuda.get_device_properties(i).total_memory / 1024**3 + 0.4) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [0]
    default_batch_size = math.floor(min(gpu_memories) // 2) if gpu_memories and min(gpu_memories) > 0 else 1
    if default_batch_size == 0:
        default_batch_size = 1

    exp_dir = experiment
    exp_path = os.path.join(index_root, exp_dir)
    logger.info(f"Experiment directory: {exp_path}")

    directory(os.path.join(exp_path, "1_16k_wavs"))
    directory(os.path.join(exp_path, "0_gt_wavs"))
    input_root = os.path.join(exp_path, "input_root")
    directory(input_root)

    input_path = os.path.join(input_root, "input.wav")
    logger.info(f"Moving input audio '{path}' to '{input_path}'")
    try:
        move(path, input_path)
    except Exception as e:
        logger.error(f"Failed to move input audio file: {e}")
        catch(e)
        return None

    filelist_path = os.path.join(exp_path,"filelist.txt")
    logger.info(f"Creating filelist: {filelist_path}")
    try:
        write(filelist_path)

    except Exception as e:
        logger.error(f"Failed to create filelist.txt: {e}")
        catch(e)
        return None

    sr = 48000
    n_p = int(_np.ceil(config.n_cpu / 1.5))
    log_file_preprocess = os.path.join(exp_path, "preprocess.log")

    f0method = "harvest"
    if_f0 = True
    gpus_rmvpe = f"{gpus}-{gpus}"
    log_file_f0_feature = os.path.join(exp_path, "extract_f0_feature.log")

    logger.info("Starting preprocessing...")
    try:
        with open(log_file_preprocess, 'w') as f_preprocess:
            cmd_preprocess = f'"{config.python_cmd}" infer/modules/train/preprocess.py "{input_root}" {sr} {n_p} "{exp_path}"'
            logger.info("Execute: " + cmd_preprocess)
            subprocess.run(cmd_preprocess, shell=True, check=True, stdout=f_preprocess, stderr=subprocess.STDOUT)

        with open(log_file_preprocess, 'r') as f_preprocess:
            log_content = f_preprocess.read()
            logger.info("Preprocessing Log:\n" + log_content)

    except subprocess.CalledProcessError as e:
        logger.error(f"Preprocessing failed with return code {e.returncode}: {e}")
        logger.error(f"Preprocessing output:\n{e.stdout.decode() if e.stdout else 'N/A'}\n{e.stderr.decode() if e.stderr else 'N/A'}")
        catch(e)
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during preprocessing: {e}")
        catch(e)
        return None
    logger.info("Preprocessing complete.")

    logger.info("Starting feature extraction...")
    try:
        with open(log_file_f0_feature, 'w') as f_f0_feature:
            if if_f0:
                logger.info(f"Extracting F0 using method: {f0method}")
                if f0method != "rmvpe_gpu":
                    cmd_f0 = f'"{config.python_cmd}" infer/modules/train/extract/extract_f0_print.py "{exp_path}" {n_p} {f0method}'
                    logger.info("Execute: " + cmd_f0)
                    subprocess.run(cmd_f0, shell=True, check=True, stdout=f_f0_feature, stderr=subprocess.STDOUT)
                else:
                    gpus_rmvpe_split = gpus_rmvpe.split("-")
                    leng = len(gpus_rmvpe_split)
                    ps = []
                    logger.info(f"Using {leng} GPUs for RMVPE extraction: {gpus_rmvpe_split}")
                    for idx, n_g in enumerate(gpus_rmvpe_split):
                        cmd_f0_rmvpe = f'"{config.python_cmd}" infer/modules/train/extract/extract_f0_rmvpe.py {leng} {idx} {n_g} "{exp_path}" {config.is_half}'
                        logger.info(f"Execute (GPU {n_g}): " + cmd_f0_rmvpe)
                        p = thread(run, cmd_f0_rmvpe)
                        ps.append(p)
                    wait(*ps)

            logger.info("Extracting features...")
            leng = len(gpus.split("-"))
            ps = []
            logger.info(f"Using {leng} GPUs for feature extraction: {gpus.split('-')}")
            for idx, n_g in enumerate(gpus.split("-")):
                cmd_feature_print = f'"{config.python_cmd}" infer/modules/train/extract_feature_print.py {config.device} {leng} {idx} "{exp_path}" v2'
                logger.info(f"Execute (GPU {n_g}): " + cmd_feature_print)
                p = thread(run, cmd_feature_print)
                ps.append(p)
            wait(*ps)

        with open(log_file_f0_feature, 'r') as f_f0_feature:
            log_content = f_f0_feature.read()
            logger.info("F0 and Feature Extraction Log:\n" + log_content)

    except Exception as e:
        logger.error(f"An error occurred during F0 or feature extraction: {e}")
        catch(e)
        return None
    logger.info("Feature extraction complete.")

    logger.info("Starting index training...")
    feature_dir = os.path.join(exp_path, "3_feature768")
    listdir_res = []
    if os.path.exists(feature_dir):
        listdir_res = os.listdir(feature_dir)

    if not os.path.exists(feature_dir) or not any(listdir_res):
        error_message = f"Error: Feature directory '{feature_dir}' is missing or empty! Cannot train index."
        logger.error(error_message)
        return None

    try:
        npys = []
        for name in sorted(listdir_res):
             if name.endswith('.npy'):
                phone = _np.load(os.path.join(feature_dir, name))
                npys.append(phone)

        if not npys:
            error_message = f"Error: No .npy files found in '{feature_dir}'! Cannot train index."
            logger.error(error_message)
            return None

        big_npy = _np.concatenate(npys, 0)
        logger.info(f"Concatenated features shape: {big_npy.shape}")

        big_npy_idx = _np.arange(big_npy.shape[0])
        _np.random.shuffle(big_npy_idx)
        big_npy = big_npy[big_npy_idx]

    except Exception as e:
        logger.error(f"An error occurred while loading and concatenating features for index training: {e}")
        catch(e)
        return None

    try:
        from sklearn.cluster import MiniBatchKMeans
        big_npy = MiniBatchKMeans(
            n_clusters=8000,
            verbose=False,
            batch_size=256 * config.n_cpu,
            compute_labels=False,
            init="random",
            n_init=3
        ).fit(big_npy).cluster_centers_
        logger.info(f"KMeans cluster centers shape: {big_npy.shape}")

        import faiss
        feature_dimension = big_npy.shape[1]
        n_ivf = min(int(16 * _np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
        n_ivf = max(1, n_ivf)
        logger.info(f"Training Faiss index with dimension {feature_dimension} and n_ivf {n_ivf}")

        index = faiss.index_factory(feature_dimension, f"IVF{n_ivf},Flat")
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = 1
        logger.info(f"Faiss index nprobe set to: {index_ivf.nprobe}")

        logger.info("Training Faiss index...")
        index.train(big_npy)
        logger.info("Faiss index training complete.")

        trained_index_path = os.path.join(exp_path, f"trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir}_v2.index")
        faiss.write_index(index, trained_index_path)
        logger.info(f"Trained Faiss index saved to: {trained_index_path}")


        logger.info("Adding features to Faiss index...")
        batch_size_add = 8192
        for i in range(0, big_npy.shape[0], batch_size_add):
            index.add(big_npy[i:i + batch_size_add])
        logger.info("Features added to Faiss index.")

        added_index_filename = f"added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir}_v2.index"
        added_index_path = os.path.join(exp_path, added_index_filename)
        faiss.write_index(index, added_index_path)
        logger.info(f"Final Faiss index saved to: {added_index_path}")

        target_link_path = os.path.join(index_root, added_index_filename)
        logger.info(f"Creating link from '{added_index_path}' to '{target_link_path}'")
        try:
            if os.path.exists(target_link_path) or os.path.islink(target_link_path):
                 os.remove(target_link_path)
                 logger.warning(f"Removed existing file/link at {target_link_path}")

            if platform.system() != "Windows":
                os.symlink(added_index_path, target_link_path)
            else:
                os.link(added_index_path, target_link_path)
            logger.info("Index linking successful.")
        except Exception as e:
            logger.error(f"Linking index failed: {e}")
            catch(e)

    except Exception as e:
        logger.error(f"An error occurred during index training: {e}")
        catch(e)
        return None
    logger.info("Index training complete.")

    logger.info("Starting model training...")
    try:
        pretrained_G = "assets/pretrained_v2/f0G48k.pth"
        pretrained_D = "assets/pretrained_v2/f0D48k.pth"

        batch_size = default_batch_size
        total_epoch = 40 * lvl
        save_epoch = 10
        if_save_latest = 1
        if_cache_gpu = 1
        if_save_every_weights = 1
        gpus_str = gpus

        config_path = "v2/48k.json"
        config_save_path = os.path.join(exp_path, "config.json")

        if not pathlib.Path(config_save_path).exists():
             logger.info(f"Saving training config to: {config_save_path}")
             try:
                 with open(config_save_path, "w", encoding="utf-8") as f:
                     json.dump(config.json_config.get(config_path, {}), f, ensure_ascii=False, indent=4, sort_keys=True)
                     f.write("\n")
             except Exception as e:
                  logger.error(f"Failed to save training config file: {e}")
                  catch(e)

        log_file_train = os.path.join(exp_path, "train.log")

        logger.info("Executing training command...")
        with open(log_file_train, 'w') as f_train:
            cmd_train = (
                f'"{config.python_cmd}" infer/modules/train/train.py '
                f'-e "{exp_dir}" '
                f'-sr 48k '
                f'-f0 1 '
                f'-bs {batch_size} '
                f'-g {gpus_str} '
                f'-te {total_epoch} '
                f'-se {save_epoch} '
                f'-pg "{pretrained_G}" '
                f'-pd "{pretrained_D}" '
                f'-l {if_save_latest} '
                f'-c {if_cache_gpu} '
                f'-sw {if_save_every_weights} '
                f'-v v2'
            )
            logger.info("Execute: " + cmd_train)
            subprocess.run(cmd_train, shell=True, check=True, stdout=f_train, stderr=subprocess.STDOUT)

        with open(log_file_train, 'r') as f_train:
            log_content = f_train.read()
            logger.info("Training Log:\n" + log_content)

    except subprocess.CalledProcessError as e:
        logger.error(f"Model training failed with return code {e.returncode}: {e}")
        logger.error(f"Training output:\n{e.stdout.decode() if e.stdout else 'N/A'}\n{e.stderr.decode() if e.stderr else 'N/A'}")
        catch(e)
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during model training: {e}")
        catch(e)
        return None
    logger.info("Model training complete.")

    logger.info("Training complete, exporting files...")
    return export_files_rvc(exp_dir)

def convert_vocal_rvc(experiment: str, path: str, semi_tones: int = 0):
    logger.info(f"Starting vocal conversion for experiment: {experiment} with pitch shift {semi_tones}")

    from .infer.modules.vc.modules import VC
    from .configs.config import Config

    now_dir = os.getcwd()
    index_root = os.path.join(now_dir, "logs")
    weight_root = os.path.join(now_dir, "assets", "weights")

    config = Config()
    vc = VC(config)

    exp_path = os.path.join(index_root, experiment)

    latest_checkpoint_filename = find_latest_checkpoint(weight_root, experiment)

    if latest_checkpoint_filename is None:
        error_message = f"Error: No latest checkpoint found for experiment '{experiment}' in '{exp_path}'. Cannot perform conversion."
        logger.error(error_message)
        return None

    pth_path = os.path.join(weight_root, latest_checkpoint_filename)
    logger.info(f"Using model checkpoint: {pth_path}")

    idx_path = None
    for root, dirs, files in os.walk(exp_path, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                idx_path = os.path.join(root, name)
                logger.info(f"Found index file: {idx_path}")
                break
        if idx_path:
            break

    if idx_path is None:
        logger.warning(f"No index file found for experiment '{experiment}' in '{exp_path}'. Conversion may be less effective.")

    index_rate = 0.5
    f0_mean_pooling = 1
    try:
        vc.get_vc(latest_checkpoint_filename, index_rate, f0_mean_pooling)
        logger.info("VC model loaded.")
    except Exception as e:
        logger.error(f"Failed to load VC model: {e}")
        catch(e)
        return None

    try:
        message, (sr, aud) = vc.vc_single(
            0,
            path,
            semi_tones,
            None,
            "crepe",
            idx_path,
            None,
            index_rate,
            3,
            0,
            0.5,
            0.7
        )
        logger.info(f"Vocal conversion message: {message}")

    except Exception as e:
        logger.error(f"An error occurred during vocal conversion: {e}")
        catch(e)
        return None

    if aud is not None and isinstance(aud, _np.ndarray) and aud.size > 0:
        logger.info(f"Conversion successful, saving output audio with shape {aud.shape} at sample rate {sr}")
        try:
            temp_dir = os.path.join(now_dir, "TEMP")
            directory(temp_dir)
            out_path = os.path.join(temp_dir, "output.wav")

            import soundfile as sf
            sf.write(out_path, aud, sr)
            logger.info(f"Output audio saved to: {out_path}")
            return out_path

        except Exception as e:
            logger.error(f"Failed to save output audio file: {e}")
            catch(e)
            return None
    else:
        logger.warning("Vocal conversion did not produce valid audio data.")
        return None

def export_audio(audio_segment, output_path_stem, format_choice):
    format_lower = format_choice.lower()
    if "mp3" in format_lower: file_format, bitrate, suffix = "mp3", "320k", ".mp3"
    elif "wav" in format_lower: file_format, bitrate, suffix = "wav", None, ".wav"
    elif "flac" in format_lower: file_format, bitrate, suffix = "flac", None, ".flac"
    else: raise ValueError(f"Unsupported format: {format_choice}")
    output_path = Path(output_path_stem).with_suffix(suffix)
    params = ["-acodec", "pcm_s16le"] if file_format == "wav" else None
    audio_segment.export(str(output_path), format=file_format, bitrate=bitrate, parameters=params)
    return str(output_path)

def create_share_links(hf_username, space_name, file_path, text_description):
    file_url = f'https://{hf_username}-{space_name}.hf.space/gradio_api/file={file_path}'
    encoded_text = quote(text_description)
    encoded_url = quote(file_url)
    twitter_link = f"https://twitter.com/intent/tweet?text={encoded_text}&url={encoded_url}"
    facebook_link = f"https://www.facebook.com/sharer/sharer.php?u={encoded_url}"
    reddit_link = f"https://www.reddit.com/submit?url={encoded_url}&title={encoded_text}"
    whatsapp_link = f"https://api.whatsapp.com/send?text={encoded_text}%20{encoded_url}"
    return f"""<div style='text-align:center; padding-top: 10px;'><p style='font-weight: bold;'>Share your creation!</p><a href='{twitter_link}' target='_blank' style='margin: 0 5px;'>X/Twitter</a> | <a href='{facebook_link}' target='_blank' style='margin: 0 5px;'>Facebook</a> | <a href='{reddit_link}' target='_blank' style='margin: 0 5px;'>Reddit</a> | <a href='{whatsapp_link}' target='_blank' style='margin: 0 5px;'>WhatsApp</a></div>"""

def humanize_audio(audio_path):
    import librosa
    import soundfile as sf
    try:
        y, sr = librosa.load(audio_path, sr=None)
        noise = np.random.randn(len(y))
        y_noisy = y + 0.0001 * noise
        y_eq = y_noisy * (1 + 0.01 * np.sin(2 * np.pi * 1000 * np.arange(len(y)) / sr))
        sf.write(audio_path, y_eq, sr)
        return audio_path
    except Exception as e:
        catch(f"Could not humanize AI output: {e}")
        return audio_path

def value_to_keys(dictionary, target_value):
    return [key for key, value in dictionary.items() if value == target_value]

def transcribe_audio(audio_path, language):
    if MODELS["speech-recognition"] is None:
        init_pretrained_model("speech-recognition")
    lang_code = value_to_keys(language_codes, language)
    return MODELS["speech-recognition"](audio_path, generate_kwargs={"language": lang_code}, return_timestamps=True)["text"]

def generate_voice(text, reference_audio, format_choice, humanize=True):
    import soundfile as sf
    import pydub

    if not MODELS["tts"]:
        init_pretrained_model("tts")
    try:
        temp_wav_path = tmp("wav", False)
        wav = MODELS["tts"].generate(
            text=text,
            audio_prompt_path=reference_audio
        )
        sf.write(temp_wav_path, wav, 24000)
        if humanize:
            temp_wav_path = humanize_audio(temp_wav_path)
        sound = pydub.AudioSegment.from_file(temp_wav_path)
        output_stem = tmp(keep=False).replace(".data","")
        final_output_path = export_audio(sound, output_stem, format_choice)
        return final_output_path
    except Exception as e:
        catch(f"Generation failed: {e}")

def generate_music(prompt, duration_s, format_choice, humanize):
    from scipy.io.wavfile import write as write_wav

    inputs = PROCESSORS["music"](text=[prompt], padding=True, return_tensors="pt").to(device())
    max_new_tokens = int(duration_s * 50)
    audio_values = MODELS["music"].generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=max_new_tokens)
    sampling_rate = MODELS["music"].config.audio_encoder.sampling_rate
    wav_output = audio_values[0, 0].cpu().numpy()
    temp_wav_path = tmp("wav", keep=False)
    write_wav(temp_wav_path, rate=sampling_rate, data=wav_output)
    if humanize:
        temp_wav_path = humanize_audio(temp_wav_path)
    sound = pydub.AudioSegment.from_file(temp_wav_path)
    output_stem = Path(temp_wav_path).with_name(f"generated_{random_string()}")
    output_path = export_audio(sound, output_stem, format_choice)
    delete(temp_wav_path)
    return output_path

def dj_mix(files, mix_type, target_bpm, transition_sec, format_choice):
    import madmom
    import pydub

    if not files or len(files) < 2:
        catch("Please upload at least two audio files.")
        return None
    transition_ms = int(transition_sec * 1000)
    processed_tracks = []
    if target_bpm is None or target_bpm == 0:
        proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
        act = madmom.features.beats.RNNBeatProcessor()(files[0].name)
        target_bpm = np.median(60 / np.diff(proc(act)))
    for file in files:
        try:
            temp_stretched_path = None
            current_path = file.name
            if "beatmatched" in mix_type.lower():
                proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
                act = madmom.features.beats.RNNBeatProcessor()(current_path)
                original_bpm = np.median(60 / np.diff(proc(act)))
                if original_bpm > 0 and target_bpm > 0:
                    speed_factor = target_bpm / original_bpm
                    temp_stretched_path = tmp(Path(current_path).suffix)
                    stretch_audio(current_path, temp_stretched_path, speed_factor)
                    current_path = temp_stretched_path
            track_segment = pydub.AudioSegment.from_file(current_path)
            processed_tracks.append(track_segment)
            if temp_stretched_path:
                delete(temp_stretched_path)
        except Exception as e:
            print(f"Could not process track {Path(file.name).name}, skipping. Error: {e}")
            continue
    if not processed_tracks:
        catch("No tracks could be processed.")
        return None
    final_mix = processed_tracks[0]
    for i in range(1, len(processed_tracks)):
        final_mix = final_mix.append(processed_tracks[i], crossfade=transition_ms)
    output_stem = tmp("dj_mix",keep=False)
    final_output_path = export_audio(final_mix, output_stem, format_choice)
    return final_output_path

def beat_visualizer(image_path, audio_path, image_effect, animation_style, scale_intensity):
    from moviepy import ImageClip, AudioFileClip
    import librosa

    img = Image.open(image_path)
    effect_map = {"Blur": ImageFilter.BLUR, "Sharpen": ImageFilter.SHARPEN, "Contour": ImageFilter.CONTOUR, "Emboss": ImageFilter.EMBOSS}
    if image_effect in effect_map: img = img.filter(effect_map[image_effect])
    temp_img_path = tmp(".png"); img.save(temp_img_path)
    output_path = tmp(".mp4")
    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration
    y, sr = librosa.load(audio_path, sr=None)
    rms = librosa.feature.rms(y=y)[0]
    scales = 1.0 + (((rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-6)) * (scale_intensity - 1.0))
    def beat_resize_func(t):
        frame_index = min(int(t * sr / 512), len(scales) - 1)
        return scales[frame_index]
    image_clip = ImageClip(temp_img_path, duration=duration)
    if animation_style == "Zoom In": image_clip = image_clip.resize(lambda t: 1 + 0.1 * (t / duration))
    elif animation_style == "Zoom Out": image_clip = image_clip.resize(lambda t: 1.1 - 0.1 * (t / duration))
    final_clip = image_clip.resize(lambda t: image_clip.w * beat_resize_func(t) / image_clip.w).set_position(('center', 'center')).set_audio(audio_clip)
    final_clip.write_videofile(output_path, codec='libx264', fps=24, audio_codec='aac', logger=None)
    delete(temp_img_path)
    return output_path

def music_video(audio_path):
    import librosa
    import madmom
    from moviepy import VideoFileClip, AudioFileClip
    from moviepy.video.VideoClip import VideoClip

    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(audio_path)
    beat_times = proc(act)
    beats = librosa.time_to_frames(beat_times, sr=sr)
    rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-6)
    centroid_norm = (spectral_centroid - np.min(spectral_centroid)) / (np.max(spectral_centroid) - np.min(spectral_centroid) + 1e-6)
    w, h = 1280, 720
    fps = 30
    def make_frame(t):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame_idx = int(t * sr / 512)
        color_val = centroid_norm[min(frame_idx, len(centroid_norm)-1)]
        r = int(10 + color_val * 60)
        g = int(20 + color_val * 40)
        b = int(40 + color_val * 90)
        frame[:,:,:] = [r, g, b]
        radius = int(50 + rms_norm[min(frame_idx, len(rms_norm)-1)] * 200)
        center_x, center_y = w // 2, h // 2
        for beat_frame in beats:
            if abs(frame_idx - beat_frame) < 2:
                radius = int(radius * 1.5)
                break
        rr, cc = np.ogrid[:h, :w]
        circle_mask = (rr - center_y)**2 + (cc - center_x)**2 <= radius**2
        frame[circle_mask] = [int(200 + color_val * 55), int(150 - color_val * 50), int(100 + color_val * 50)]
        return frame
    output_path = tmp(".mp4")
    animation = VideoClip(make_frame, duration=duration)
    audio_clip = AudioFileClip(audio_path)
    final_clip = animation.set_audio(audio_clip)
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=fps)
    return output_path

def lyric_video(audio_path, background_path, lyrics_text, text_position):
    from moviepy import ImageClip, AudioFileClip, VideoFileClip, ColorClip, TextClip, CompositeVideoClip

    font_path = "./Alef-Bold.ttf"
    if not os.path.exists(font_path):
        print("Font not found, downloading...")
        google_drive_download("1C48KkYWQDYu7ypbNtSXAUJ6kuzoZ42sI", font_path)

    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration
    if background_path:
        bg_clip_class = ImageClip if background_path.lower().endswith(('.png', '.jpg', '.jpeg')) else VideoFileClip
        background_clip = bg_clip_class(background_path, duration=duration)
    else:
        background_clip = ColorClip(size=(1280, 720), color=(0,0,0), duration=duration)
    background_clip = background_clip.resize(width=1280)
    lines = [line for line in lyrics_text.strip().split('\n') if line.strip()]
    if not lines:
        catch("Lyrics text is empty.")
        return None
    line_duration = duration / len(lines)
    lyric_clips = [TextClip(line, fontsize=70, color='white', font=font_path, stroke_color='black', stroke_width=2).set_position(text_position).set_start(i * line_duration).set_duration(line_duration) for i, line in enumerate(lines)]
    final_clip = CompositeVideoClip([background_clip] + lyric_clips, size=background_clip.size).set_audio(audio_clip)
    output_path = tmp(".mp4")
    final_clip.write_videofile(output_path, codec='libx264', fps=24, audio_codec='aac', logger=None)
    return output_path

def stretch_audio(input_path, output_path, speed_factor, crispness=6):
    if not os.path.exists(input_path):
        return None
    command = ["rubberband", "--tempo", str(speed_factor), "--crispness", str(crispness), "-q", input_path, output_path]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return output_path
    except Exception as e:
        catch(f"Error during audio stretching with rubberband: {e}")
        return None

def get_audio_feedback(audio_path):
    import librosa
    from scipy.stats import pearsonr

    if not audio_path:
        catch("Please upload an audio file for feedback.")
        return None
    try:
        y_stereo, sr = librosa.load(audio_path, sr=None, mono=False)
        y_mono = librosa.to_mono(y_stereo) if y_stereo.ndim > 1 else y_stereo
        rms = librosa.feature.rms(y=y_mono)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y_mono, sr=sr)
        stft = librosa.stft(y_mono)
        freqs = librosa.fft_frequencies(sr=sr)
        bass_energy = np.mean(np.abs(stft[ (freqs >= 20) & (freqs < 250) ]))
        high_energy = np.mean(np.abs(stft[ (freqs >= 5000) & (freqs < 20000) ]))
        peak_amp = np.max(np.abs(y_mono))
        mean_rms = np.mean(rms)
        crest_factor = 20 * np.log10(peak_amp / mean_rms) if mean_rms > 0 else 0
        stereo_width = 0
        if y_stereo.ndim > 1 and y_stereo.shape[0] == 2:
            corr, _ = pearsonr(y_stereo[0], y_stereo[1])
            stereo_width = (1 - corr) * 100
        feedback = "### AI Track Feedback\n\n"
        feedback += "#### Technical Analysis\n"
        feedback += f"- **Loudness & Dynamics:** The track has a crest factor of **{crest_factor:.2f} dB**. "
        if crest_factor > 14:
            feedback += "This suggests the track is very dynamic and punchy.\n"
        elif crest_factor > 8:
            feedback += "This is a good balance between punch and loudness, typical for many genres.\n"
        else:
            feedback += "This suggests the track is heavily compressed or limited, prioritizing loudness over dynamic range.\n"
        feedback += f"- **Stereo Image:** The stereo width is estimated at **{stereo_width:.1f}%**. "
        if stereo_width > 60:
            feedback += "The mix feels wide and immersive.\n"
        elif stereo_width > 20:
            feedback += "The mix has a balanced stereo field.\n"
        else:
            feedback += "The mix is narrow or mostly mono.\n"
        feedback += f"- **Frequency Balance:** Bass energy is at **{bass_energy:.2f}** and high-frequency energy is at **{high_energy:.2f}**. "
        if bass_energy > high_energy * 2:
            feedback += "The track is bass-heavy.\n"
        elif high_energy > bass_energy * 2:
            feedback += "The track is bright or treble-heavy.\n"
        else:
            feedback += "The track has a relatively balanced frequency spectrum.\n"
        feedback += "\n#### Advice\n"
        if crest_factor < 8:
            feedback += "- **Compression:** The track might be over-compressed. Consider reducing the amount of compression to bring back some life and punch to the transients.\n"
        if stereo_width < 20 and y_stereo.ndim > 1:
            feedback += "- **Stereo Width:** To make the mix sound bigger, try using stereo widening tools or panning instruments differently to create more space.\n"
        if bass_energy > high_energy * 2.5:
            feedback += "- **Bass Management:** The low-end might be overpowering. Ensure it's not masking other instruments. A high-pass filter on non-bass elements can clean up muddiness.\n"
        if high_energy > bass_energy * 2.5:
            feedback += "- **Tame the Highs:** The track is very bright, which can be fatiguing. Check for harshness in cymbals or vocals, and consider using a de-esser or a gentle high-shelf cut.\n"
        if mean_rms < 0.05:
            feedback += "- **Mastering:** The overall volume is low. The track would benefit from mastering to increase its loudness and competitiveness with commercial tracks.\n"
        else:
            feedback += "- **General Mix:** The track has a solid technical foundation. Focus on creative choices, arrangement, and ensuring all elements have their own space in the mix.\n"
        return feedback
    except Exception as e:
        raise catch(f"Analysis failed: {e}")
        return None

def analyze_audio_features(audio_path):
    import librosa
    import madmom

    try:
        proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
        act = madmom.features.beats.RNNBeatProcessor()(audio_path)
        bpm = np.median(60 / np.diff(proc(act)))
        y, sr = librosa.load(audio_path)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_map = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = key_map[np.argmax(np.sum(chroma, axis=1))]
        return f"{key}, {bpm:.2f} BPM"
    except Exception as e:
        catch(f"Analysis failed: {e}")
        return None

def change_audio_speed(audio_path, speed_factor, preserve_pitch, format_choice):
    import pydub

    sound_out = None
    if preserve_pitch:
        audio_path_out = tmp(Path(audio_path).suffix)
        if stretch_audio(audio_path, audio_path_out, speed_factor):
            sound_out = pydub.AudioSegment.from_file(audio_path_out)
            delete(audio_path_out)
        else:
            catch("Failed to stretch audio while preserving pitch.")
            return None
    else:
        sound = pydub.AudioSegment.from_file(audio_path)
        new_frame_rate = int(sound.frame_rate * speed_factor)
        sound_out = sound._spawn(sound.raw_data, overrides={"frame_rate": new_frame_rate}).set_frame_rate(sound.frame_rate)
    if sound_out:
        output_stem = str(Path(audio_path).with_name(f"{Path(audio_path).stem}_speed_{speed_factor}x"))
        return export_audio(sound_out, output_stem, format_choice)
    else:
        catch("Could not process audio speed change.")
        return None

def separate_stems(audio_path, separation_type, format_choice):
    import pydub
    output_dir = tmp(dir=True)
    run(f'"{sys.executable}" -m demucs.separate -n htdemucs_ft --two-stems=vocals -o "{output_dir}" "{audio_path}"')
    separated_dir = Path(output_dir) / "htdemucs_ft" / Path(audio_path).stem
    vocals_path = separated_dir / "vocals.wav"
    accompaniment_path = separated_dir / "no_vocals.wav"
    if not vocals_path.exists() or not accompaniment_path.exists():
        delete(output_dir)
        catch("Stem separation failed.")
        return None
    chosen_stem_path, suffix = (vocals_path, "_acapella") if "acapella" in separation_type.lower() else (accompaniment_path, "_karaoke")
    sound = pydub.AudioSegment.from_file(chosen_stem_path)
    output_stem = str(Path(audio_path).with_name(Path(audio_path).stem + suffix))
    final_output_path = export_audio(sound, output_stem, format_choice)
    delete(output_dir)
    return final_output_path

def pitch_shift_vocals(audio_path, pitch_shift, format_choice):
    import librosa
    import pydub
    import soundfile as sf

    separation_dir = tmp(dir=True)
    run(f'"{sys.executable}" -m demucs.separate -n htdemucs_ft --two-stems=vocals -o "{separation_dir}" "{audio_path}"')
    separated_dir = Path(separation_dir) / "htdemucs_ft" / Path(audio_path).stem
    vocals_path = separated_dir / "vocals.wav"
    instrumental_path = separated_dir / "no_vocals.wav"
    if not vocals_path.exists() or not instrumental_path.exists():
        delete(separation_dir)
        catch("Vocal separation failed.")
        return None
    y_vocals, sr = librosa.load(str(vocals_path), sr=None)
    y_shifted = librosa.effects.pitch_shift(y=y_vocals, sr=sr, n_steps=float(pitch_shift))
    shifted_vocals_path = tmp("shifted_vocals.wav", keep=False)
    sf.write(shifted_vocals_path, y_shifted, sr)
    instrumental = pydub.AudioSegment.from_file(instrumental_path)
    shifted_vocals = pydub.AudioSegment.from_file(shifted_vocals_path)
    combined = instrumental.overlay(shifted_vocals)
    output_stem = str(Path(audio_path).with_name(f"{Path(audio_path).stem}_vocal_pitch_shifted"))
    final_output_path = export_audio(combined, output_stem, format_choice)
    delete(separation_dir)
    delete(shifted_vocals_path)
    return final_output_path

def create_spectrum_visualization(audio_path):
    import librosa
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    try:
        y, sr = librosa.load(audio_path)

        n_fft = 128
        
        start_sample = (len(y) - n_fft) // 2
        y_sample = y[start_sample : start_sample + n_fft]

        if len(y_sample) < n_fft:
            y_sample = np.pad(y_sample, (0, n_fft - len(y_sample)))

        window = np.hanning(len(y_sample))
        y_windowed = y_sample * window

        fft_result = np.fft.fft(y_windowed)
        freqs = np.fft.fftfreq(len(fft_result), 1/sr)

        mask = freqs >= 0
        freqs = freqs[mask]
        magnitude = np.abs(fft_result[mask])

        magnitude_db = 20 * np.log10(magnitude + 1e-9)

        magnitude_db -= np.max(magnitude_db)

        fig, ax = plt.subplots(figsize=(8, 5), facecolor='#f0f0f0')
        ax.set_facecolor('white')

        ax.fill_between(freqs, magnitude_db, y2=-84, color='#7c3aed', alpha=0.8, zorder=2)
        ax.plot(freqs, magnitude_db, color='#4c2a8c', linewidth=1, zorder=3)

        ax.set_xscale('log')
        ax.set_xlim(20, 22000)
        ax.set_ylim(-84, 0)

        xticks = [400, 1000, 2000, 4000, 7000, 20000]
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

        ax.set_yticks(np.arange(-84, 1, 12))
        ax.grid(True, which="both", ls="--", color='gray', alpha=0.6, zorder=1)

        ax.set_title('Frequency Analysis', color='black')
        ax.set_xlabel('Frequency (Hz)', color='black')
        ax.set_ylabel('Amplitude (dB)', color='black')
        ax.tick_params(colors='black', which='both')

        audible_mask = freqs > 20
        if np.any(audible_mask):
            peak_idx = np.argmax(magnitude_db[audible_mask])
            peak_freq = freqs[audible_mask][peak_idx]
            peak_db = magnitude_db[audible_mask][peak_idx]

            peak_text = f'Peak: {peak_freq:.0f} Hz at {peak_db:.1f} dB'
            ax.text(0.98, 0.95, peak_text, transform=ax.transAxes, color='black',
                      ha='right', va='top')

        fig.tight_layout()
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name
        fig.savefig(temp_path, facecolor=fig.get_facecolor())
        plt.close(fig)

        return temp_path

    except Exception as e:
        print(f"Error creating spectrum: {e}")
        return None

def stem_mixer(files, format_choice):
    from scipy.io.wavfile import write as write_wav
    import librosa
    import madmom
    import  soundfile as sf
    import pydub

    if not files or len(files) < 2:
        catch("Please upload at least two stem files.")
        return None
    processed_stems = []
    target_sr = None
    target_bpm = None
    for i, file in enumerate(files):
        y, sr = librosa.load(file.name, sr=None)
        if target_sr is None:
            target_sr = sr
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
        act = madmom.features.beats.RNNBeatProcessor()(file.name)
        tempo = np.median(60 / np.diff(proc(act)))
        if i == 0:
            target_bpm = tempo
        if tempo != target_bpm:
            speed_factor = target_bpm / tempo
            temp_stretched_path = tmp(".wav")
            temp_original_path = tmp(".wav")
            sf.write(temp_original_path, y, target_sr)
            stretch_audio(temp_original_path, temp_stretched_path, speed_factor)
            y, _ = librosa.load(temp_stretched_path, sr=target_sr)
            delete(temp_original_path)
            delete(temp_stretched_path)
        processed_stems.append(y)
    max_length = max(len(y) for y in processed_stems)
    mixed_y = np.zeros(max_length)
    for y in processed_stems:
        mixed_y[:len(y)] += y
    mixed_y /= len(processed_stems)
    temp_wav_path = tmp(".wav")
    write_wav(temp_wav_path, target_sr, (mixed_y * 32767).astype(np.int16))
    sound = pydub.AudioSegment.from_file(temp_wav_path)
    output_stem = Path(temp_wav_path).with_name(f"stem_mix_{random_string()}")
    output_path = export_audio(sound, output_stem, format_choice)
    delete(temp_wav_path)
    return output_path

def identify_instruments(audio_path):
    if MODELS["audio-classification"] is None:
        catch("Audio identification model is not available.")
        return None
    predictions = MODELS["audio-classification"](audio_path, top_k=10)
    instrument_list = [
        "guitar", "piano", "violin", "drum", "bass", "saxophone", "trumpet", "flute",
        "cello", "clarinet", "synthesizer", "organ", "accordion", "banjo", "harp", "voice", "speech"
    ]
    detected_instruments = "### Detected Instruments\n\n"
    found = False
    for p in predictions:
        label = p['label'].lower()
        if any(instrument in label for instrument in instrument_list):
            detected_instruments += f"- **{p['label'].title()}** (Score: {p['score']:.2f})\n"
            found = True
    if not found:
        detected_instruments += "Could not identify specific instruments with high confidence. Top sound events:\n"
        for p in predictions[:3]:
            detected_instruments += f"- {p['label'].title()} (Score: {p['score']:.2f})\n"
    return detected_instruments

def extend_audio(audio_path, extend_duration_s, format_choice, humanize = True):
    import librosa
    import soundfile as sf
    import pydub

    if MODELS["music"] is None or PROCESSORS["music"] is None:
        catch("MusicGen model is not available for audio extension.")
        return None
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    prompt_duration_s = min(15.0, len(y) / sr)
    prompt_wav = y[-int(prompt_duration_s * sr):]
    inputs = PROCESSORS["music"](
        audio=prompt_wav,
        sampling_rate=sr,
        return_tensors="pt"
    ).to(device())
    total_duration_s = prompt_duration_s + extend_duration_s
    max_new_tokens = int(total_duration_s * 50)
    generated_audio_values = MODELS["music"].generate(
        **inputs,
        do_sample=True,
        guidance_scale=3,
        max_new_tokens=max_new_tokens
    )
    generated_wav = generated_audio_values[0, 0].cpu().numpy()
    extension_start_sample = int(prompt_duration_s * MODELS["music"].config.audio_encoder.sampling_rate)
    extension_wav = generated_wav[extension_start_sample:]
    temp_extension_path = tmp(".wav")
    sf.write(temp_extension_path, extension_wav, MODELS["music"].config.audio_encoder.sampling_rate)
    if humanize:
        temp_extension_path = humanize_audio(temp_extension_path)
    original_sound = pydub.AudioSegment.from_file(audio_path)
    extension_sound = pydub.AudioSegment.from_file(temp_extension_path)
    if original_sound.channels != extension_sound.channels:
        extension_sound = extension_sound.set_channels(original_sound.channels)
    final_sound = original_sound + extension_sound
    output_stem = str(Path(audio_path).with_name(f"{Path(audio_path).stem}_extended"))
    final_output_path = export_audio(final_sound, output_stem, format_choice)
    delete(temp_extension_path)
    return final_output_path

def audio_to_midi(audio_path):

    import madmom
    from basic_pitch.inference import predict, Model
    from basic_pitch import ICASSP_2022_MODEL_PATH

    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(audio_path)
    bpm = np.median(60 / np.diff(proc(act)))

    model_output, midi_data, note_events = predict(
        audio_path,
        midi_tempo=bpm,
        onset_threshold=0.95, # Segmentation
        frame_threshold=0.25, # Confidence
        minimum_note_length=80, # Length
        minimum_frequency=60,
        maximum_frequency=4200
    )
    
    name = random_string() + ".mid"
    midi_data.write(f'./{ name }')
    return name

def midi_to_audio(midi_path, format_choice):
    from midi2audio import FluidSynth
    import pydub

    soundfont_paths = [
        os.path.join(os.path.expanduser("~"), "app_dependencies", "soundfonts", "FluidR3_GM.sf2"),
        "/usr/share/sounds/sf2/FluidR3_GM.sf2",
        "C:/Windows/System32/drivers/gm.dls"
    ]
    soundfont_file = None
    for path in soundfont_paths:
        if os.path.exists(path):
            soundfont_file = path
            break
    if soundfont_file is None:
        catch("SoundFont file not found. MIDI to Audio conversion cannot proceed. Please re-run the dependency installer.")
        return None
    fs = FluidSynth(sound_font=soundfont_file)
    temp_wav_path = tmp(".wav")
    fs.midi_to_audio(midi_path, temp_wav_path)
    sound = pydub.AudioSegment.from_file(temp_wav_path)
    output_stem = str(Path(midi_path).with_name(f"{Path(midi_path).stem}_render"))
    final_output_path = export_audio(sound, output_stem, format_choice)
    delete(temp_wav_path)
    return final_output_path

def autotune_vocals(audio_path, strength, format_choice):
    import librosa
    import madmom
    import soundfile as sf
    import pydub

    separation_dir = tmp(dir=True)
    try:
        run(f'"{sys.executable}" -m demucs.separate -n htdemucs_ft --two-stems=vocals -o "{separation_dir}" "{audio_path}"')
        separated_dir = Path(separation_dir) / "htdemucs_ft" / Path(audio_path).stem
        vocals_path = separated_dir / "vocals.wav"
        instrumental_path = separated_dir / "no_vocals.wav"
        if not vocals_path.exists() or not instrumental_path.exists():
            catch("Vocal separation failed.")
            return None
        y_original, sr = librosa.load(str(vocals_path), sr=None, mono=True)
        y = np.copy(y_original)
        print("Starting vocal rhythm correction...")
        try:
            proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
            act = madmom.features.beats.RNNBeatProcessor()(str(instrumental_path))
            beat_times = proc(act)
            vocal_intervals = librosa.effects.split(y, top_db=40, frame_length=2048, hop_length=512)
            if len(vocal_intervals) > 0 and len(beat_times) > 0:
                y_timed = np.zeros_like(y)
                last_end_sample = 0
                for start_frame, end_frame in vocal_intervals:
                    start_sample = librosa.frames_to_samples(start_frame, hop_length=512)
                    end_sample = librosa.frames_to_samples(end_frame, hop_length=512)
                    segment = y[start_sample:end_sample]
                    if len(segment) == 0:
                        continue
                    start_time = librosa.samples_to_time(start_sample, sr=sr)
                    quantized_start_time_idx = np.argmin(np.abs(beat_times - start_time))
                    quantized_start_time = beat_times[quantized_start_time_idx]
                    quantized_start_sample = librosa.time_to_samples(quantized_start_time, sr=sr)
                    if quantized_start_sample < last_end_sample:
                        next_beat_candidates = beat_times[beat_times > librosa.samples_to_time(last_end_sample, sr=sr)]
                        if len(next_beat_candidates) > 0:
                            quantized_start_sample = librosa.time_to_samples(next_beat_candidates[0], sr=sr)
                        else:
                            quantized_start_sample = last_end_sample
                    start_pos = quantized_start_sample
                    end_pos = start_pos + len(segment)
                    segment_to_place = segment
                    if end_pos > len(y_timed):
                        segment_to_place = segment[:len(y_timed) - start_pos]
                    if len(segment_to_place) > 0:
                        y_timed[start_pos : start_pos + len(segment_to_place)] += segment_to_place
                        last_end_sample = start_pos + len(segment_to_place)
                max_amp = np.max(np.abs(y_timed))
                if max_amp > 1.0:
                    y_timed /= max_amp
                if np.max(np.abs(y_timed)) < 0.01:
                    print("Rhythm correction resulted in near-silence. Reverting to original vocal timing.")
                    y = y_original
                else:
                    y = y_timed
                    print("Vocal rhythm correction applied successfully.")
            else:
                print("Could not detect beats or vocal segments, skipping rhythm correction.")
        except Exception as e:
            catch(f"Could not apply rhythm correction, proceeding with pitch correction only. Error: {e}")
        n_fft = 2048
        hop_length = 512
        stft_vocals = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        magnitudes = np.abs(stft_vocals)
        f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr, frame_length=n_fft, hop_length=hop_length)
        f0 = np.nan_to_num(f0)
        target_f0 = np.copy(f0)
        for i in range(len(f0)):
            if voiced_flag[i]:
                current_f0 = f0[i]
                if current_f0 > 0:
                    target_midi = round(librosa.hz_to_midi(current_f0))
                    ideal_f0 = librosa.midi_to_hz(target_midi)
                    target_f0[i] = current_f0 + (ideal_f0 - current_f0) * strength
        phase = np.angle(stft_vocals)
        new_phase = np.zeros_like(phase)
        new_phase[:, 0] = phase[:, 0]
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        expected_phase_advance = 2 * np.pi * freqs * hop_length / sr
        for t in range(1, stft_vocals.shape[1]):
            dphase = phase[:, t] - phase[:, t-1] - expected_phase_advance
            dphase = dphase - 2 * np.pi * np.round(dphase / (2 * np.pi))
            true_freq = expected_phase_advance + dphase
            ratio = 1.0
            if t < len(f0) and f0[t] > 0 and target_f0[t] > 0:
                ratio = target_f0[t] / f0[t]
            shifted_phase_advance = true_freq * ratio
            new_phase[:, t] = new_phase[:, t-1] + shifted_phase_advance
        stft_tuned = magnitudes * np.exp(1j * new_phase)
        y_tuned = librosa.istft(stft_tuned, hop_length=hop_length, length=len(y))
        temp_tuned_vocals_path = tmp("tuned_vocals.wav")
        sf.write(temp_tuned_vocals_path, y_tuned, sr)
        instrumental = pydub.AudioSegment.from_file(instrumental_path)
        tuned_vocals = pydub.AudioSegment.from_file(temp_tuned_vocals_path)
        if instrumental.channels == 2 and tuned_vocals.channels == 1:
            tuned_vocals = tuned_vocals.set_channels(2)
        combined = instrumental.overlay(tuned_vocals)
        output_stem = str(Path(audio_path).with_name(f"{Path(audio_path).stem}_autotuned"))
        final_output_path = export_audio(combined, output_stem, format_choice)
        delete(temp_tuned_vocals_path)
        return final_output_path
    finally:
        delete(separation_dir)