from __future__ import annotations

import types
from pathlib import Path

import numpy as np


class FakeGradioError(Exception):
    pass


def build_fake_gradio_module():
    module = types.ModuleType("gradio")
    module.Error = FakeGradioError
    return module


class FakeDataset:
    def __init__(self, data):
        self._data = {key: list(values) for key, values in dict(data).items()}
        self.column_names = tuple(self._data)

    @classmethod
    def from_dict(cls, data):
        return cls(data)

    def __len__(self):
        if not self._data:
            return 0
        return len(next(iter(self._data.values())))

    def __getitem__(self, key):
        return self._data[key]

    def remove_columns(self, columns):
        columns_to_remove = set(columns)
        return FakeDataset(
            {
                key: values
                for key, values in self._data.items()
                if key not in columns_to_remove
            }
        )


def build_fake_datasets_module():
    module = types.ModuleType("datasets")
    module.Dataset = FakeDataset
    return module


def build_fake_soundfile_module():
    module = types.ModuleType("soundfile")
    module._records = {}

    def write(path, data, sample_rate):
        normalized_path = str(path)
        Path(normalized_path).parent.mkdir(parents=True, exist_ok=True)
        Path(normalized_path).write_bytes(b"soundfile")
        module._records[normalized_path] = (
            np.asarray(data, dtype=np.float32),
            int(sample_rate),
        )

    def read(path):
        return module._records[str(path)]

    module.write = write
    module.read = read
    return module


def build_fake_librosa_module(audio_store=None):
    module = types.ModuleType("librosa")
    stored_audio = {} if audio_store is None else dict(audio_store)

    def load(path, sr=None, mono=False):
        audio_signal, sample_rate = stored_audio[str(path)]
        normalized_signal = np.asarray(audio_signal, dtype=np.float32)
        if mono and normalized_signal.ndim > 1:
            normalized_signal = np.mean(normalized_signal, axis=0)
        return normalized_signal, int(sample_rate if sr is None else sr)

    module.load = load
    return module


def build_fake_cv2_module():
    module = types.ModuleType("cv2")
    image_store: dict[str, np.ndarray | None] = {}
    video_store: dict[str, dict[str, object]] = {}

    module.COLOR_BGR2GRAY = 1

    def imwrite(path, image):
        normalized_path = str(path)
        Path(normalized_path).parent.mkdir(parents=True, exist_ok=True)
        Path(normalized_path).write_bytes(b"cv2")
        image_store[normalized_path] = np.asarray(image, dtype=np.uint8)
        return True

    def imread(path):
        return image_store.get(str(path))

    def cvtColor(image, code):
        if code != module.COLOR_BGR2GRAY:
            raise ValueError("Unsupported color conversion")
        return np.mean(np.asarray(image), axis=2).astype(np.uint8)

    def calcHist(images, channels, mask, hist_size, ranges):
        channel_index = int(channels[0])
        values = np.asarray(images[0])[:, :, channel_index].ravel()
        histogram, _ = np.histogram(
            values,
            bins=int(hist_size[0]),
            range=(float(ranges[0]), float(ranges[1])),
        )
        return histogram.astype(np.float32).reshape(-1, 1)

    def Canny(gray_image, threshold1, threshold2):
        image_values = np.asarray(gray_image, dtype=np.float32)
        return (image_values > float(image_values.mean())).astype(
            np.uint8
        ) * 255

    def VideoWriter_fourcc(*args):
        return 0

    class VideoWriter:
        def __init__(self, path, fourcc, fps, size):
            self.path = str(path)
            self.fps = fps
            self.frames = []

        def write(self, frame):
            self.frames.append(np.asarray(frame, dtype=np.uint8))

        def release(self):
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.path).write_bytes(b"video")
            video_store[self.path] = {
                "frames": list(self.frames),
                "fps": self.fps,
            }

    class VideoCapture:
        def __init__(self, path):
            self.path = str(path)
            self.index = 0

        def isOpened(self):
            return self.path in video_store

        def read(self):
            stored = video_store.get(self.path)
            if stored is None:
                return False, None
            frames = stored.get("frames", [])
            if self.index >= len(frames):
                return False, None
            frame = np.asarray(frames[self.index], dtype=np.uint8)
            self.index += 1
            return True, frame

        def release(self):
            return None

    module.imwrite = imwrite
    module.imread = imread
    module.cvtColor = cvtColor
    module.calcHist = calcHist
    module.Canny = Canny
    module.VideoWriter_fourcc = VideoWriter_fourcc
    module.VideoWriter = VideoWriter
    module.VideoCapture = VideoCapture
    module._image_store = image_store
    module._video_store = video_store
    return module
