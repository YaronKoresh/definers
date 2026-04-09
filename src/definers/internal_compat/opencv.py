from __future__ import annotations

import colorsys
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

COLOR_BGR2GRAY = 6
COLOR_GRAY2BGR = 8
COLOR_RGB2BGR = 4
COLOR_BGR2RGB = 5
COLOR_HSV2BGR = 54
CV_8U = np.uint8
FONT_HERSHEY_SIMPLEX = 0
IMREAD_UNCHANGED = -1
LINE_AA = 16
NORM_MINMAX = 32


def _to_uint8(array):
    return np.clip(np.asarray(array), 0, 255).astype(np.uint8)


def _prepare_image_for_save(image):
    array = _to_uint8(image)
    if array.ndim == 2:
        return Image.fromarray(array, mode="L")
    if array.ndim == 3 and array.shape[2] == 3:
        return Image.fromarray(array[:, :, ::-1], mode="RGB")
    if array.ndim == 3 and array.shape[2] == 4:
        return Image.fromarray(array[:, :, [2, 1, 0, 3]], mode="RGBA")
    raise ValueError("Unsupported image shape")


def _prepare_image_for_load(image):
    array = np.array(image)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] == 4:
        return array[:, :, [2, 1, 0, 3]]
    if array.ndim == 3 and array.shape[2] == 3:
        return array[:, :, ::-1]
    return array


def imwrite(filename, image):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    _prepare_image_for_save(image).save(path)
    return True


def imread(filename, flags=None):
    path = Path(filename)
    if not path.exists():
        return None
    try:
        image = Image.open(path)
    except Exception:
        return None
    if flags == IMREAD_UNCHANGED:
        return _prepare_image_for_load(image)
    if image.mode not in {"L", "RGB", "RGBA"}:
        image = image.convert("RGB")
    return _prepare_image_for_load(image)


def cvtColor(src, code):
    array = np.asarray(src)
    if code in {COLOR_BGR2RGB, COLOR_RGB2BGR}:
        return np.array(array[..., ::-1], copy=True)
    if code == COLOR_BGR2GRAY:
        if array.ndim == 2:
            return np.array(array, copy=True)
        bgr = np.asarray(array, dtype=np.float32)
        gray = 0.114 * bgr[..., 0] + 0.587 * bgr[..., 1] + 0.299 * bgr[..., 2]
        return gray.astype(np.uint8)
    if code == COLOR_GRAY2BGR:
        if array.ndim == 3:
            return np.array(array, copy=True)
        return np.repeat(np.asarray(array)[..., None], 3, axis=2).astype(
            np.uint8
        )
    if code == COLOR_HSV2BGR:
        hsv = np.asarray(array, dtype=np.float32)
        if hsv.ndim == 1:
            hsv = hsv.reshape(1, 1, -1)
        output = np.empty_like(hsv, dtype=np.uint8)
        for index in np.ndindex(hsv.shape[:-1]):
            h, s, v = hsv[index]
            r, g, b = colorsys.hsv_to_rgb(h / 179.0, s / 255.0, v / 255.0)
            output[index] = [int(b * 255), int(g * 255), int(r * 255)]
        return output
    raise ValueError(f"Unsupported conversion code: {code}")


def calcHist(images, channels, mask, hist_size, ranges):
    _ = mask
    image = np.asarray(images[0])
    channel = int(channels[0])
    bins = int(hist_size[0])
    lower = float(ranges[0])
    upper = float(ranges[1])
    values = image[..., channel].reshape(-1)
    histogram, _ = np.histogram(values, bins=bins, range=(lower, upper))
    return histogram.astype(np.float32).reshape(-1, 1)


def Canny(image, threshold1, threshold2):
    gray = np.asarray(image, dtype=np.float32)
    if gray.ndim == 3:
        gray = cvtColor(gray, COLOR_BGR2GRAY).astype(np.float32)
    grad_y = np.zeros_like(gray)
    grad_x = np.zeros_like(gray)
    grad_y[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    grad_x[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    magnitude = np.hypot(grad_x, grad_y)
    threshold = max(float(threshold1), float(threshold2) * 0.5)
    return (magnitude >= threshold).astype(np.uint8) * 255


def normalize(src, dst, alpha, beta, norm_type, dtype=None):
    _ = norm_type
    array = np.asarray(src, dtype=np.float32)
    min_value = float(np.min(array)) if array.size else 0.0
    max_value = float(np.max(array)) if array.size else 0.0
    if max_value == min_value:
        scaled = np.full_like(array, float(alpha), dtype=np.float32)
    else:
        scaled = (array - min_value) / (max_value - min_value)
        scaled = scaled * (float(beta) - float(alpha)) + float(alpha)
    if dtype is CV_8U:
        return _to_uint8(scaled)
    return scaled.astype(array.dtype, copy=False)


def addWeighted(src1, alpha, src2, beta, gamma, dst=None):
    result = np.asarray(src1, dtype=np.float32) * float(alpha)
    result += np.asarray(src2, dtype=np.float32) * float(beta)
    result += float(gamma)
    output = _to_uint8(result)
    if dst is not None:
        dst[...] = output
        return dst
    return output


def VideoWriter_fourcc(*codes):
    return "".join(str(code) for code in codes)


class VideoWriter:
    def __init__(self, filename, fourcc, fps, frame_size):
        self.filename = filename
        self.fourcc = fourcc
        self.fps = fps
        self.frame_size = tuple(frame_size)
        self.frames = []

    def write(self, frame):
        array = np.asarray(frame, dtype=np.uint8)
        self.frames.append(np.array(array, copy=True))

    def release(self):
        path = Path(self.filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fps": self.fps,
            "frame_size": self.frame_size,
            "frames": self.frames,
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle)


class VideoCapture:
    def __init__(self, filename):
        self.filename = filename
        self.index = 0
        self.payload = None
        try:
            with Path(filename).open("rb") as handle:
                self.payload = pickle.load(handle)
        except Exception:
            self.payload = None

    def isOpened(self):
        return isinstance(self.payload, dict)

    def read(self):
        if not self.isOpened():
            return False, None
        frames = self.payload.get("frames", [])
        if self.index >= len(frames):
            return False, None
        frame = np.array(frames[self.index], copy=True)
        self.index += 1
        return True, frame

    def release(self):
        self.payload = None


def resize(image, dsize):
    width, height = dsize
    array = np.asarray(image)
    pil_image = _prepare_image_for_save(array)
    resized = pil_image.resize((int(width), int(height)))
    return _prepare_image_for_load(resized)


def circle(image, center, radius, color, thickness=1, lineType=None):
    _ = (center, radius, color, thickness, lineType)
    return image


def line(image, pt1, pt2, color, thickness=1, lineType=None):
    _ = (pt1, pt2, color, thickness, lineType)
    return image


def rectangle(image, pt1, pt2, color, thickness=1):
    _ = (pt1, pt2, color, thickness)
    return image


def polylines(image, pts, isClosed, color, thickness=1, lineType=None):
    _ = (pts, isClosed, color, thickness, lineType)
    return image


def putText(
    image, text, org, fontFace, fontScale, color, thickness=1, lineType=None
):
    _ = (text, org, fontFace, fontScale, color, thickness, lineType)
    return image


def getTextSize(text, fontFace, fontScale, thickness):
    _ = (text, fontFace)
    width = int(max(len(str(text)), 1) * max(float(fontScale), 1.0) * 10)
    height = int(max(float(fontScale), 1.0) * 20)
    return (width, height), int(thickness)
