import math
import os
from pathlib import Path

from definers.runtime_numpy import get_array_module, get_numpy_module

np = get_array_module()
_np = get_numpy_module()


def _fallback_local_binary_pattern(gray_image, radius: int):
    image = _np.asarray(gray_image, dtype=_np.float32)
    padded = _np.pad(image, radius, mode="edge")
    center = padded[radius:-radius, radius:-radius]
    offsets = (
        (-radius, -radius),
        (-radius, 0),
        (-radius, radius),
        (0, radius),
        (radius, radius),
        (radius, 0),
        (radius, -radius),
        (0, -radius),
    )
    pattern = _np.zeros_like(center, dtype=_np.float32)
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = padded[
            radius + dy : radius + dy + center.shape[0],
            radius + dx : radius + dx + center.shape[1],
        ]
        pattern += (neighbor >= center).astype(_np.float32) * (1 << bit)
    return pattern


def _log_media_exception(message, exception):
    import definers.logger as logger_module

    logger_exception = getattr(logger_module, "exception", None)
    if callable(logger_exception):
        logger_exception(message)
        return
    from definers.logger import init_logger

    init_logger().exception(message)


def _extract_visual_features(image, gray_image):
    import cv2

    histograms = [
        cv2.calcHist([image], [channel], None, [256], [0, 256]).flatten()
        for channel in range(3)
    ]
    color_histogram = _np.concatenate(histograms).astype(_np.float32)
    radius = 1
    n_points = 8 * radius
    try:
        import skimage.feature as skf

        local_binary_pattern = skf.local_binary_pattern(
            gray_image,
            n_points,
            radius,
            method="uniform",
        )
    except Exception:
        local_binary_pattern = _fallback_local_binary_pattern(
            gray_image,
            radius,
        )
    local_binary_pattern = local_binary_pattern.flatten().astype(_np.float32)
    edges = cv2.Canny(gray_image, 100, 200).flatten().astype(_np.float32)
    return _np.concatenate((color_histogram, local_binary_pattern, edges))


def _reconstruct_visual_frame(predicted_features, frame_shape):
    import cv2

    (height, width, channels) = frame_shape
    hist_size = 256 * 3
    spatial_size = height * width
    expected_size = hist_size + spatial_size * 2
    features = _np.asarray(predicted_features, dtype=_np.float32).reshape(-1)
    if features.size != expected_size:
        raise ValueError(
            f"Expected {expected_size} features for shape {frame_shape}, got {features.size}."
        )
    color_histogram = features[:hist_size].reshape(3, 256)
    local_binary_pattern = features[
        hist_size : hist_size + spatial_size
    ].reshape(height, width)
    edge_features = features[hist_size + spatial_size :].reshape(height, width)
    reconstructed_frame = np.zeros(frame_shape, dtype=np.uint8)
    for channel_index in range(channels):
        channel_histogram = color_histogram[min(channel_index, 2)]
        max_value = (
            float(_np.max(channel_histogram)) if channel_histogram.size else 0.0
        )
        if max_value <= 0.0:
            continue
        channel_value = np.uint8(channel_histogram / max_value * 255.0)
        reconstructed_frame[:, :, channel_index] = np.sum(channel_value).astype(
            np.uint8
        )
    lbp_scaled = cv2.normalize(
        local_binary_pattern, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
    )
    edge_scaled = cv2.normalize(
        edge_features, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
    )
    reconstructed_gray = cv2.addWeighted(lbp_scaled, 0.5, edge_scaled, 0.5, 0)
    reconstructed_frame = cv2.cvtColor(reconstructed_frame, cv2.COLOR_BGR2GRAY)
    reconstructed_frame = cv2.addWeighted(
        reconstructed_frame, 0.5, reconstructed_gray, 0.5, 0
    )
    return cv2.cvtColor(reconstructed_frame, cv2.COLOR_GRAY2BGR)


def extract_image_features(image_path):
    import cv2

    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image could not be read.")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return _extract_visual_features(image, gray_image)
    except Exception as exception:
        _log_media_exception("Failed to extract image features", exception)
        return None


def features_to_image(predicted_features, image_shape=(1024, 1024, 3)):
    try:
        return _reconstruct_visual_frame(predicted_features, image_shape)
    except Exception as exception:
        _log_media_exception(
            "Failed to reconstruct image from features", exception
        )
        return None


def get_max_resolution(width, height, mega_pixels=0.25, factor=16):
    max_pixels = mega_pixels * 1000 * 1000
    ratio = width / height
    best_candidate = None
    best_error = float("inf")
    best_pixels = -1
    max_h = int((max_pixels / max(ratio, 1e-9)) ** 0.5) + factor
    for h_factored in range(factor, max_h + factor, factor):
        w_estimate = int(h_factored * ratio)
        w_factored = max(factor, (w_estimate // factor) * factor)
        if w_factored * h_factored > max_pixels:
            w_factored -= factor
        if w_factored < factor:
            continue
        current_pixels = w_factored * h_factored
        if current_pixels <= 0 or current_pixels > max_pixels:
            continue
        current_error = abs(ratio - (w_factored / h_factored))
        if (current_error < best_error) or (
            abs(current_error - best_error) < 1e-12
            and current_pixels > best_pixels
        ):
            best_error = current_error
            best_pixels = current_pixels
            best_candidate = (w_factored, h_factored)
    if best_candidate is None:
        best_candidate = (factor, factor)
    (w_factored, h_factored) = best_candidate
    ratio_error_factored = abs(ratio - (w_factored / h_factored))

    h_exact = int((max_pixels / ratio) ** 0.5)
    w_exact = int(h_exact * ratio)
    if w_exact * h_exact > max_pixels:
        h_exact = max(1, h_exact - 1)
        w_exact = max(1, int(h_exact * ratio))
    ratio_error_exact = abs(ratio - (w_exact / h_exact))
    if ratio_error_factored <= 0.005:
        return (w_factored, h_factored)
    allow_exact_fallback = mega_pixels >= 0.5 and factor == 16
    if (
        allow_exact_fallback
        and w_exact * h_exact <= max_pixels
        and ratio_error_exact < ratio_error_factored
    ):
        return (w_exact, h_exact)
    return (w_factored, h_factored)


def save_image(img, path=None):
    from definers.system.output_paths import managed_output_path

    random_name = _image_random_string()
    if path is None:
        name = managed_output_path(
            "png",
            section="image",
            stem=f"img_{random_name}",
        )
    else:
        resolved_path = str(path)
        if os.path.splitext(resolved_path)[1]:
            Path(resolved_path).parent.mkdir(parents=True, exist_ok=True)
            name = resolved_path
        else:
            os.makedirs(resolved_path, exist_ok=True)
            name = os.path.join(resolved_path, f"img_{random_name}.png")
    img.save(name)
    return name


def _image_random_string():
    from definers.text import random_string

    return random_string()


def resize_image(image_path, target_width, target_height, anti_aliasing=True):
    import imageio as iio
    from PIL import Image
    from skimage.transform import resize

    from definers.system import catch, tmp

    image_data = iio.imread(image_path)
    try:
        if image_data.ndim < 2:
            raise ValueError(
                "Input image must have at least 2 dimensions (height, width)."
            )
        resized_image = resize(
            image_data,
            (target_height, target_width),
            anti_aliasing=anti_aliasing,
        )
        image = Image.fromarray((resized_image * 255).astype(np.uint8))
        output_path = save_image(image, tmp("png", keep=False))
        return (output_path, image)
    except Exception as exception:
        catch(exception)
        return None


def image_resolution(image_path):
    from PIL import Image

    with Image.open(image_path) as image:
        return image.size


def write_on_image(
    image_path, top_title=None, middle_title=None, bottom_title=None
):
    from PIL import Image, ImageDraw, ImageFont

    from definers.media.web_transfer import google_drive_download
    from definers.system.output_paths import managed_output_path

    font_path = managed_output_path(
        section="image_assets",
        filename="Alef-Bold.ttf",
        unique=False,
    )
    if not os.path.exists(font_path):
        google_drive_download(
            "1C48KkYWQDYu7ypbNtSXAUJ6kuzoZ42sI",
            font_path,
        )
    img = Image.open(image_path)
    (w, h) = img.size
    draw = ImageDraw.Draw(img)

    def draw_text_block(text_block, vertical_position):
        if not text_block:
            return
        text_block = text_block.strip()
        num_lines = max(1, len(text_block.split("\n")))
        font_size = min(math.ceil(w / 12), math.ceil(h / (num_lines * 4)))
        font = ImageFont.truetype(font_path, font_size)
        text_bbox = draw.textbbox((0, 0), text_block, font=font)
        total_text_height = text_bbox[3] - text_bbox[1]
        if vertical_position == "top":
            y = h * 0.15 - total_text_height / 2
        elif vertical_position == "middle":
            y = h / 2 - total_text_height / 2
        else:
            y = h * 0.85 - total_text_height / 2
        text_width = text_bbox[2] - text_bbox[0]
        x = (w - text_width) / 2
        stroke_width = math.ceil(font_size / 20)
        if vertical_position == "top":
            (fill_color, stroke_color) = ((255, 255, 255), (0, 0, 0))
        elif vertical_position == "middle":
            (fill_color, stroke_color) = ((255, 255, 255), (64, 64, 64))
        else:
            (fill_color, stroke_color) = ((0, 0, 0), (255, 255, 255))
        draw.text(
            (x, y),
            text_block,
            font=font,
            fill=fill_color,
            stroke_width=stroke_width,
            stroke_fill=stroke_color,
            spacing=4,
        )

    draw_text_block(top_title, "top")
    draw_text_block(middle_title, "middle")
    draw_text_block(bottom_title, "bottom")
    return save_image(img)
