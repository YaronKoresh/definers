import os
import tempfile

from definers.audio import analyze_audio, get_color_palette
from definers.constants import STYLES_DB

from definers.data import init_cupy_numpy

np, _ = init_cupy_numpy()

def render_frame_base(
    style, t, width, height, audio_data, params, rms, is_beat, img_array=None
):
    import cv2

    colors = get_color_palette(params["palette"])
    (cx, cy) = (width // 2, height // 2)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    if style == "Psychedelic Geometry":
        bg_color = np.array(colors[0]) * (0.1 + 0.1 * np.sin(t * 0.5))
        frame[:] = bg_color
        (grid_x, grid_y) = np.meshgrid(np.arange(width), np.arange(height))
        dist = np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
        angle = np.arctan2(grid_y - cy, grid_x - cx)
        num_arms = int(5 + 10 * rms)
        pattern = np.sin(
            dist / (30 - 10 * rms) + angle * num_arms - t * (5 + 10 * rms)
        )
        pattern_rgb = np.zeros_like(frame)
        mask = pattern > 0
        pattern_rgb[mask] = colors[1]
        pattern_rgb[~mask] = np.array(colors[2]) * rms
        frame = cv2.addWeighted(
            frame, 0.6, pattern_rgb.astype(np.uint8), 0.4 * rms, 0
        )
        radius = int(height * 0.15 + rms * (height * 0.25))
        if is_beat:
            cv2.circle(
                frame,
                (cx, cy),
                int(radius * 1.4),
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.circle(frame, (cx, cy), radius, colors[2], 4)
    elif style == "Spectrum Bars":
        frame[:] = (10, 10, 15)
        total_frames = audio_data["stft"].shape[1]
        frame_idx = int(t * audio_data["sr"] / audio_data["hop_length"])
        safe_idx = min(frame_idx, total_frames - 1)
        stft_col = audio_data["stft"][:, safe_idx]
        num_bars = 64
        bar_w = width // num_bars
        for i in range(num_bars):
            val = np.mean(stft_col[i * 2 : (i + 1) * 2]) * params["sensitivity"]
            bar_h = int(val * height * 0.8)
            c = colors[i % len(colors)]
            if is_beat and i % 4 == 0:
                c = (200, 200, 255)
            cv2.rectangle(
                frame,
                (i * bar_w, height),
                ((i + 1) * bar_w - 2, height - bar_h),
                c,
                -1,
            )
    elif style == "Particle Tunnel":
        for i in range(60):
            seed = i * 13
            speed_boost = 3.0 if is_beat else 1.0
            dist_base = (t * 200 * speed_boost + seed * 10) % (
                max(width, height) / 1.1
            )
            angle = i * (360 / 60) + t * 20
            rad = np.deg2rad(angle)
            px = int(cx + dist_base * np.cos(rad))
            py = int(cy + dist_base * np.sin(rad))
            c = colors[i % len(colors)]
            size = int(3 + dist_base / (width / 2) * 10 * rms)
            cv2.circle(frame, (px, py), size, c, -1)
    elif style == "Glitch Art":
        noise = np.random.randint(
            0, 50 + int(100 * rms), (height, width, 3), dtype=np.uint8
        )
        frame = noise
        if rms > 0.2:
            (x, y) = (np.random.randint(0, width), np.random.randint(0, height))
            (w, h) = (np.random.randint(50, 400), np.random.randint(10, 100))
            cv2.rectangle(frame, (x, y), (x + w, y + h), colors[0], -1)
    elif style == "Liquid Bass":
        frame[:] = (0, 0, 20)
        for i in range(10):
            y_off = np.sin(t * 2 + i) * 50 * rms
            pts = []
            for x in range(0, width, 20):
                y = (
                    cy
                    + np.sin(x * 0.01 + t * (1 + i * 0.2)) * (100 + rms * 200)
                    + y_off
                )
                pts.append((x, int(y)))
            cv2.polylines(
                frame, [np.array(pts)], False, colors[i % 3], 2 + int(rms * 5)
            )
    elif style == "Image Pulse" and img_array is not None:
        scale = 1.0 + (0.2 if is_beat else 0.0) + rms * 0.15
        (h, w_img) = img_array.shape[:2]
        (new_w, new_h) = (int(w_img / scale), int(h / scale))
        (sx, sy) = (max(0, (w_img - new_w) // 2), max(0, (h - new_h) // 2))
        crop = img_array[sy : sy + new_h, sx : sx + new_w]
        if crop.size > 0:
            frame = cv2.resize(crop, (width, height))
        if is_beat:
            frame = cv2.addWeighted(
                frame, 0.7, np.full_like(frame, 255), 0.3, 0
            )
    return frame


def draw_custom_element(frame, element_type, config, t, width, height, rms):
    import cv2

    if element_type == "None":
        return frame
    pos_x = int(config.get("x", 0.5) * width)
    pos_y = int(config.get("y", 0.5) * height)
    scale = config.get("scale", 1.0)
    opacity = config.get("opacity", 1.0)
    overlay = frame.copy()
    if element_type == "Custom Text":
        text = config.get("text_content", "AI VIDEO")
        font_scale = 2 * scale + rms * 0.5
        color = (255, 255, 255)
        thickness = 2
        text_size = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )[0]
        tx = pos_x - text_size[0] // 2
        ty = pos_y + text_size[1] // 2
        cv2.putText(
            overlay,
            text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
        )
    elif element_type == "Logo Image" and config.get("logo_path"):
        try:
            logo = cv2.imread(config["logo_path"], cv2.IMREAD_UNCHANGED)
            if logo is not None:
                target_h = int(height * 0.2 * scale)
                ratio = target_h / logo.shape[0]
                target_w = int(logo.shape[1] * ratio)
                logo = cv2.resize(logo, (target_w, target_h))
                y1 = pos_y - target_h // 2
                x1 = pos_x - target_w // 2
                if (
                    x1 >= 0
                    and y1 >= 0
                    and (x1 + target_w <= width)
                    and (y1 + target_h <= height)
                ):
                    if logo.shape[2] == 4:
                        alpha = logo[:, :, 3] / 255.0
                        bg_patch = frame[y1 : y1 + target_h, x1 : x1 + target_w]
                        for c in range(3):
                            bg_patch[:, :, c] = (1.0 - alpha) * bg_patch[
                                :, :, c
                            ] + alpha * logo[:, :, c] * opacity
                        frame[y1 : y1 + target_h, x1 : x1 + target_w] = bg_patch
                        return frame
                    else:
                        frame[y1 : y1 + target_h, x1 : x1 + target_w] = logo
        except Exception as e:
            print(f"Error drawing logo: {e}")
    elif element_type == "Spectrum Circle":
        radius = int(100 * scale * (1 + rms))
        color = (0, 255, 255)
        cv2.circle(overlay, (pos_x, pos_y), radius, color, 2)
        cv2.circle(overlay, (pos_x, pos_y), int(radius * 0.8), (255, 0, 255), 1)
    if element_type != "Logo Image":
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
    return frame


def apply_global_overlays(
    frame, t, width, height, active_features, rms, is_beat, duration
):
    import cv2

    if "Neon Border" in active_features:
        thickness = int(10 + 20 * rms)
        color = (int(255 * abs(np.sin(t))), 50, 255)
        cv2.rectangle(frame, (0, 0), (width, height), color, thickness)
    if "Progress Bar" in active_features and duration > 0:
        bar_height = 10
        progress = t / duration
        bar_width = int(width * progress)
        cv2.rectangle(
            frame, (0, height - bar_height), (width, height), (50, 50, 50), -1
        )
        cv2.rectangle(
            frame,
            (0, height - bar_height),
            (bar_width, height),
            (0, 255, 0),
            -1,
        )
    if "Audio Waveform" in active_features:
        center_y = height - 100
        pts = []
        for x in range(0, width, 5):
            amp = rms * 50 * np.sin(x * 0.1 + t * 10)
            pts.append((x, int(center_y + amp)))
        cv2.polylines(frame, [np.array(pts)], False, (255, 255, 255), 2)
    if "Timer" in active_features:
        (m, s) = (int(t // 60), int(t % 60))
        cv2.putText(
            frame,
            f"{m:02d}:{s:02d}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
    return frame


def normalize_arr(a):
    import numpy as np

    if hasattr(a, "size") and a.size == 0:
        return a

    ptp = np.ptp(a)
    if ptp == 0:
        return np.zeros_like(a, dtype=float)
    return (a - np.min(a)) / ptp


def apply_post_fx(frame, effects, rms):
    import cv2

    if "Vignette" in effects:
        (rows, cols) = frame.shape[:2]
        (Y, X) = np.ogrid[:rows, :cols]
        (center_y, center_x) = (rows / 2, cols / 2)
        dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        mask = 1 - normalize_arr(dist_from_center)
        mask = np.clip(mask * 1.5, 0, 1)
        mask = np.dstack([mask] * 3)
        frame = (frame * mask).astype(np.uint8)
    if "Scanlines" in effects:
        frame[::4, :] = (frame[::4, :] * 0.7).astype(np.uint8)
    if "Noise" in effects:
        noise = np.random.normal(0, 10 + 20 * rms, frame.shape).astype(np.uint8)
        frame = cv2.addWeighted(frame, 0.9, noise, 0.1, 0)
    return frame


def get_rms_and_beat(t, adata, reactivity_band, sensitivity):
    total_frames = adata["stft"].shape[1]
    frame_idx = int(t * adata["sr"] / adata["hop_length"])
    safe_idx = min(frame_idx, total_frames - 1)
    if reactivity_band == "Low":
        raw_rms = adata["rms_low"][safe_idx]
    elif reactivity_band == "Mid":
        raw_rms = adata["rms_mid"][safe_idx]
    elif reactivity_band == "High":
        raw_rms = adata["rms_high"][safe_idx]
    else:
        raw_rms = adata["rms"][safe_idx]
    rms = raw_rms * sensitivity
    is_beat = any(abs(frame_idx - bf) < 3 for bf in adata["beat_frames"])
    return (rms, is_beat)


def prepare_common_resources(audio, image, resolution):
    import cv2
    from PIL import Image, ImageOps

    if not audio:
        return (None, None, None, "Error: No Audio File")
    res_map = {
        "Square (1:1)": (720, 720),
        "Portrait (9:16)": (720, 1280),
        "Landscape (16:9)": (1280, 720),
    }
    (w, h) = res_map.get(resolution, (1280, 720))
    img_array = None
    if image:
        try:
            pil = Image.open(image).convert("RGB")
            pil = ImageOps.fit(pil, (w, h), Image.Resampling.LANCZOS)
            img_array = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except:
            print("Failed to load background image")
    return (w, h, img_array, None)


def generate_preview_handler(
    audio,
    image,
    style,
    resolution,
    sensitivity,
    reactivity_band,
    palette,
    active_overlays,
    post_effects,
    custom_elem_type,
    ce_x,
    ce_y,
    ce_scale,
    ce_opacity,
    ce_text,
    ce_logo,
):
    import cv2

    (w, h, img_array, error) = prepare_common_resources(
        audio, image, resolution
    )
    if error:
        return (None, error)
    try:
        adata = analyze_audio(audio, duration=10)
        preview_t = 3.0 if adata["duration"] > 3 else adata["duration"] / 2
    except Exception as e:
        return (None, f"Audio Analysis Failed: {e}")
    params = {"sensitivity": sensitivity, "palette": palette}
    (rms, is_beat) = get_rms_and_beat(
        preview_t, adata, reactivity_band, sensitivity
    )
    custom_config = {
        "x": ce_x,
        "y": ce_y,
        "scale": ce_scale,
        "opacity": ce_opacity,
        "text_content": ce_text,
        "logo_path": ce_logo,
    }
    frame = render_frame_base(
        style, preview_t, w, h, adata, params, rms, is_beat, img_array
    )
    frame = draw_custom_element(
        frame, custom_elem_type, custom_config, preview_t, w, h, rms
    )
    frame = apply_global_overlays(
        frame, preview_t, w, h, active_overlays, rms, is_beat, adata["duration"]
    )
    frame = apply_post_fx(frame, post_effects, rms)
    return (
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        f"Preview at {preview_t:.2f}s | BPM: {adata['bpm']}",
    )


def generate_video_handler(
    audio,
    image,
    style,
    resolution,
    fps,
    sensitivity,
    reactivity_band,
    palette,
    active_overlays,
    post_effects,
    custom_elem_type,
    ce_x,
    ce_y,
    ce_scale,
    ce_opacity,
    ce_text,
    ce_logo,
):
    import cv2
    from moviepy import AudioFileClip, VideoClip

    (w, h, img_array, error) = prepare_common_resources(
        audio, image, resolution
    )
    if error:
        return (None, error)
    print("Analyzing full audio...")
    adata = analyze_audio(audio)
    params = {"sensitivity": sensitivity, "palette": palette}
    custom_config = {
        "x": ce_x,
        "y": ce_y,
        "scale": ce_scale,
        "opacity": ce_opacity,
        "text_content": ce_text,
        "logo_path": ce_logo,
    }

    def make_frame(t):
        (rms, is_beat) = get_rms_and_beat(
            t, adata, reactivity_band, sensitivity
        )
        frame = render_frame_base(
            style, t, w, h, adata, params, rms, is_beat, img_array
        )
        frame = draw_custom_element(
            frame, custom_elem_type, custom_config, t, w, h, rms
        )
        frame = apply_global_overlays(
            frame, t, w, h, active_overlays, rms, is_beat, adata["duration"]
        )
        frame = apply_post_fx(frame, post_effects, rms)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    clip = VideoClip(make_frame, duration=adata["duration"])
    clip = clip.with_audio(AudioFileClip(audio))

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    output = tmp.name

    try:
        tmp.close()

        print("Rendering...")
        clip.write_videofile(
            output,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            logger=None,
        )
        return (output, f"Render Complete! Duration: {adata['duration']:.2f}s")
    finally:
        if os.path.exists(output):
            try:
                os.remove(output)
                print("Temporary file cleaned up successfully.")
            except OSError as e:
                print(f"Error cleaning up: {e}")


def filter_styles(query, category):
    import gradio as gr

    filtered = []
    for name, data in STYLES_DB.items():
        text_match = (
            query.lower() in name.lower()
            or query.lower() in data["desc"].lower()
        )
        tag_match = category == "All" or category in data["tags"]
        if text_match and tag_match:
            filtered.append(name)
    return gr.update(choices=filtered, value=filtered[0] if filtered else None)


def draw_star_of_david(frame, center, radius, angle, color, thickness):
    import cv2

    (center_x, center_y) = center
    points = []
    for i in range(6):
        point_angle = np.deg2rad(60 * i + 30 + angle)
        x = center_x + radius * np.cos(point_angle)
        y = center_y + radius * np.sin(point_angle)
        points.append((int(x), int(y)))
    triangle1 = np.array([points[0], points[2], points[4]], np.int32)
    triangle2 = np.array([points[1], points[3], points[5]], np.int32)
    cv2.polylines(
        frame,
        [triangle1],
        isClosed=True,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    cv2.polylines(
        frame,
        [triangle2],
        isClosed=True,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
