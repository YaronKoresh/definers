from pathlib import Path

from definers.constants import STYLES_DB
from definers.runtime_numpy import get_array_module, get_numpy_module

np = get_array_module()


def _ui_update(**kwargs):
    try:
        import gradio as gr

        return gr.update(**kwargs)
    except Exception:
        return dict(kwargs)


def _build_custom_element_config(
    ce_x,
    ce_y,
    ce_scale,
    ce_opacity,
    ce_text,
    ce_logo,
):
    return {
        "x": ce_x,
        "y": ce_y,
        "scale": ce_scale,
        "opacity": ce_opacity,
        "text_content": ce_text,
        "logo_path": ce_logo,
    }


def _render_composed_frame(
    t,
    style,
    width,
    height,
    audio_data,
    params,
    reactivity_band,
    sensitivity,
    img_array,
    custom_elem_type,
    custom_config,
    active_overlays,
    post_effects,
    duration,
):
    (rms, is_beat) = get_rms_and_beat(
        t, audio_data, reactivity_band, sensitivity
    )
    frame = render_frame_base(
        style, t, width, height, audio_data, params, rms, is_beat, img_array
    )
    frame = draw_custom_element(
        frame, custom_elem_type, custom_config, width, height, rms
    )
    frame = apply_global_overlays(
        frame, t, width, height, active_overlays, rms, duration
    )
    return apply_post_fx(frame, post_effects, rms)


def normalize_audio_payload(audio_data):
    if not isinstance(audio_data, dict):
        return {}

    normalized = dict(audio_data)
    stft = normalized.get("stft")
    if stft is None:
        stft = normalized.get("stft_db")
        if stft is not None:
            normalized["stft"] = stft

    hop_length = normalized.get("hop_length")
    if not isinstance(hop_length, int) or hop_length <= 0:
        normalized["hop_length"] = 1024

    if normalized.get("rms") is None:
        band_arrays = []
        for key in ("rms_low", "rms_mid", "rms_high"):
            band = normalized.get(key)
            if getattr(band, "size", 0) > 0:
                band_arrays.append(np.asarray(band))
        if band_arrays:
            min_length = min(band.shape[0] for band in band_arrays)
            normalized["rms"] = np.mean(
                np.vstack([band[:min_length] for band in band_arrays]), axis=0
            )
        elif getattr(stft, "shape", ()) and len(stft.shape) > 1:
            normalized["rms"] = np.zeros(stft.shape[1], dtype=float)
        else:
            normalized["rms"] = np.array([], dtype=float)

    if normalized.get("beat_frames") is None:
        normalized["beat_frames"] = []

    return normalized


def render_frame_base(
    style, t, width, height, audio_data, params, rms, is_beat, img_array=None
):
    import cv2

    from definers.audio import get_color_palette

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
        stft = audio_data.get("stft")
        if stft is None or getattr(stft, "size", 0) == 0 or len(stft.shape) < 2:
            return frame
        total_frames = stft.shape[1]
        sr = audio_data.get("sr", 0)
        hop_length = audio_data.get("hop_length", 1024)
        if total_frames <= 0 or sr <= 0 or hop_length <= 0:
            return frame
        frame_idx = max(0, int(t * sr / hop_length))
        safe_idx = min(frame_idx, total_frames - 1)
        stft_col = stft[:, safe_idx]
        num_bars = 64
        bar_w = width // num_bars
        for i in range(num_bars):
            band = stft_col[i * 2 : (i + 1) * 2]
            val = np.mean(band) * params["sensitivity"] if band.size else 0.0
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


def draw_custom_element(frame, element_type, config, width, height, rms):
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
    frame, t, width, height, active_features, rms, duration
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
    np = get_numpy_module()

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
    stft = adata.get("stft")
    if stft is None or getattr(stft, "size", 0) == 0 or len(stft.shape) < 2:
        return (0.0, False)

    total_frames = stft.shape[1]
    sr = adata.get("sr", 0)
    hop_length = adata.get("hop_length", 1024)
    if total_frames <= 0 or sr <= 0 or hop_length <= 0:
        return (0.0, False)

    frame_idx = max(0, int(t * sr / hop_length))
    safe_idx = max(0, min(frame_idx, total_frames - 1))

    def resolve_rms(series_name):
        series = adata.get(series_name)
        if series is None or getattr(series, "size", 0) == 0:
            return 0.0
        if safe_idx >= len(series):
            return float(series[-1]) if len(series) else 0.0
        return float(series[safe_idx])

    if reactivity_band == "Low":
        raw_rms = resolve_rms("rms_low")
    elif reactivity_band == "Mid":
        raw_rms = resolve_rms("rms_mid")
    elif reactivity_band == "High":
        raw_rms = resolve_rms("rms_high")
    else:
        raw_rms = resolve_rms("rms")
    rms = raw_rms * sensitivity
    try:
        is_beat = any(
            abs(frame_idx - bf) < 3 for bf in adata.get("beat_frames", [])
        )
    except Exception:
        is_beat = False
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

    from definers.audio import analyze_audio
    from definers.system.download_activity import (
        create_activity_reporter,
    )
    from definers.system.output_paths import managed_output_path

    report = create_activity_reporter(5)
    report(
        1,
        "Validate media",
        detail="Checking the selected media and composition settings.",
    )
    (w, h, img_array, error) = prepare_common_resources(
        audio, image, resolution
    )
    if error:
        return (None, error)
    print("Analyzing full audio...")
    report(
        2,
        "Analyze audio",
        detail="Analyzing the input audio track.",
    )
    adata = normalize_audio_payload(analyze_audio(audio))
    duration = adata.get("duration", 0.0)
    if duration <= 0:
        return (None, "Audio Analysis Failed: Invalid duration")
    params = {"sensitivity": sensitivity, "palette": palette}
    custom_config = _build_custom_element_config(
        ce_x, ce_y, ce_scale, ce_opacity, ce_text, ce_logo
    )
    render_frame_total = max(int(duration * max(int(fps), 1)), 1)
    render_report = create_activity_reporter(render_frame_total)
    render_update_interval = max(render_frame_total // 180, 1)
    last_reported_frame = 0

    def make_frame(t):
        nonlocal last_reported_frame

        frame_number = min(
            max(int(t * max(int(fps), 1)) + 1, 1),
            render_frame_total,
        )
        if (
            frame_number == 1
            or frame_number == render_frame_total
            or frame_number - last_reported_frame >= render_update_interval
        ) and frame_number != last_reported_frame:
            last_reported_frame = frame_number
            render_report(
                frame_number,
                "Render video frames",
                detail=f"Rendering frame {frame_number}/{render_frame_total}.",
            )
        frame = _render_composed_frame(
            t,
            style,
            w,
            h,
            adata,
            params,
            reactivity_band,
            sensitivity,
            img_array,
            custom_elem_type,
            custom_config,
            active_overlays,
            post_effects,
            duration,
        )
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    report(
        3,
        "Prepare composition",
        detail="Building the video composition pipeline.",
    )
    clip = VideoClip(make_frame, duration=duration)
    clip = clip.with_audio(AudioFileClip(audio))

    output = managed_output_path(
        "mp4",
        section="video",
        stem=f"{Path(audio).stem}_composition",
    )

    print("Rendering...")
    report(
        4,
        "Render video frames",
        detail=f"Encoding {render_frame_total} video frames.",
    )
    clip.write_videofile(
        output,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        preset="ultrafast",
        logger=None,
    )
    report(
        5,
        "Finalize video",
        detail="Writing the rendered video output.",
    )
    return (output, f"Render Complete! Duration: {duration:.2f}s")


def filter_styles(query, category):
    filtered = []
    for name, data in STYLES_DB.items():
        text_match = (
            query.lower() in name.lower()
            or query.lower() in data["desc"].lower()
        )
        tag_match = category == "All" or category in data["tags"]
        if text_match and tag_match:
            filtered.append(name)
    return _ui_update(
        choices=filtered,
        value=filtered[0] if filtered else None,
    )


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
