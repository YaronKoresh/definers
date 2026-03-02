import gc
import math
import os
import random
import re
import tempfile

from definers._audio import analyze_audio, get_color_palette, value_to_keys
from definers._constants import MODELS, STYLES_DB, language_codes
from definers._cuda import device
from definers._image import (
    get_max_resolution,
    init_upscale,
    upscale,
    write_on_image,
)
from definers._ml import (
    answer,
    build_faiss,
    init_pretrained_model,
    keep_alive,
    optimize_prompt_realism,
    pipe,
)
from definers._system import (
    catch,
    cores,
    delete,
    full_path,
    install_ffmpeg,
    log,
    pip_install,
    run,
    tmp,
    unique,
)
from definers._text import ai_translate, language, simple_text

try:
    import cupy as np
except Exception:
    import numpy as np


def theme():
    import gradio as gr

    return gr.themes.Base(
        primary_hue=gr.themes.colors.slate,
        secondary_hue=gr.themes.colors.indigo,
        font=(
            gr.themes.GoogleFont("Inter"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ),
    ).set(
        body_background_fill_dark="#111827",
        block_background_fill_dark="#1f2937",
        block_border_width="1px",
        block_title_background_fill_dark="#374151",
        button_primary_background_fill_dark="linear-gradient(90deg, #4f46e5, #7c3aed)",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill_dark="#374151",
        button_secondary_text_color_dark="#ffffff",
        slider_color_dark="#6366f1",
    )


def css():
    return '\n\nvideo {\n    border-radius: 20px;\n}\n\ndiv:has(>video) {\n    padding: 20px 20px 0px 20px !important;\n}\n\nspan {\n    margin-block: 0 !important;\n}\n\nlabel.container > span {\n    width: 100% !important;\n}\n\nhtml > body .gradio-container > main *:not(img, svg, span, :has(>svg)):is(*, *::placeholder) {\n    scrollbar-width: none !important;\n    text-align: center !important;\n    max-width: 100% !important;\n}\n\ndiv:not(.styler) > :is(.block:has(+ .block), .block + .block):has(:not(span, div, h1, h2, h3, h4, h5, h6, p, strong)) {\n    border: 1px dotted slategray !important;\n    margin-block: 10px !important;\n}\n\n.row {\n    padding-block: 20px !important;\n}\n\nlabel > input[type="radio"] {\n    border: 2px ridge black !important;\n    flex-grow: 0 !important;\n}\n\nlabel.selected > input[type="radio"] {\n    background: lime !important;\n}\n\nlabel:has(>input[type="radio"]) {\n    flex-grow: 0 !important;\n}\n\ndiv.form:has(>fieldset.block) {\n    margin-block: 10px !important;\n}\n\ndiv.controls {\n    width: 100% !important;\n}\n\ndiv.controls > * {\n    flex-grow: 1 !important;\n}\n\n    html > body .gradio-container {\n        padding: 0 !important;\n    }\n\n        html > body footer {\n            opacity: 0 !important;\n            visibility: hidden !important;\n            width: 0px !important;\n            height: 0px !important;\n        }\n\n    tr.file > td.download {\n        min-width: unset !important;\n        width: auto !important;\n    }\n\nhtml > body main {\n    padding-inline: 20px !important;\n}\n\n    button {\n        border-radius: 2mm !important;\n        border: none !important;\n        cursor: pointer !important;\n        margin-inline: 8px !important;\n    }\n\n    button:not(:has(svg)) {\n        width: auto !important;\n    }\n\n    textarea {\n        border: 1px solid #ccc !important;\n        border-radius: 5px !important;\n        padding: 8px !important;\n    }\n\n    textarea:focus{\n        border-color: #4CAF50 !important;\n        outline: none !important;\n        box-shadow: 0 0 5px rgba(76, 175, 80, 0.5) !important;\n    }\n\n    h1 {\n        color: #333 !important;\n    }\n\n    h2 {\n        color: #444 !important;\n    }\n\n    h3{\n        color: #555 !important;\n    }\n\n    '


def get_chat_response(message, history: list):
    history = list(history)
    orig_lang = None
    including = []
    if message["files"]:
        including.append("files")
        for file_path in message["files"]:
            history.append({"role": "user", "content": {"path": file_path}})
    if message["text"]:
        including.append("text")
        orig_lang = language(message["text"])
        if orig_lang != "en":
            message["text"] = ai_translate(message["text"])
        message["text"] = simple_text(message["text"])
        history.append({"role": "user", "content": message["text"]})
        if message["files"]:
            history.append(
                {
                    "role": "user",
                    "content": "and please read the media from my new message carefully",
                }
            )
    nl = "\n"
    including = "\n".join(including)
    if orig_lang is None:
        msg = f"Got a new message.{nl}{nl}The message including the following types of data:{nl}{including}"
    else:
        msg = f"Got a new message in {language_codes[orig_lang]}.{nl}{nl}The message including the following types of data:{nl}{including}"
    log("Chat", msg)
    response = answer(history)
    log("Chatbot response", response)
    return response


def init_chat(title="Chatbot", handler=get_chat_response):
    import gradio as gr

    chatbot = gr.Chatbot(
        elem_id="chatbot", type="messages", show_copy_button=True
    )
    return gr.ChatInterface(
        fn=handler,
        type="messages",
        chatbot=chatbot,
        multimodal=True,
        theme=theme(),
        title=title,
        css=css(),
        save_history=True,
        show_progress="hidden",
        concurrency_limit=None,
    )


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

    return (a - np.min(a)) / np.ptp(a)


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


def calculate_gpu_duration(
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
    from moviepy import AudioFileClip

    try:
        if not audio:
            return False
        clip = AudioFileClip(audio)
        audio_dur = clip.duration
        clip.close()
        res_factor = 1.0
        if resolution != "Square (1:1)":
            res_factor = 1.5
        fps_factor = fps / 20.0
        estimated_time = audio_dur * res_factor * fps_factor + 20
        final_duration = int(max(60, min(int(estimated_time), 360)))
        print(
            f"ZeroGPU Timeout: {final_duration}s (Audio: {audio_dur}s, FPS: {fps}, Res: {resolution})"
        )
        return final_duration
    except Exception as e:
        print(f"Error calculating duration: {e}")
        return False


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


def music_video(audio_path, width=1920, height=1080, fps=30):
    import cv2
    import librosa
    import madmom
    from moviepy import AudioFileClip
    from moviepy.video.VideoClip import VideoClip

    hop_length = 512
    (y, sr) = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)
    stft = librosa.stft(y, hop_length=hop_length)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_length
    )[0]
    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(audio_path)
    beat_times = proc(act)
    beat_frames = librosa.time_to_frames(
        beat_times, sr=sr, hop_length=hop_length
    )
    rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-06)
    centroid_norm = (spectral_centroid - np.min(spectral_centroid)) / (
        np.max(spectral_centroid) - np.min(spectral_centroid) + 1e-06
    )
    stft_norm = (stft_db - np.min(stft_db)) / (
        np.max(stft_db) - np.min(stft_db) + 1e-06
    )
    (w, h) = (width, height)
    (center_x, center_y) = (w // 2, h // 2)

    def make_frame(t):
        frame_idx = int(t * sr / hop_length)
        safe_idx = min(frame_idx, len(rms_norm) - 1, len(centroid_norm) - 1)
        (grid_x, grid_y) = np.meshgrid(np.arange(w), np.arange(h))
        rms_val = rms_norm[safe_idx]
        centroid_val = centroid_norm[safe_idx]
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        base_radius = h * 0.12
        radius = int(base_radius + rms_val * (h * 0.2))
        is_beat = any(abs(frame_idx - bf) < 3 for bf in beat_frames)
        if is_beat:
            radius = int(radius / rms_val / centroid_val)
            frame += random.randint(32, 192)
        angle = np.arctan2(grid_y - center_y, grid_x - center_x)
        dist = np.sqrt((grid_y - center_y) ** 2 + (grid_x - center_x) ** 2)
        num_arms1 = round(5 + 10 * rms_val * centroid_val)
        num_arms2 = round(5 + 10 * centroid_val)
        pattern1 = np.sin(dist / 30.0 + angle * num_arms1 - t * 10.0)
        pattern2 = np.cos(dist / 10.0 - angle * num_arms2 + t * 30.0)
        pattern_freq = 10.0 + centroid_val * 20.0
        base_pattern = np.sin(
            grid_x / (60 + rms_val * 100) * pattern_freq + t * 5
        ) * np.cos(grid_y / 40 * pattern_freq - t * 3)
        brightness = 0.5 + rms_val * 0.5
        r = 128 + 127 * np.sin(base_pattern * np.pi) * brightness
        g = 128 + 127 * np.sin(base_pattern * np.pi + np.pi / 2) * brightness
        b = (
            60
            + 100 * centroid_val
            + 60 * np.sin(base_pattern * np.pi + np.pi) * brightness
        )
        frame_rgb = np.stack((r, g, b), axis=-1)
        final_pattern = np.clip(pattern1 * pattern2, -1.0, 1.0)
        rms_shift = int(rms_val * 64)
        if rms_shift > 1:
            frame[:, :, 2] = np.roll(frame[:, :, 2], rms_shift, axis=1)
            frame[:, :, 0] = np.roll(frame[:, :, 0], -rms_shift, axis=1)
        y_start = np.random.randint(0, h - h // 4)
        y_end = y_start + np.random.randint(h // 20, h // 4)
        block_shift = np.random.randint(-w // 4, w // 4)
        frame[y_start:y_end, :] = np.roll(
            frame[y_start:y_end, :], block_shift, axis=1
        )
        (shake_x, shake_y) = np.random.randint(-64, 64, size=2)
        frame = np.roll(np.roll(frame, shake_y, axis=0), shake_x, axis=1)
        ISRAEL_BLUE = (0, 56, 184)
        METALIC_BLACK = (44, 44, 43)
        bg_color_top = np.array([255, 255, 255])
        bg_color_bottom = np.array([230, 230, 250])
        pulse = 0.5 + 0.5 * np.sin(t * np.pi)
        for i in range(h):
            interp_color = bg_color_top * (1 - i / h) + bg_color_bottom * (
                i / h
            ) * (1 - rms_val * 0.2 * pulse)
            frame[i, :] = interp_color.astype(np.uint8)
        stripe_height = int(h * 0.15)
        gap_height = int(h * 0.1)
        frame[gap_height : gap_height + stripe_height] = ISRAEL_BLUE
        frame[h - gap_height - stripe_height : h - gap_height] = ISRAEL_BLUE
        frame = cv2.addWeighted(frame, 0.7, frame_rgb.astype(np.uint8), 0.3, 0)
        rotation_angle = t * 15 + centroid_val * 70
        if is_beat:
            shockwave_radius = int(radius * 1.5)
            shockwave_color = (200, 200, 200)
            cv2.circle(
                frame,
                (center_x, center_y),
                shockwave_radius,
                shockwave_color,
                4,
            )
        draw_star_of_david(
            frame,
            (center_x, center_y),
            radius,
            rotation_angle,
            METALIC_BLACK,
            14,
        )
        distortion_strength = 50 * rms_val
        distortion = final_pattern * distortion_strength
        distortion_3d = distortion[..., np.newaxis]
        frame = np.clip(frame.astype(np.int16) + distortion_3d, 0, 255).astype(
            np.uint8
        )
        scanline_intensity = 0.9 * rms_val
        scanline_effect = (
            np.sin(grid_y * 2 + t * 60) * 25 * scanline_intensity
        ).reshape(h, w, 1)
        frame = np.clip(frame.astype(np.int16) - scanline_effect, 0, 255)
        num_bars = 128
        spectrum = stft_norm[:, min(frame_idx, stft_norm.shape[1] - 1)]
        log_freq_indices = np.logspace(
            0, np.log10(len(spectrum) - 1), num_bars + 1, dtype=int
        )
        bar_values = []
        for i in range(num_bars):
            start_idx = log_freq_indices[i]
            end_idx = log_freq_indices[i + 1]
            if start_idx >= end_idx:
                value = 0.0
            else:
                value = np.mean(spectrum[start_idx:end_idx])
            bar_values.append(value)
        min_radius = 60 + 150 * rms_val
        rotation = t * 30
        points = []
        for i in range(num_bars):
            value = bar_values[i]
            angle = np.deg2rad(i * (360 / num_bars) + rotation)
            bar_length = value * (h * 0.35)
            start_x = int(center_x + min_radius * np.cos(angle))
            start_y = int(center_y + min_radius * np.sin(angle))
            end_x = int(center_x + (min_radius + bar_length) * np.cos(angle))
            end_y = int(center_y + (min_radius + bar_length) * np.sin(angle))
            points.append([end_x, end_y])
            hue = int(i / num_bars * 180)
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][
                0
            ].tolist()
            thickness = 12 if is_beat else 4
            cv2.line(
                frame,
                (start_x, start_y),
                (end_x, end_y),
                color_bgr,
                thickness,
                lineType=cv2.LINE_AA,
            )
        cv2.polylines(
            frame,
            [np.array(points)],
            isClosed=True,
            color=(200, 200, 200),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        center_color_val = min(int(100 + 150 * centroid_val), 255)
        center_color = (center_color_val, center_color_val, 200)
        cv2.circle(
            frame,
            (center_x, center_y),
            int(min_radius * 0.7),
            center_color,
            -1,
            lineType=cv2.LINE_AA,
        )
        if is_beat:
            pulse_color = (255, 255, 255)
            cv2.circle(
                frame,
                (center_x, center_y),
                int(min_radius),
                pulse_color,
                3,
                lineType=cv2.LINE_AA,
            )
        return frame.astype(np.uint8)

    output_path = audio_path.rsplit(".", 1)[0] + "_video.mp4"
    animation = VideoClip(make_frame, duration=duration)
    final_clip = animation.with_audio(AudioFileClip(audio_path))
    final_clip.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=fps,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
        preset="ultrafast",
        threads=cores(),
    )
    return output_path


def strip_nikud(text: str) -> str:
    return "".join(char for char in text if not "֑" <= char <= "ׇ")


def init_stable_whisper():
    import stable_whisper

    global MODELS
    print("Loading multilingual transcription model (stable-ts)...")
    MODELS["stable-whisper"] = stable_whisper.load_model("tiny", device="cpu")


def lyric_video(
    audio_path,
    background_path,
    lyrics_text,
    text_position,
    *,
    max_dim=640,
    font_size=70,
    text_color="white",
    stroke_color="black",
    stroke_width=2,
    fade_duration=0.5,
):
    import edlib
    import torch
    from moviepy import (
        AudioFileClip,
        ColorClip,
        CompositeVideoClip,
        ImageClip,
        TextClip,
        VideoFileClip,
    )
    from moviepy.video import fx as vfx

    def clean_word(text):
        return "".join(filter(str.isalnum, text.lower()))

    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration
    lyrics_text = strip_nikud(lyrics_text)
    detected_lang = language(lyrics_text)
    print(f"🌍 Detected language: {detected_lang}")
    print("🎤 Starting automatic lyric synchronization...")
    timed_lyrics = []
    lines = [
        line.strip() for line in lyrics_text.strip().split("\n") if line.strip()
    ]
    if not lines:
        print("Warning: Lyrics text is empty.")
    else:
        lyrics_text = "\n".join(lines)
        try:
            model = MODELS["stable-whisper"]
            print("Transcribing audio with music-optimized settings...")
            result = model.transcribe(
                audio_path,
                vad=True,
                language=detected_lang,
                no_speech_threshold=None,
                denoiser="demucs",
                denoiser_options={"device": "cpu"},
                word_timestamps=True,
            )
            processed_timestamps = [
                {"word": clean_word(w.word), "start": w.start, "end": w.end}
                for segment in result.segments
                for w in segment.words
            ]
            log("Processed Timestamps", processed_timestamps)
            processed_lines = [
                (
                    line,
                    [clean_word(w) for w in re.findall("\\b[\\w'-]+\\b", line)],
                )
                for line in lines
            ]
            log("Processed Lines", processed_lines)
            correct_words_flat = [
                word
                for (_, line_words) in processed_lines
                for word in line_words
            ]
            transcript_words_flat = [p["word"] for p in processed_timestamps]
            line_boundaries = []
            word_counter = 0
            for _, line_words in processed_lines:
                start_index = word_counter
                word_counter += len(line_words)
                end_index = word_counter
                line_boundaries.append((start_index, end_index))
            alignment = edlib.align(
                correct_words_flat,
                transcript_words_flat,
                mode="NW",
                task="path",
            )
            correct_to_transcript_map = {}
            transcript_idx = -1
            correct_idx = -1
            if alignment["cigar"]:
                operations = re.findall("(\\d+)([=XDI])", alignment["cigar"])
                for length, op in operations:
                    for _ in range(int(length)):
                        if op in ("=", "X"):
                            transcript_idx += 1
                            correct_idx += 1
                            correct_to_transcript_map[correct_idx] = (
                                transcript_idx
                            )
                        elif op == "D":
                            transcript_idx += 1
                        elif op == "I":
                            correct_idx += 1
                            correct_to_transcript_map[correct_idx] = -1
            for i, original_line in enumerate(lines):
                (start_word_idx, end_word_idx) = line_boundaries[i]
                (first_transcript_idx, last_transcript_idx) = (-1, -1)
                for word_i in range(start_word_idx, end_word_idx):
                    mapped_idx = correct_to_transcript_map.get(word_i, -1)
                    if mapped_idx != -1:
                        if first_transcript_idx == -1:
                            first_transcript_idx = mapped_idx
                        last_transcript_idx = mapped_idx
                if first_transcript_idx != -1 and last_transcript_idx != -1:
                    start_time = processed_timestamps[first_transcript_idx][
                        "start"
                    ]
                    end_time = processed_timestamps[last_transcript_idx]["end"]
                    timed_lyrics.append((start_time, end_time, original_line))
            del model, result
            gc.collect()
        except Exception as e:
            catch(
                f"Could not automatically sync lyrics: {e}. Video will have no lyrics."
            )
            return None
    log("Timed Lyrics", timed_lyrics)
    print("✅ Synchronization complete.")
    output_size = (max_dim, max_dim)
    if background_path:
        is_image = background_path.lower().endswith((".png", ".jpg", ".jpeg"))
        if is_image:
            background_clip = ImageClip(background_path, duration=duration)
        else:
            background_clip = VideoFileClip(background_path)
            if background_clip.duration < duration:
                background_clip = vfx.loop(background_clip, duration=duration)
            background_clip = background_clip.with_duration(duration)
        (w, h) = background_clip.size
        if w > max_dim or h > max_dim:
            if w > h:
                new_w = max_dim
                new_h = int(h * (max_dim / w))
            else:
                new_h = max_dim
                new_w = int(w * (max_dim / h))
            background_clip = background_clip.resized(width=new_w, height=new_h)
            print(
                f"Background detected. Downscaling to: {new_w}x{new_h} pixels."
            )
        else:
            print(f"Background detected. Using original size: {w}x{h} pixels.")
        output_size = background_clip.size
    else:
        background_clip = ColorClip(
            size=output_size, color=(0, 0, 0), duration=duration
        )
        print(
            f"No background provided. Using default size: {output_size[0]}x{output_size[1]} pixels."
        )
    lyric_clips = []
    for start, end, line in timed_lyrics:
        clip_duration = round(end - start, 3)
        log("Clip duration", clip_duration)
        if clip_duration <= 0:
            continue
        text_clip = TextClip(
            line,
            font_size=font_size,
            color=text_color,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            method="caption",
            size=(output_size[0] * 0.9, None),
        )
        safe_fade = min(fade_duration, clip_duration / 2)
        text_clip = (
            text_clip.with_position(text_position)
            .with_start(start)
            .with_duration(clip_duration)
            .crossfadein(safe_fade)
            .crossfadeout(safe_fade)
        )
        lyric_clips.append(text_clip)
    log("Lyric clips", len(lyric_clips))
    final_clip = CompositeVideoClip(
        [background_clip] + lyric_clips, size=output_size
    ).with_audio(audio_clip)
    output_path = tmp("mp4", keep=False)
    print(f"Writing video to temporary file: {output_path}")
    final_clip.write_videofile(
        output_path,
        codec="libx264",
        fps=20,
        audio_codec="aac",
        preset="ultrafast",
        threads=cores(),
    )
    print("Video rendering complete.")
    return output_path


def start(proj: str):
    import gradio as gr
    import spaces

    proj = proj.strip().lower()
    if proj == "translate":
        init_pretrained_model("translate", True)

        def title(image_path, top, middle, bottom):
            return write_on_image(image_path, top, middle, bottom)

        @spaces.GPU(duration=20)
        def handle_translate(txt, tgt_lang):
            return ai_translate(txt, value_to_keys(language_codes, tgt_lang)[0])

        with gr.Blocks(theme=theme(), css=css()) as app:
            gr.Markdown("# AI Translator")
            gr.Markdown(
                "### An AI-based translation software for the community"
            )
            with gr.Row():
                with gr.Column():
                    txt = gr.Textbox(
                        placeholder="Some text...",
                        value="",
                        lines=4,
                        label="Input",
                        container=True,
                        max_length=2000,
                    )
                    lang = gr.Dropdown(
                        choices=unique(language_codes.values()), value="english"
                    )
                with gr.Column():
                    res = gr.Textbox(
                        label="Results",
                        container=True,
                        value="",
                        lines=6,
                        show_copy_button=True,
                    )
            with gr.Row():
                btn = gr.Button(value="Translate")
            btn.click(fn=handle_translate, inputs=[txt, lang], outputs=[res])
        app.launch(server_name="0.0.0.0", server_port=7860)
    elif proj == "animation":
        import torch
        from diffusers.utils import export_to_gif
        from PIL import Image, ImageOps

        init_pretrained_model("video", True)
        init_pretrained_model("summary")
        init_pretrained_model("translate")
        FRAMES_PER_CHUNK = 5
        fps = 20
        steps = 30

        @spaces.GPU(duration=120)
        def generate_chunk(
            chunks_path,
            txt,
            img,
            dur,
            seed,
            chunk_state,
            progress=gr.Progress(),
        ):
            txt = optimize_prompt_realism(txt)
            total_frames = int(dur * fps)
            total_chunks = math.ceil(total_frames / FRAMES_PER_CHUNK)
            current_chunk_index = chunk_state["current_chunk"]
            if current_chunk_index > total_chunks:
                raise gr.Error(
                    "All chunks have been generated. Please combine them now."
                )
            if current_chunk_index == 1:
                input_image = ImageOps.fit(
                    img, (1024, 576), Image.Resampling.LANCZOS
                )
                gr.Info("Generating first chunk using the initial image...")
            else:
                previous_chunk_path = chunk_state["chunk_paths"][-1]
                with Image.open(previous_chunk_path) as gif:
                    gif.seek(gif.n_frames - 1)
                    last_frame = gif.copy()
                input_image = last_frame
                gr.Info(
                    f"Generating chunk {current_chunk_index}/{total_chunks} using context from previous chunk..."
                )
            if input_image.mode == "RGBA":
                input_image = input_image.convert("RGB")
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)
            generator = torch.Generator(device()).manual_seed(
                int(seed) + current_chunk_index
            )
            output = MODELS["video"](
                prompt=txt,
                image=input_image,
                generator=generator,
                num_inference_steps=steps,
                num_frames=FRAMES_PER_CHUNK,
            )
            chunk_path = full_path(
                chunks_path, f"chunk_{current_chunk_index}.gif"
            )
            export_to_gif(output.frames[0], chunk_path, fps=fps)
            chunk_state["chunk_paths"].append(chunk_path)
            chunk_state["current_chunk"] += 1
            progress_text = f"Finished chunk {current_chunk_index - 1}/{total_chunks}. Ready for next chunk."
            if current_chunk_index - 1 == total_chunks:
                progress_text = (
                    "All chunks generated! You can now combine them."
                )
            return (
                chunk_path,
                chunk_state,
                gr.update(value=progress_text),
                gr.update(visible=True),
            )

        def combine_chunks(chunk_state):
            if not chunk_state["chunk_paths"]:
                raise gr.Error("No chunks to combine.")
            gr.Info(
                f"Combining {len(chunk_state['chunk_paths'])} chunks into final GIF..."
            )
            all_frames = []
            for chunk_path in chunk_state["chunk_paths"]:
                with Image.open(chunk_path) as gif:
                    for i in range(gif.n_frames):
                        gif.seek(i)
                        all_frames.append(gif.copy().convert("RGBA"))
            final_gif_path = "final_animation.gif"
            all_frames[0].save(
                final_gif_path,
                save_all=True,
                append_images=all_frames[1:],
                loop=0,
                duration=int(1000 / fps),
                optimize=True,
            )
            return (final_gif_path, gr.update(visible=False))

        def reset_state(chunks_path):
            delete(chunks_path)
            chunks_path = tmp(dir=True)
            initial_state = {"current_chunk": 1, "chunk_paths": []}
            return (
                chunks_path,
                initial_state,
                None,
                "Ready to generate the first chunk.",
                gr.update(visible=False),
                gr.update(interactive=True),
            )

        with gr.Blocks(theme=theme(), css=css()) as app:
            chunk_state = gr.State({"current_chunk": 1, "chunk_paths": []})
            gr.Markdown("# Image to Animation: Manual Chunking Method")
            gr.Markdown(
                "This app generates long animations piece-by-piece to avoid timeouts on free services. **You must click 'Generate Next Chunk' repeatedly** until all chunks are created, then click 'Combine'."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    img = gr.Image(label="Input Image", type="pil", height=420)
                    txt = gr.Textbox(
                        placeholder="Describe the scene",
                        value="",
                        lines=4,
                        label="Prompt",
                        container=True,
                    )
                    dur = gr.Slider(
                        minimum=1,
                        maximum=30,
                        value=3,
                        step=1,
                        label="Total Duration (s)",
                    )
                    with gr.Accordion("Advanced Settings", open=False):
                        seed = gr.Number(
                            label="Seed (-1 for random)", minimum=-1, value=-1
                        )
                with gr.Column(scale=1):
                    out = gr.Image(
                        label="Latest Generated Chunk / Final Animation",
                        interactive=False,
                        height=420,
                    )
                    prog = gr.Markdown("Ready to generate the first chunk.")
            with gr.Row():
                generate_button = gr.Button(
                    "Generate Next Chunk", variant="primary"
                )
                combine_button = gr.Button(
                    "Combine Chunks into Final GIF",
                    variant="stop",
                    visible=False,
                )
                reset_button = gr.Button("Start Over")
            with gr.Row(visible=False):
                chunks_path = gr.Textbox(value=tmp(dir=True))
            generate_button.click(
                fn=keep_alive(generate_chunk, 4),
                inputs=[chunks_path, txt, img, dur, seed, chunk_state],
                outputs=[out, chunk_state, prog, combine_button],
            )
            combine_button.click(
                fn=combine_chunks,
                inputs=[chunk_state],
                outputs=[out, combine_button],
            )
            reset_button.click(
                fn=reset_state,
                inputs=[chunks_path],
                outputs=[
                    chunks_path,
                    chunk_state,
                    out,
                    prog,
                    combine_button,
                    generate_button,
                ],
            )
        app.launch(server_name="0.0.0.0", server_port=7860)
    elif proj == "image":
        init_pretrained_model("translate", True)
        init_pretrained_model("summary", True)
        init_pretrained_model("image", True)
        init_upscale()

        def title(image_path, top, middle, bottom):
            return write_on_image(image_path, top, middle, bottom)

        @spaces.GPU(duration=150)
        def handle_generation(text, w, h):
            (w, h) = get_max_resolution(w, h, mega_pixels=2.5)
            text = optimize_prompt_realism(text)
            return pipe("image", prompt=text, resolution=f"{w}x{h}")

        @spaces.GPU(duration=150)
        def handle_upscaling(path):
            return upscale(path)

        with gr.Blocks(theme=theme(), css=css()) as app:
            gr.Markdown("# Text-to-Image generator")
            gr.Markdown("### Realistic. Upscalable. Multilingual.")
            with gr.Row():
                with gr.Column(scale=1):
                    width_input = gr.Slider(
                        minimum=1, maximum=16, step=1, label="Width"
                    )
                    height_input = gr.Slider(
                        minimum=1, maximum=16, step=1, label="Height"
                    )
                    data = gr.Textbox(
                        placeholder="Input data",
                        value="",
                        max_length=1000,
                        lines=4,
                        label="Prompt",
                        container=True,
                    )
                    top = gr.Textbox(
                        placeholder="Top title",
                        value="",
                        max_lines=1,
                        label="Top Title",
                    )
                    middle = gr.Textbox(
                        placeholder="Middle title",
                        value="",
                        max_lines=1,
                        label="Middle Title",
                    )
                    bottom = gr.Textbox(
                        placeholder="Bottom title",
                        value="",
                        max_lines=1,
                        label="Bottom Title",
                    )
                with gr.Column(scale=1):
                    cover = gr.Image(
                        interactive=False,
                        label="Result",
                        type="filepath",
                        show_share_button=False,
                        container=True,
                        show_download_button=True,
                    )
                    generate_image = gr.Button("Generate")
                    upscale_now = gr.Button("Upscale")
                    add_titles = gr.Button("Add title(s)")
            generate_image.click(
                fn=keep_alive(handle_generation),
                inputs=[data, width_input, height_input],
                outputs=[cover],
            )
            upscale_now.click(
                fn=keep_alive(handle_upscaling), inputs=[cover], outputs=[cover]
            )
            add_titles.click(
                fn=keep_alive(title),
                inputs=[cover, top, middle, bottom],
                outputs=[cover],
            )
        app.launch(server_name="0.0.0.0", server_port=7860)
    elif proj == "chat":
        init_pretrained_model("summary", True)
        init_pretrained_model("answer", True)
        init_pretrained_model("translate", True)
        install_ffmpeg()

        @spaces.GPU(duration=70)
        def _get_chat_response(message, history):
            return get_chat_response(message, history)

        with gr.Blocks(theme=theme(), title="AI Chatbot", css=css()) as app:
            init_chat("AI Chatbot", _get_chat_response)
        app.launch(server_name="0.0.0.0", server_port=7860)
    elif proj == "faiss":
        whl = build_faiss()

        @spaces.GPU(duration=10)
        def nop():
            return None

        with gr.Blocks() as app:
            gr.File(label="Download faiss wheel", value=whl)
        app.launch(server_name="0.0.0.0", server_port=7860)
    elif proj == "video":
        import gradio as gr
        import spaces

        custom_css = "\n        body { color: #00ff41; font-family: monospace; }\n        .gr-button.primary { background: #00f3ff; color: black; font-weight: bold; box-shadow: 0 0 10px #00f3ff; }\n        .gr-button.secondary { background: #222; color: white; border: 1px solid #444; }\n        .section-header { color: #ff003c; font-weight: bold; margin-bottom: 5px; border-bottom: 1px solid #333; padding-bottom: 5px; }\n        textarea { overflow-y: auto !important; }\n        "
        video_theme = gr.themes.Base(primary_hue="cyan", neutral_hue="slate")

        @spaces.GPU(duration=calculate_gpu_duration)
        def _generate_video_handler_gpu(*args):
            return generate_video_handler(*args)

        with gr.Blocks(
            title="AI VIDEO ARCHITECT", css=custom_css, theme=video_theme
        ) as app:
            gr.Markdown("# 🏗️ AI VIDEO ARCHITECT")
            gr.Markdown("### Advanced Composition & Layout Engine")
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Group():
                        gr.Markdown(
                            "### 📂 Media & Style",
                            elem_classes="section-header",
                        )
                        audio_in = gr.Audio(label="Audio File", type="filepath")
                        with gr.Row():
                            search_txt = gr.Textbox(
                                placeholder="Search styles...",
                                label="Search",
                                scale=2,
                            )
                            cat_filter = gr.Dropdown(
                                [
                                    "All",
                                    "Abstract",
                                    "Cyberpunk",
                                    "Retro",
                                    "Simple",
                                    "Sci-Fi",
                                ],
                                value="All",
                                label="Category Filter",
                                scale=1,
                            )
                        style_dd = gr.Dropdown(
                            choices=list(STYLES_DB.keys()),
                            value="Psychedelic Geometry",
                            label="Select Style (Base Layer)",
                        )
                        search_txt.change(
                            filter_styles, [search_txt, cat_filter], style_dd
                        )
                        cat_filter.change(
                            filter_styles, [search_txt, cat_filter], style_dd
                        )
                    with gr.Tabs():
                        with gr.TabItem("✨ Custom Element"):
                            gr.Markdown(
                                "Add a single element with full transform control"
                            )
                            with gr.Row():
                                ce_type = gr.Dropdown(
                                    [
                                        "None",
                                        "Custom Text",
                                        "Logo Image",
                                        "Spectrum Circle",
                                    ],
                                    value="None",
                                    label="Element Type",
                                )
                                ce_text = gr.Textbox(
                                    value="MY MUSIC", label="Text Content"
                                )
                            gr.Markdown("Transform")
                            with gr.Row():
                                ce_x = gr.Slider(
                                    0, 1, 0.5, 0.05, label="Position X"
                                )
                                ce_y = gr.Slider(
                                    0, 1, 0.5, 0.05, label="Position Y"
                                )
                            with gr.Row():
                                ce_scale = gr.Slider(
                                    0.1, 5.0, 1.0, 0.1, label="Scale"
                                )
                                ce_opacity = gr.Slider(
                                    0.1, 1.0, 1.0, 0.1, label="Opacity"
                                )
                        with gr.TabItem("🌐 Global"):
                            active_overlays = gr.CheckboxGroup(
                                [
                                    "Neon Border",
                                    "Progress Bar",
                                    "Audio Waveform",
                                    "Timer",
                                ],
                                label="Persistent Elements",
                            )
                            post_fx = gr.CheckboxGroup(
                                ["Vignette", "Scanlines", "Noise"],
                                label="Post-Processing FX",
                            )
                        with gr.TabItem("⚙️ Settings"):
                            res_dd = gr.Dropdown(
                                [
                                    "Square (1:1)",
                                    "Portrait (9:16)",
                                    "Landscape (16:9)",
                                ],
                                value="Square (1:1)",
                                label="Aspect Ratio",
                            )
                            fps_sl = gr.Slider(14, 20, 20, 1, label="FPS")
                            sens_sl = gr.Slider(
                                0.5, 3.0, 1.5, 1.0, label="Audio Reactivity"
                            )
                            band_dd = gr.Dropdown(
                                ["All", "Low", "Mid", "High"],
                                value="All",
                                label="Frequency Band",
                            )
                            palette_dd = gr.Dropdown(
                                [
                                    "Cyberpunk",
                                    "Sunset",
                                    "Israel",
                                    "Gold",
                                    "Matrix",
                                ],
                                value="Cyberpunk",
                                label="Color Palette",
                            )
                            bg_img = gr.Image(
                                label="Background Image (for Image Pulse)",
                                type="filepath",
                            )
                    with gr.Row():
                        preview_btn = gr.Button(
                            "📸 Generate Preview (Frame)",
                            elem_classes="secondary",
                        )
                        render_btn = gr.Button(
                            "🚀 Render Full Video", elem_classes="primary"
                        )
                with gr.Column(scale=1):
                    ce_logo = gr.Image(
                        label="Logo File", type="filepath", height=300
                    )
                    preview_out = gr.Image(label="Preview Snapshot", height=300)
                    out_vid = gr.Video(label="Final Output", height=400)
            with gr.Row():
                out_stat = gr.Textbox(label="System Log")
            preview_btn.click(
                fn=generate_preview_handler,
                inputs=[
                    audio_in,
                    bg_img,
                    style_dd,
                    res_dd,
                    sens_sl,
                    band_dd,
                    palette_dd,
                    active_overlays,
                    post_fx,
                    ce_type,
                    ce_x,
                    ce_y,
                    ce_scale,
                    ce_opacity,
                    ce_text,
                    ce_logo,
                ],
                outputs=[preview_out, out_stat],
            )
            render_btn.click(
                fn=_generate_video_handler_gpu,
                inputs=[
                    audio_in,
                    bg_img,
                    style_dd,
                    res_dd,
                    fps_sl,
                    sens_sl,
                    band_dd,
                    palette_dd,
                    active_overlays,
                    post_fx,
                    ce_type,
                    ce_x,
                    ce_y,
                    ce_scale,
                    ce_opacity,
                    ce_text,
                    ce_logo,
                ],
                outputs=[out_vid, out_stat],
            )
        app.launch(server_name="0.0.0.0", server_port=7860)
    elif proj == "audio":
        pip_install("git+https://github.com/YaronKoresh/audio-studio-pro.git")
        run("audio-studio-pro")
    elif proj == "train":
        pip_install("git+https://github.com/YaronKoresh/teachless.git")
        run("teachless")
    else:
        catch(f"Error: No project called '{proj}' !")
