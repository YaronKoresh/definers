import gc
import math
import random
import re

from definers._audio import value_to_keys
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
    optimize_prompt_realism,
    pipe,
)
from definers._system import (
    catch,
    cores,
    delete,
    exist,
    full_path,
    log,
    tmp,
    unique,
)
from definers._text import ai_translate, language, simple_text
from definers._video_gui import (
    draw_star_of_david,
    filter_styles,
    generate_video_handler,
)

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
        title=title,
        save_history=True,
        show_progress="hidden",
        concurrency_limit=None,
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


def _gui_translate():
    import gradio as gr

    init_pretrained_model("translate", True)

    def title(image_path, top, middle, bottom):
        return write_on_image(image_path, top, middle, bottom)

    def handle_translate(txt, tgt_lang):
        return ai_translate(txt, value_to_keys(language_codes, tgt_lang)[0])

    with gr.Blocks() as app:
        gr.Markdown("# AI Translator")
        gr.Markdown("### An AI-based translation software for the community")
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
    app.launch(
        server_name="0.0.0.0", theme=theme(), css=css(), server_port=7860
    )


def _gui_animation():
    import gradio as gr
    import torch
    from diffusers.utils import export_to_gif
    from PIL import Image, ImageOps

    init_pretrained_model("video", True)
    init_pretrained_model("summary")
    init_pretrained_model("translate")
    FRAMES_PER_CHUNK = 5
    fps = 20
    steps = 30

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
        chunk_path = full_path(chunks_path, f"chunk_{current_chunk_index}.gif")
        export_to_gif(output.frames[0], chunk_path, fps=fps)
        chunk_state["chunk_paths"].append(chunk_path)
        chunk_state["current_chunk"] += 1
        progress_text = f"Finished chunk {current_chunk_index - 1}/{total_chunks}. Ready for next chunk."
        if current_chunk_index - 1 == total_chunks:
            progress_text = "All chunks generated! You can now combine them."
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

    with gr.Blocks() as app:
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
            fn=generate_chunk,
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
    app.launch(
        server_name="0.0.0.0", theme=theme(), css=css(), server_port=7860
    )


def _gui_image():
    import gradio as gr

    init_pretrained_model("translate", True)
    init_pretrained_model("summary", True)
    init_pretrained_model("image", True)
    init_upscale()

    def title(image_path, top, middle, bottom):
        return write_on_image(image_path, top, middle, bottom)

    def handle_generation(text, w, h):
        (w, h) = get_max_resolution(w, h, mega_pixels=2.5)
        text = optimize_prompt_realism(text)
        return pipe("image", prompt=text, resolution=f"{w}x{h}")

    def handle_upscaling(path):
        return upscale(path)

    with gr.Blocks() as app:
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
            fn=handle_generation,
            inputs=[data, width_input, height_input],
            outputs=[cover],
        )
        upscale_now.click(fn=handle_upscaling, inputs=[cover], outputs=[cover])
        add_titles.click(
            fn=title,
            inputs=[cover, top, middle, bottom],
            outputs=[cover],
        )
    app.launch(
        server_name="0.0.0.0", theme=theme(), css=css(), server_port=7860
    )


def _gui_chat():
    import gradio as gr

    from definers import _system

    init_pretrained_model("summary", True)
    init_pretrained_model("answer", True)
    init_pretrained_model("translate", True)
    _system.install_ffmpeg()

    def _get_chat_response(message, history):
        return get_chat_response(message, history)

    with gr.Blocks(title="AI Chatbot") as app:
        init_chat("AI Chatbot", _get_chat_response)
    app.launch(
        server_name="0.0.0.0", theme=theme(), css=css(), server_port=7860
    )


def _gui_faiss():
    import gradio as gr

    whl = build_faiss()

    def nop():
        return None

    with gr.Blocks() as app:
        gr.File(label="Download faiss wheel", value=whl)
    app.launch(
        server_name="0.0.0.0", theme=theme(), css=css(), server_port=7860
    )


def _gui_video():
    import gradio as gr

    custom_css = "\n        body { color: #00ff41; font-family: monospace; }\n        .gr-button.primary { background: #00f3ff; color: black; font-weight: bold; box-shadow: 0 0 10px #00f3ff; }\n        .gr-button.secondary { background: #222; color: white; border: 1px solid #444; }\n        .section-header { color: #ff003c; font-weight: bold; margin-bottom: 5px; border-bottom: 1px solid #333; padding-bottom: 5px; }\n        textarea { overflow-y: auto !important; }\n        "
    video_theme = gr.themes.Base(primary_hue="cyan", neutral_hue="slate")

    with gr.Blocks(
        title="AI VIDEO ARCHITECT", css=custom_css, theme=video_theme
    ) as app:
        gr.Markdown("# 🏗️ AI VIDEO ARCHITECT")
        with gr.Tabs():
            with gr.TabItem("Composer"):
                gr.Markdown("### Advanced Composition & Layout Engine")
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown(
                                "### 📂 Media & Style",
                                elem_classes="section-header",
                            )
                            audio_in = gr.Audio(
                                label="Audio File", type="filepath"
                            )
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
                                filter_styles,
                                [search_txt, cat_filter],
                                style_dd,
                            )
                            cat_filter.change(
                                filter_styles,
                                [search_txt, cat_filter],
                                style_dd,
                            )

                            with gr.Row():
                                with gr.Column():
                                    image_in = gr.Image(
                                        label="Background Image",
                                        type="filepath",
                                    )
                                    resolution = gr.Dropdown(
                                        [
                                            "Square (1:1)",
                                            "Portrait (9:16)",
                                            "Landscape (16:9)",
                                        ],
                                        value="Landscape (16:9)",
                                        label="Resolution",
                                    )
                                    fps = gr.Slider(
                                        minimum=1,
                                        maximum=60,
                                        value=20,
                                        label="FPS",
                                    )
                                    sensitivity = gr.Slider(
                                        minimum=0.1,
                                        maximum=5,
                                        value=1,
                                        label="Sensitivity",
                                    )
                                    reactivity = gr.Dropdown(
                                        ["Low", "Mid", "High", "Full"],
                                        value="Full",
                                        label="Reactivity Band",
                                    )
                                with gr.Column():
                                    palette = gr.Dropdown(
                                        choices=list(STYLES_DB.keys()),
                                        value=list(STYLES_DB.keys())[0],
                                        label="Color Palette",
                                    )
                                    overlays = gr.CheckboxGroup(
                                        [
                                            "Neon Border",
                                            "Progress Bar",
                                            "Audio Waveform",
                                            "Timer",
                                        ],
                                        label="Overlays",
                                    )
                                    effects = gr.CheckboxGroup(
                                        ["Vignette", "Scanlines", "Noise"],
                                        label="Post Effects",
                                    )
                                    custom_elem = gr.Dropdown(
                                        [
                                            "None",
                                            "Custom Text",
                                            "Logo Image",
                                            "Spectrum Circle",
                                        ],
                                        label="Custom Element",
                                    )
                            ce_x = gr.Slider(0, 1, 0.5, label="Element X")
                            ce_y = gr.Slider(0, 1, 0.5, label="Element Y")
                            ce_scale = gr.Slider(
                                0.1, 5, 1, label="Element Scale"
                            )
                            ce_opacity = gr.Slider(
                                0, 1, 1, label="Element Opacity"
                            )
                            ce_text = gr.Textbox(label="Custom Text")
                            ce_logo = gr.File(label="Custom Logo")

                            btn = gr.Button("Generate Video")
                            out_vid = gr.Video(label="Output Video")
                            btn.click(
                                fn=generate_video_handler,
                                inputs=[
                                    audio_in,
                                    image_in,
                                    style_dd,
                                    resolution,
                                    fps,
                                    sensitivity,
                                    reactivity,
                                    palette,
                                    overlays,
                                    effects,
                                    custom_elem,
                                    ce_x,
                                    ce_y,
                                    ce_scale,
                                    ce_opacity,
                                    ce_text,
                                    ce_logo,
                                ],
                                outputs=[out_vid],
                            )
            with gr.TabItem("Lyric Video"):
                with gr.Group():
                    gr.Markdown("### 📝 Lyric Video Creator")
                    lv_audio = gr.Audio(label="Audio File", type="filepath")
                    lv_bg = gr.Image(label="Background Image", type="filepath")
                    lv_lyrics = gr.Textbox(label="Lyrics", lines=6)
                    lv_pos = gr.Dropdown(
                        ["top", "center", "bottom"],
                        value="bottom",
                        label="Text Position",
                    )
                    lv_max = gr.Number(value=640, label="Max Dimension")
                    lv_font = gr.Number(value=70, label="Font Size")
                    lv_color = gr.Textbox(value="white", label="Text Color")
                    lv_stroke = gr.Textbox(value="black", label="Stroke Color")
                    lv_width = gr.Slider(0, 10, value=2, label="Stroke Width")
                    lv_fade = gr.Slider(
                        0.0, 5.0, value=0.5, label="Fade Duration"
                    )
                    lv_btn = gr.Button("Make Lyric Video")
                    lv_out = gr.Video(label="Lyric Output")
                    lv_btn.click(
                        fn=lyric_video,
                        inputs=[
                            lv_audio,
                            lv_bg,
                            lv_lyrics,
                            lv_pos,
                            lv_max,
                            lv_font,
                            lv_color,
                            lv_stroke,
                            lv_width,
                            lv_fade,
                        ],
                        outputs=[lv_out],
                    )
            with gr.TabItem("Visualizer"):
                with gr.Group():
                    gr.Markdown("### 🎶 Music Visualizer")
                    mv_audio = gr.Audio(label="Audio File", type="filepath")
                    mv_width = gr.Number(value=1920, label="Width")
                    mv_height = gr.Number(value=1080, label="Height")
                    mv_fps = gr.Slider(
                        minimum=1, maximum=60, value=30, label="FPS"
                    )
                    mv_btn = gr.Button("Generate Visualizer")
                    mv_out = gr.Video(label="Visualizer Output")
                    mv_btn.click(
                        fn=music_video,
                        inputs=[mv_audio, mv_width, mv_height, mv_fps],
                        outputs=[mv_out],
                    )
    app.launch(server_name="0.0.0.0", server_port=7860)


def _gui_audio():
    import gradio as gr

    from definers import (
        analyze_audio_features,
        answer,
        audio_to_midi,
        beat_visualizer,
        change_audio_speed,
        convert_vocal_rvc,
        create_share_links,
        create_spectrum_visualization,
        cwd,
        device,
        dj_mix,
        enhance_audio,
        extend_audio,
        generate_music,
        generate_voice,
        get_audio_feedback,
        google_drive_download,
        identify_instruments,
        init_pretrained_model,
        install_audio_effects,
        install_ffmpeg,
        language_codes,
        midi_to_audio,
        pitch_shift_vocals,
        random_string,
        save_temp_text as save_text_to_file,
        separate_stems,
        set_system_message,
        stem_mixer,
        stretch_audio,
        train_model_rvc,
        transcribe_audio,
    )

    install_audio_effects()
    install_ffmpeg()

    init_stable_whisper()
    init_pretrained_model("tts")

    svc_installed = False
    with cwd():
        if exist("./assets"):
            svc_installed = True

    if not svc_installed:
        init_pretrained_model("svc")

    init_pretrained_model("speech-recognition")
    init_pretrained_model("audio-classification")
    init_pretrained_model("music")
    init_pretrained_model("summary")
    init_pretrained_model("answer")
    init_pretrained_model("translate")

    set_system_message(
        name="Fazzer",
        role="the official chat assistant for the 'Audio Studio Pro' application",
        rules=[
            "guide users with the application usage",
            "explain the purpose of each tool in the application",
            "provide simple, step-by-step instructions on how to use the features based on their UI",
        ],
        data=[
            "The name of the software you help with, is Audio Studio Pro",
            "The name of your creator, is Yaron Koresh",
            "The origin country of your creator, is Israel",
            "Audio Studio Pro is licensed under the Open source MIT license",
            "The official link to Audio Studio Pro original source code, is https://github.com/YaronKoresh/audio-studio-pro",
            "The main AI models that Audio Studio Pro depends on, are: openai/whisper-large-v3, MIT/ast-finetuned-audioset-10-10-0.4593, and facebook/musicgen-small",
            "The supported output formats, are: MP3 (320k), FLAC (16-bit), and WAV (16-bit PCM)",
            "The export process is by clicking on the small down-arrow download button",
            """The complete list of the software's features with usage instructions:
    an audio enhancement tool to auto-tune and master your track - upload your track, choose an output format, and click 'Enhance Audio';
    audio to midi converter - upload an audio file and click 'Convert to MIDI';
    midi to audio converter - upload a MIDI file and click 'Convert to Audio';
    an audio extender that uses AI to seamlessly continue a piece of music - upload your audio, use the 'Extend Duration' slider to choose how many seconds to add, and click 'Extend Audio';
    a stem mixer that mixes individual instrument tracks (stems) together - upload multiple audio files (e.g., drums.wav, bass.wav). The tool automatically beatmatches them to the first track and mixes them;
    a track feedbacks generator that provides an analysis and advice on your mix - upload your track and click 'Get Feedback' for written analysis on its dynamics, stereo width, and frequency balance;
    an instrument identifier from an audio file - upload an audio file and click 'Identify Instruments';
    a video generator which creates a simple and abstract music visualizer - upload an audio file and click 'Generate Video' to create a video with a pulsing circle that reacts to the music;
    a speed & pitch changer which changes the playback speed of a track - upload audio, use the 'Speed Factor' slider (e.g., 1.5x is faster), and check 'Preserve Pitch' for a more natural sound;
    a stems separator which splits a song into vocals and instrumental - upload a full song and choose either 'Acapella (Vocals Only)' or 'Karaoke (Instrumental Only)';
    a vocal pitch shifter which changes the pitch of only the vocals in a song - upload a song and use the 'Vocal Pitch Shift' slider to raise or lower the vocal pitch in semitones;
    a voice cloning and conversion tool for voice manipulation, preserving the melody - upload your training audio files, click 'Train' to create a voice model, then use the 'Convert' tab to apply that voice to a new audio input;
    a dj tool which automatically mixes multiple songs together - upload two or more tracks. Choose 'Beatmatched Crossfade' for a smooth, tempo-synced mix and adjust the 'Transition Duration';
    an AI music generator which creates original music from a text description - write a description of the music you want (e.g., 'upbeat synthwave'), set the duration, and click 'Generate Music';
    an AI voice generator which clones a voice to say anything you type - upload a clean 5-15 second 'Reference Voice' sample, type the 'Text to Speak', and click 'Generate Voice';
    a bpm & key analysis tools which detects a track's musical key and tempo - upload your audio and click 'Analyze Audio';
    a speech-to-text tool which transcribes speech from an audio file into text - upload an audio file with speech, select the language, and click 'Transcribe Audio'.
    a spectrum analyzer which creates a visual graph (spectrogram) of an audio's frequencies - upload an audio file and click 'Generate Spectrum'.
    a beat visualizer which creates a video where an image pulses to the music's beat - upload an image and an audio file. Adjust the 'Beat Intensity' slider to control how much the image reacts.
    a lyric video creation tool which creates a simple lyric video - upload a song and a background image/video. Then, paste your lyrics into the text box, with each line representing a new phrase on screen.
    a support chat (that's you!) which answer questions like 'What is Stem Mixing?' or 'How do I use the Vocal Pitch Shifter?' based on his knowledge-base.""",
        ],
        formal=True,
        creative=False,
    )

    def handle_training(experiment, inp, lvl):
        with cwd():
            return train_model_rvc(experiment, inp, lvl), lvl + 1

    format_choices = ["MP3", "WAV", "FLAC"]
    language_choices = sorted(list(set(language_codes.values())))

    with gr.Blocks(title="Audio Studio Pro") as app:
        gr.HTML(
            """<div id="header"><h1>Audio Studio Pro</h1><p>Your complete suite for professional audio production and AI-powered sound creation.</p></div>"""
        )

        tool_map = {
            "Audio Enhancer": "enhancer",
            "MIDI Tools": "midi_tools",
            "Audio Extender": "audio_extender",
            "Stem Mixer": "stem_mixer",
            "Track Feedback": "feedback",
            "Instrument ID": "instrument_id",
            "Music Clip Generation": "video_gen",
            "Speed & Pitch": "speed",
            "Stem Separation": "stem",
            "Vocal Pitch Shifter": "vps",
            "Voice Lab": "voice_lab",
            "DJ AutoMix": "dj",
            "Music Gen": "music_gen",
            "Voice Gen": "voice_gen",
            "Analysis": "analysis",
            "Speech-to-Text": "stt",
            "Spectrum": "spectrum",
            "Beat Visualizer": "beat_vis",
            "Lyric Video": "lyric_vid",
            "Support Chat": "chatbot",
        }

        with gr.Row(elem_id="nav-dropdown-wrapper"):
            nav_dropdown = gr.Dropdown(
                choices=list(tool_map.keys()),
                value="Audio Enhancer",
                label="Select a Tool",
                elem_id="nav-dropdown",
            )

        with gr.Row(elem_id="main-row"):
            with gr.Column(scale=1, elem_id="main-content"):
                with gr.Group(
                    visible=True, elem_classes="tool-container"
                ) as view_enhancer:
                    gr.Markdown("## Audio Enhancer")
                    with gr.Row():
                        with gr.Column():
                            enhancer_input = gr.Audio(
                                label="Upload Track", type="filepath"
                            )
                            with gr.Row():
                                enhancer_btn = gr.Button(
                                    "Enhance Audio", variant="primary"
                                )
                                clear_enhancer_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(visible=False) as enhancer_output_box:
                                enhancer_output = gr.Audio(
                                    label="Enhancer Output",
                                    interactive=False,
                                    show_download_button=True,
                                )
                                enhancer_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_midi_tools:
                    gr.Markdown("## MIDI Tools")
                    with gr.Tabs():
                        with gr.TabItem("Audio to MIDI"):
                            with gr.Row():
                                with gr.Column():
                                    a2m_input = gr.Audio(
                                        label="Upload Audio", type="filepath"
                                    )
                                    with gr.Row():
                                        a2m_btn = gr.Button(
                                            "Convert to MIDI", variant="primary"
                                        )
                                        clear_a2m_btn = gr.Button(
                                            "Clear", variant="secondary"
                                        )
                                with gr.Column():
                                    with gr.Group(
                                        visible=False
                                    ) as a2m_output_box:
                                        a2m_output = gr.File(
                                            label="Output MIDI",
                                            interactive=False,
                                        )
                                        a2m_share_links = gr.Markdown()
                        with gr.TabItem("MIDI to Audio"):
                            with gr.Row():
                                with gr.Column():
                                    m2a_input = gr.File(
                                        label="Upload MIDI",
                                        file_types=[".mid", ".midi"],
                                    )
                                    m2a_format = gr.Radio(
                                        format_choices,
                                        label="Output Format",
                                        value=format_choices[0],
                                    )
                                    with gr.Row():
                                        m2a_btn = gr.Button(
                                            "Convert to Audio",
                                            variant="primary",
                                        )
                                        clear_m2a_btn = gr.Button(
                                            "Clear", variant="secondary"
                                        )
                                with gr.Column():
                                    with gr.Group(
                                        visible=False
                                    ) as m2a_output_box:
                                        m2a_output = gr.Audio(
                                            label="Output Audio",
                                            interactive=False,
                                            show_download_button=True,
                                        )
                                        m2a_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_audio_extender:
                    gr.Markdown("## Audio Extender")
                    with gr.Row():
                        with gr.Column():
                            extender_input = gr.Audio(
                                label="Upload Audio to Extend", type="filepath"
                            )
                            extender_duration = gr.Slider(
                                5,
                                60,
                                15,
                                step=1,
                                label="Extend Duration (seconds)",
                            )
                            extender_format = gr.Radio(
                                format_choices,
                                label="Output Format",
                                value=format_choices[0],
                            )
                            with gr.Row():
                                extender_btn = gr.Button(
                                    "Extend Audio", variant="primary"
                                )
                                clear_extender_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(visible=False) as extender_output_box:
                                extender_output = gr.Audio(
                                    label="Extended Audio",
                                    interactive=False,
                                    show_download_button=True,
                                )
                                extender_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_stem_mixer:
                    gr.Markdown("## Stem Mixer")
                    with gr.Row():
                        with gr.Column():
                            stem_mixer_files = gr.File(
                                label="Upload Stems (Drums, Bass, Vocals, etc.)",
                                file_count="multiple",
                                type="filepath",
                            )
                            stem_mixer_format = gr.Radio(
                                format_choices,
                                label="Output Format",
                                value=format_choices[0],
                            )
                            with gr.Row():
                                stem_mixer_btn = gr.Button(
                                    "Mix Stems", variant="primary"
                                )
                                clear_stem_mixer_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(
                                visible=False
                            ) as stem_mixer_output_box:
                                stem_mixer_output = gr.Audio(
                                    label="Mixed Track",
                                    interactive=False,
                                    show_download_button=True,
                                )
                                stem_mixer_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_feedback:
                    gr.Markdown("## AI Track Feedback")
                    with gr.Row():
                        with gr.Column():
                            feedback_input = gr.Audio(
                                label="Upload Your Track", type="filepath"
                            )
                            with gr.Row():
                                feedback_btn = gr.Button(
                                    "Get Feedback", variant="primary"
                                )
                                clear_feedback_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            feedback_output = gr.Markdown(label="Feedback")
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_instrument_id:
                    gr.Markdown("## Instrument Identification")
                    with gr.Row():
                        with gr.Column():
                            instrument_id_input = gr.Audio(
                                label="Upload Audio", type="filepath"
                            )
                            with gr.Row():
                                instrument_id_btn = gr.Button(
                                    "Identify Instruments", variant="primary"
                                )
                                clear_instrument_id_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            instrument_id_output = gr.Markdown(
                                label="Detected Instruments"
                            )
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_video_gen:
                    gr.Markdown("## AI Music Clip Generation")
                    with gr.Row():
                        with gr.Column():
                            video_gen_audio = gr.Audio(
                                label="Upload Audio", type="filepath"
                            )
                            with gr.Row():
                                video_gen_btn = gr.Button(
                                    "Generate Video", variant="primary"
                                )
                                clear_video_gen_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(
                                visible=False
                            ) as video_gen_output_box:
                                video_gen_output = gr.Video(
                                    label="Generated Clip",
                                    interactive=False,
                                    show_download_button=True,
                                )
                                video_gen_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_speed:
                    gr.Markdown("## Speed & Pitch")
                    with gr.Row():
                        with gr.Column():
                            speed_input = gr.Audio(
                                label="Upload Track", type="filepath"
                            )
                            speed_factor = gr.Slider(
                                minimum=0.5,
                                maximum=2.0,
                                value=1.0,
                                step=0.01,
                                label="Speed Factor",
                            )
                            preserve_pitch = gr.Checkbox(
                                label="Preserve Pitch (higher quality)",
                                value=True,
                            )
                            speed_format = gr.Radio(
                                format_choices,
                                label="Output Format",
                                value=format_choices[0],
                            )
                            with gr.Row():
                                speed_btn = gr.Button(
                                    "Change Speed", variant="primary"
                                )
                                clear_speed_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(visible=False) as speed_output_box:
                                speed_output = gr.Audio(
                                    label="Modified Audio",
                                    interactive=False,
                                    show_download_button=True,
                                )
                                speed_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_stem:
                    gr.Markdown("## Stem Separation")
                    with gr.Row():
                        with gr.Column():
                            stem_input = gr.Audio(
                                label="Upload Full Mix", type="filepath"
                            )
                            stem_type = gr.Radio(
                                [
                                    "Acapella (Vocals Only)",
                                    "Karaoke (Instrumental Only)",
                                ],
                                label="Separation Type",
                                value="Acapella (Vocals Only)",
                            )
                            stem_format = gr.Radio(
                                format_choices,
                                label="Output Format",
                                value=format_choices[0],
                            )
                            with gr.Row():
                                stem_btn = gr.Button(
                                    "Separate Stems", variant="primary"
                                )
                                clear_stem_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(visible=False) as stem_output_box:
                                stem_output = gr.Audio(
                                    label="Separated Track",
                                    interactive=False,
                                    show_download_button=True,
                                )
                                stem_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_vps:
                    gr.Markdown("## Vocal Pitch Shifter")
                    with gr.Row():
                        with gr.Column():
                            vps_input = gr.Audio(
                                label="Upload Full Song", type="filepath"
                            )
                            vps_pitch = gr.Slider(
                                -12,
                                12,
                                0,
                                step=1,
                                label="Vocal Pitch Shift (Semitones)",
                            )
                            vps_format = gr.Radio(
                                format_choices,
                                label="Output Format",
                                value=format_choices[0],
                            )
                            with gr.Row():
                                vps_btn = gr.Button(
                                    "Shift Vocal Pitch", variant="primary"
                                )
                                clear_vps_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(visible=False) as vps_output_box:
                                vps_output = gr.Audio(
                                    label="Pitch Shifted Song",
                                    interactive=False,
                                    show_download_button=True,
                                )
                                vps_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_voice_lab:
                    gr.Markdown("## 🔬 Voice Lab")
                    with gr.Row(visible=False):
                        experiment = gr.Textbox(value=random_string())
                    with gr.Row():
                        inp = gr.File(label="Input", type="filepath")
                        outp = gr.File(
                            label="Output",
                            type="filepath",
                            file_count="multiple",
                        )
                    with gr.Row(visible=False):
                        lvl = gr.Number(
                            label="(re-)training step",
                            value=1,
                            minimum=1,
                            step=1,
                        )
                    with gr.Row():
                        but1 = gr.Button("Train", variant="primary")
                        but1.click(
                            fn=handle_training,
                            inputs=[experiment, inp, lvl],
                            outputs=[outp, lvl],
                        )
                        but2 = gr.Button("Convert", variant="primary")
                        but2.click(
                            fn=convert_vocal_rvc,
                            inputs=[experiment, inp],
                            outputs=[outp],
                        )
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_dj:
                    gr.Markdown("## DJ AutoMix")
                    with gr.Row():
                        with gr.Column():
                            dj_files = gr.File(
                                label="Upload Audio Tracks",
                                file_count="multiple",
                                type="filepath",
                                allow_reordering=True,
                            )
                            dj_mix_type = gr.Radio(
                                ["Simple Crossfade", "Beatmatched Crossfade"],
                                label="Mix Type",
                                value="Beatmatched Crossfade",
                            )
                            dj_target_bpm = gr.Number(
                                label="Target BPM (Optional)"
                            )
                            dj_transition = gr.Slider(
                                1,
                                15,
                                5,
                                step=1,
                                label="Transition Duration (seconds)",
                            )
                            dj_format = gr.Radio(
                                format_choices,
                                label="Output Format",
                                value=format_choices[0],
                            )
                            with gr.Row():
                                dj_btn = gr.Button(
                                    "Create DJ Mix", variant="primary"
                                )
                                clear_dj_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(visible=False) as dj_output_box:
                                dj_output = gr.Audio(
                                    label="Final DJ Mix",
                                    interactive=False,
                                    show_download_button=True,
                                )
                                dj_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_music_gen:
                    gr.Markdown("## AI Music Generation")
                    if device() == "cpu":
                        gr.Markdown(
                            "<p style='color:orange;text-align:center;'>Running on a CPU. Music generation will be very slow.</p>"
                        )
                    with gr.Row():
                        with gr.Column():
                            gen_prompt = gr.Textbox(
                                lines=4,
                                label="Music Prompt",
                                placeholder="e.g., '80s synthwave, retro, upbeat'",
                            )
                            gen_duration = gr.Slider(
                                5, 30, 10, step=1, label="Duration (seconds)"
                            )
                            gen_format = gr.Radio(
                                format_choices,
                                label="Output Format",
                                value=format_choices[0],
                            )
                            with gr.Row():
                                gen_btn = gr.Button(
                                    "Generate Music",
                                    variant="primary",
                                    interactive=True,
                                )
                                clear_gen_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(visible=False) as gen_output_box:
                                gen_output = gr.Audio(
                                    label="Generated Music",
                                    interactive=False,
                                    show_download_button=True,
                                )
                                gen_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_voice_gen:
                    gr.Markdown("## AI Voice Generation")
                    with gr.Row():
                        with gr.Column():
                            vg_ref = gr.Audio(
                                label="Reference Voice (Clear, 5-15s)",
                                type="filepath",
                            )
                            vg_text = gr.Textbox(
                                lines=4,
                                label="Text to Speak",
                                placeholder="Enter the text you want the generated voice to say...",
                            )
                            vg_format = gr.Radio(
                                format_choices,
                                label="Output Format",
                                value=format_choices[0],
                            )
                            with gr.Row():
                                vg_btn = gr.Button(
                                    "Generate Voice",
                                    variant="primary",
                                    interactive=True,
                                )
                                clear_vg_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            with gr.Group(visible=False) as vg_output_box:
                                vg_output = gr.Audio(
                                    label="Generated Voice Audio",
                                    interactive=False,
                                    show_download_button=True,
                                )
                                vg_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_analysis:
                    gr.Markdown("## BPM & Key Analysis")
                    with gr.Row():
                        with gr.Column(scale=1):
                            analysis_input = gr.Audio(
                                label="Upload Audio", type="filepath"
                            )
                            with gr.Row():
                                analysis_btn = gr.Button(
                                    "Analyze Audio", variant="primary"
                                )
                                clear_analysis_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column(scale=1):
                            analysis_bpm_key_output = gr.Textbox(
                                label="Detected Key & BPM", interactive=False
                            )
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_stt:
                    gr.Markdown("## Speech-to-Text")
                    with gr.Row():
                        with gr.Column():
                            stt_input = gr.Audio(
                                label="Upload Speech Audio", type="filepath"
                            )
                            stt_language = gr.Dropdown(
                                language_choices,
                                label="Language",
                                value="english",
                            )
                            with gr.Row():
                                stt_btn = gr.Button(
                                    "Transcribe Audio",
                                    variant="primary",
                                    interactive=True,
                                )
                                clear_stt_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                        with gr.Column():
                            stt_output = gr.Textbox(
                                label="Transcription Result",
                                interactive=False,
                                lines=10,
                            )
                            stt_file_output = gr.File(
                                label="Download Transcript",
                                interactive=False,
                                visible=False,
                            )
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_spectrum:
                    gr.Markdown("## Spectrum Analyzer")
                    spec_input = gr.Audio(label="Upload Audio", type="filepath")
                    with gr.Row():
                        spec_btn = gr.Button(
                            "Generate Spectrum", variant="primary"
                        )
                        clear_spec_btn = gr.Button("Clear", variant="secondary")
                    spec_output = gr.Image(
                        label="Spectrum Plot", interactive=False
                    )
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_beat_vis:
                    gr.Markdown("## Beat Visualizer")
                    with gr.Row():
                        with gr.Column():
                            vis_image_input = gr.Image(
                                label="Upload Image", type="filepath"
                            )
                            vis_audio_input = gr.Audio(
                                label="Upload Audio", type="filepath"
                            )
                        with gr.Column():
                            vis_effect = gr.Radio(
                                [
                                    "None",
                                    "Blur",
                                    "Sharpen",
                                    "Contour",
                                    "Emboss",
                                ],
                                label="Image Effect",
                                value="None",
                            )
                            vis_animation = gr.Radio(
                                ["None", "Zoom In", "Zoom Out"],
                                label="Animation Style",
                                value="None",
                            )
                            vis_intensity = gr.Slider(
                                1.05,
                                1.5,
                                1.15,
                                step=0.01,
                                label="Beat Intensity",
                            )
                            with gr.Row():
                                vis_btn = gr.Button(
                                    "Create Beat Visualizer", variant="primary"
                                )
                                clear_vis_btn = gr.Button(
                                    "Clear", variant="secondary"
                                )
                    with gr.Group(visible=False) as vis_output_box:
                        vis_output = gr.Video(
                            label="Visualizer Output", show_download_button=True
                        )
                        vis_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_lyric_vid:
                    gr.Markdown("## Lyric Video Creator")
                    with gr.Row():
                        with gr.Column():
                            lyric_audio = gr.Audio(
                                label="Upload Song", type="filepath"
                            )
                            lyric_bg = gr.File(
                                label="Upload Background (Image or Video)",
                                type="filepath",
                            )
                            lyric_position = gr.Radio(
                                ["center", "bottom"],
                                label="Text Position",
                                value="bottom",
                            )
                        with gr.Column():
                            lyric_text = gr.Textbox(
                                label="Lyrics",
                                lines=15,
                                placeholder="Enter lyrics here, one line per phrase...",
                            )
                            load_transcript_btn = gr.Button(
                                "Get Lyrics from Audio (via Speech-to-Text)"
                            )
                            lyric_language = gr.Dropdown(
                                language_choices,
                                label="Lyrics language (for Speech-to-Text)",
                                value="english",
                            )
                    with gr.Row():
                        lyric_btn = gr.Button(
                            "Create Lyric Video", variant="primary"
                        )
                        clear_lyric_btn = gr.Button(
                            "Clear", variant="secondary"
                        )
                    with gr.Group(visible=False) as lyric_output_box:
                        lyric_output = gr.Video(
                            label="Lyric Video Output",
                            show_download_button=True,
                        )
                        lyric_share_links = gr.Markdown()
                with gr.Group(
                    visible=False, elem_classes="tool-container"
                ) as view_chatbot:
                    init_chat("Audio Studio Pro AI support", get_chat_response)

        views = {
            "enhancer": view_enhancer,
            "midi_tools": view_midi_tools,
            "audio_extender": view_audio_extender,
            "stem_mixer": view_stem_mixer,
            "feedback": view_feedback,
            "instrument_id": view_instrument_id,
            "video_gen": view_video_gen,
            "speed": view_speed,
            "stem": view_stem,
            "vps": view_vps,
            "voice_lab": view_voice_lab,
            "dj": view_dj,
            "music_gen": view_music_gen,
            "voice_gen": view_voice_gen,
            "analysis": view_analysis,
            "stt": view_stt,
            "spectrum": view_spectrum,
            "beat_vis": view_beat_vis,
            "lyric_vid": view_lyric_vid,
            "chatbot": view_chatbot,
        }

        def switch_view(selected_tool_name):
            selected_view_key = tool_map[selected_tool_name]
            return {
                view: gr.update(visible=(key == selected_view_key))
                for key, view in views.items()
            }

        nav_dropdown.change(
            fn=switch_view, inputs=nav_dropdown, outputs=list(views.values())
        )

        def create_ui_handler(
            btn, out_el, out_box, out_share, logic_func, *inputs
        ):
            def ui_handler_generator(*args):

                try:
                    result = logic_func(*args)
                    share_text = (
                        "Check out this creation from Audio Studio Pro! 🎶"
                    )
                    share_html = create_share_links(
                        "yaron123", "audio-studio-pro", result, share_text
                    )
                    return (
                        gr.update(value=btn.value, interactive=True),
                        gr.update(visible=True),
                        gr.update(value=result),
                        gr.update(value=share_html),
                    )
                except Exception:
                    return (
                        gr.update(value=btn.value, interactive=True),
                        gr.update(visible=False),
                        gr.update(value=None),
                        gr.update(value=""),
                    )

            btn.click(
                ui_handler_generator,
                inputs=inputs,
                outputs=[btn, out_box, out_el, out_share],
            )

        create_ui_handler(
            enhancer_btn,
            enhancer_output,
            enhancer_output_box,
            enhancer_share_links,
            enhance_audio,
            enhancer_input,
        )
        create_ui_handler(
            a2m_btn,
            a2m_output,
            a2m_output_box,
            a2m_share_links,
            audio_to_midi,
            a2m_input,
        )
        create_ui_handler(
            m2a_btn,
            m2a_output,
            m2a_output_box,
            m2a_share_links,
            midi_to_audio,
            m2a_input,
            m2a_format,
        )
        create_ui_handler(
            extender_btn,
            extender_output,
            extender_output_box,
            extender_share_links,
            extend_audio,
            extender_input,
            extender_duration,
            extender_format,
        )
        create_ui_handler(
            stem_mixer_btn,
            stem_mixer_output,
            stem_mixer_output_box,
            stem_mixer_share_links,
            stem_mixer,
            stem_mixer_files,
            stem_mixer_format,
        )
        create_ui_handler(
            video_gen_btn,
            video_gen_output,
            video_gen_output_box,
            video_gen_share_links,
            music_video,
            video_gen_audio,
        )
        create_ui_handler(
            speed_btn,
            speed_output,
            speed_output_box,
            speed_share_links,
            change_audio_speed,
            speed_input,
            speed_factor,
            preserve_pitch,
            speed_format,
        )
        create_ui_handler(
            stem_btn,
            stem_output,
            stem_output_box,
            stem_share_links,
            separate_stems,
            stem_input,
            stem_type,
            stem_format,
        )
        create_ui_handler(
            vps_btn,
            vps_output,
            vps_output_box,
            vps_share_links,
            pitch_shift_vocals,
            vps_input,
            vps_pitch,
            vps_format,
        )
        create_ui_handler(
            dj_btn,
            dj_output,
            dj_output_box,
            dj_share_links,
            dj_mix,
            dj_files,
            dj_mix_type,
            dj_target_bpm,
            dj_transition,
            dj_format,
        )
        create_ui_handler(
            gen_btn,
            gen_output,
            gen_output_box,
            gen_share_links,
            generate_music,
            gen_prompt,
            gen_duration,
            gen_format,
        )
        create_ui_handler(
            vg_btn,
            vg_output,
            vg_output_box,
            vg_share_links,
            generate_voice,
            vg_text,
            vg_ref,
            vg_format,
        )
        create_ui_handler(
            vis_btn,
            vis_output,
            vis_output_box,
            vis_share_links,
            beat_visualizer,
            vis_image_input,
            vis_audio_input,
            vis_effect,
            vis_animation,
            vis_intensity,
        )
        create_ui_handler(
            lyric_btn,
            lyric_output,
            lyric_output_box,
            lyric_share_links,
            lyric_video,
            lyric_audio,
            lyric_bg,
            lyric_text,
            lyric_position,
        )

        def feedback_ui(audio_path):
            yield {
                feedback_btn: gr.update(
                    value="Analyzing...", interactive=False
                ),
                feedback_output: "",
            }
            try:
                feedback_text = get_audio_feedback(audio_path)
                yield {
                    feedback_btn: gr.update(
                        value="Get Feedback", interactive=True
                    ),
                    feedback_output: feedback_text,
                }
            except Exception as e:
                yield {
                    feedback_btn: gr.update(
                        value="Get Feedback", interactive=True
                    )
                }
                raise gr.Error(str(e))

        feedback_btn.click(
            feedback_ui, [feedback_input], [feedback_btn, feedback_output]
        )

        def instrument_id_ui(audio_path):
            yield {
                instrument_id_btn: gr.update(
                    value="Identifying...", interactive=False
                ),
                instrument_id_output: "",
            }
            try:
                instrument_text = identify_instruments(audio_path)
                yield {
                    instrument_id_btn: gr.update(
                        value="Identify Instruments", interactive=True
                    ),
                    instrument_id_output: instrument_text,
                }
            except Exception as e:
                yield {
                    instrument_id_btn: gr.update(
                        value="Identify Instruments", interactive=True
                    )
                }
                raise gr.Error(str(e))

        instrument_id_btn.click(
            instrument_id_ui,
            [instrument_id_input],
            [instrument_id_btn, instrument_id_output],
        )

        def analysis_ui(audio_path):
            yield {
                analysis_btn: gr.update(
                    value="Analyzing...", interactive=False
                ),
                analysis_bpm_key_output: "",
            }
            try:
                bpm_key = analyze_audio_features(audio_path)
                yield {
                    analysis_btn: gr.update(
                        value="Analyze Audio", interactive=True
                    ),
                    analysis_bpm_key_output: bpm_key,
                }
            except Exception as e:
                yield {
                    analysis_btn: gr.update(
                        value="Analyze Audio", interactive=True
                    )
                }
                raise gr.Error(str(e))

        analysis_btn.click(
            analysis_ui,
            [analysis_input],
            [analysis_btn, analysis_bpm_key_output],
        )

        def stt_ui(audio_path, language):
            yield {
                stt_btn: gr.update(value="Transcribing...", interactive=False),
                stt_output: "",
                stt_file_output: gr.update(visible=False),
            }
            try:
                transcript = transcribe_audio(audio_path, language)
                file_path = save_text_to_file(transcript)
                yield {
                    stt_btn: gr.update(
                        value="Transcribe Audio", interactive=True
                    ),
                    stt_output: transcript,
                    stt_file_output: gr.update(visible=True, value=file_path),
                }
            except Exception as e:
                yield {
                    stt_btn: gr.update(
                        value="Transcribe Audio", interactive=True
                    )
                }
                raise gr.Error(str(e))

        stt_btn.click(
            stt_ui,
            [stt_input, stt_language],
            [stt_btn, stt_output, stt_file_output],
        )

        def spec_ui(audio_path):
            yield {
                spec_btn: gr.update(value="Generating...", interactive=False),
                spec_output: None,
            }
            try:
                spec_image = create_spectrum_visualization(audio_path)
                yield {
                    spec_btn: gr.update(
                        value="Generate Spectrum", interactive=True
                    ),
                    spec_output: spec_image,
                }
            except Exception as e:
                yield {
                    spec_btn: gr.update(
                        value="Generate Spectrum", interactive=True
                    )
                }
                raise gr.Error(str(e))

        spec_btn.click(spec_ui, [spec_input], [spec_btn, spec_output])

        def clear_ui(*components):
            updates = {}
            for comp in components:
                if isinstance(
                    comp,
                    (
                        gr.Audio,
                        gr.Video,
                        gr.Image,
                        gr.File,
                        gr.Textbox,
                        gr.Markdown,
                    ),
                ):
                    updates[comp] = None
                if isinstance(comp, gr.Group):
                    updates[comp] = gr.update(visible=False)
            return updates

        clear_enhancer_btn.click(
            lambda: clear_ui(
                enhancer_input, enhancer_output, enhancer_output_box
            ),
            [],
            [enhancer_input, enhancer_output, enhancer_output_box],
        )
        clear_a2m_btn.click(
            lambda: clear_ui(a2m_input, a2m_output, a2m_output_box),
            [],
            [a2m_input, a2m_output, a2m_output_box],
        )
        clear_m2a_btn.click(
            lambda: clear_ui(m2a_input, m2a_output, m2a_output_box),
            [],
            [m2a_input, m2a_output, m2a_output_box],
        )
        clear_extender_btn.click(
            lambda: clear_ui(
                extender_input, extender_output, extender_output_box
            ),
            [],
            [extender_input, extender_output, extender_output_box],
        )
        clear_stem_mixer_btn.click(
            lambda: clear_ui(
                stem_mixer_files, stem_mixer_output, stem_mixer_output_box
            ),
            [],
            [stem_mixer_files, stem_mixer_output, stem_mixer_output_box],
        )
        clear_feedback_btn.click(
            lambda: clear_ui(feedback_input, feedback_output),
            [],
            [feedback_input, feedback_output],
        )
        clear_instrument_id_btn.click(
            lambda: clear_ui(instrument_id_input, instrument_id_output),
            [],
            [instrument_id_input, instrument_id_output],
        )
        clear_video_gen_btn.click(
            lambda: clear_ui(
                video_gen_audio, video_gen_output, video_gen_output_box
            ),
            [],
            [video_gen_audio, video_gen_output, video_gen_output_box],
        )
        clear_speed_btn.click(
            lambda: clear_ui(speed_input, speed_output, speed_output_box),
            [],
            [speed_input, speed_output, speed_output_box],
        )
        clear_stem_btn.click(
            lambda: clear_ui(stem_input, stem_output, stem_output_box),
            [],
            [stem_input, stem_output, stem_output_box],
        )
        clear_vps_btn.click(
            lambda: clear_ui(vps_input, vps_output, vps_output_box),
            [],
            [vps_input, vps_output, vps_output_box],
        )
        clear_dj_btn.click(
            lambda: clear_ui(dj_files, dj_output, dj_output_box),
            [],
            [dj_files, dj_output, dj_output_box],
        )
        clear_gen_btn.click(
            lambda: {
                **clear_ui(gen_output, gen_output_box),
                **{gen_prompt: ""},
            },
            [],
            [gen_output, gen_output_box, gen_prompt],
        )
        clear_vg_btn.click(
            lambda: {
                **clear_ui(vg_ref, vg_output, vg_output_box),
                **{vg_text: ""},
            },
            [],
            [vg_ref, vg_output, vg_output_box, vg_text],
        )
        clear_analysis_btn.click(
            lambda: {
                **clear_ui(analysis_input),
                **{analysis_bpm_key_output: ""},
            },
            [],
            [analysis_input, analysis_bpm_key_output],
        )
        clear_stt_btn.click(
            lambda: clear_ui(stt_input, stt_output, stt_file_output),
            [],
            [stt_input, stt_output, stt_file_output],
        )
        clear_spec_btn.click(
            lambda: clear_ui(spec_input, spec_output),
            [],
            [spec_input, spec_output],
        )
        clear_vis_btn.click(
            lambda: clear_ui(
                vis_image_input, vis_audio_input, vis_output, vis_output_box
            ),
            [],
            [vis_image_input, vis_audio_input, vis_output, vis_output_box],
        )
        clear_lyric_btn.click(
            lambda: {
                **clear_ui(
                    lyric_audio, lyric_bg, lyric_output, lyric_output_box
                ),
                **{lyric_text: ""},
            },
            [],
            [lyric_audio, lyric_bg, lyric_output, lyric_output_box, lyric_text],
        )

        load_transcript_btn.click(
            transcribe_audio, [lyric_audio, lyric_language], [lyric_text]
        )

    app.launch(
        server_name="0.0.0.0", theme=theme(), css=css(), server_port=7860
    )


def _gui_train():
    import gradio as gr

    from definers import (
        _system,
        css,
        infer,
        theme,
        train,
    )

    _system.install_ffmpeg()

    def handle_training(
        features,
        labels,
        model_path,
        remote_src,
        dataset_label_columns,
        revision,
        url_type,
        drop_list,
        selected_rows,
    ):
        return train(
            model_path,
            remote_src,
            revision,
            url_type,
            features,
            labels,
            dataset_label_columns,
            drop_list,
            selected_rows,
        )

    with gr.Blocks() as app:
        gr.Markdown("# Train your own models")

        with gr.Tabs():
            with gr.TabItem("Train"):
                with gr.Row():
                    with gr.Column():
                        model_train = gr.File(
                            label="Upload Model (for re-training)"
                        )
                        remote = gr.Textbox(
                            placeholder="Remote Dataset",
                            label="HuggingFace name or URL",
                        )
                        revision = gr.Textbox(
                            placeholder="Dataset Revision", label="Revision"
                        )
                        tpe = gr.Dropdown(
                            label="Remote Dataset Type",
                            choices=[
                                "parquet",
                                "json",
                                "csv",
                                "arrow",
                                "webdataset",
                                "txt",
                            ],
                        )
                        drop_list = gr.Textbox(
                            placeholder="Ignored Columns (semi-colon separated)",
                            label="Drop List",
                        )
                        label_columns = gr.Textbox(
                            placeholder="Label Columns (semi-colon separated)",
                            label="Label Columns",
                        )
                        selected_rows = gr.Textbox(
                            placeholder="Single rows and ranges (space separated, use a hyphen to choose a range or rows)",
                            label="Selected Rows",
                        )

                    with gr.Column():
                        local_features = gr.File(
                            label="Local Features",
                            file_count="multiple",
                            allow_reordering=True,
                        )
                        local_labels = gr.File(
                            label="Local Labels (for supervised training)",
                            file_count="multiple",
                            allow_reordering=True,
                        )
                        train_button = gr.Button("Train", elem_classes="btn")
                        train_output = gr.File(label="Trained Model Output")

                train_button.click(
                    fn=handle_training,
                    inputs=[
                        local_features,
                        local_labels,
                        model_train,
                        remote,
                        label_columns,
                        revision,
                        tpe,
                        drop_list,
                        selected_rows,
                    ],
                    outputs=[train_output],
                )

            with gr.TabItem("Predict"):
                with gr.Row():
                    with gr.Column():
                        model_predict = gr.File(
                            label="Upload Model (for prediction)"
                        )
                        prediction_data = gr.File(label="Prediction Data")

                    with gr.Column():
                        predict_button = gr.Button(
                            "Predict", elem_classes="btn"
                        )
                        predict_output = gr.File(label="Prediction Output")

                predict_button.click(
                    fn=infer,
                    inputs=[model_predict, prediction_data],
                    outputs=[predict_output],
                )

    app.launch(
        server_name="0.0.0.0", theme=theme(), css=css(), server_port=7860
    )


def start(proj: str):
    proj = proj.strip().lower()
    func_name = f"_gui_{proj}"
    func = globals().get(func_name)
    if callable(func):
        return func()
    else:
        catch(f"Error: No project called '{proj}' !")
