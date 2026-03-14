import gc
import importlib
import random
import re

from definers.presentation.gui_registry import register_gui_launchers
from definers.presentation.launchers import create_gui_project_starter

try:
    np = importlib.import_module("cupy")
except Exception:
    np = importlib.import_module("numpy")


def _gui_translate():
    from definers.presentation.apps.translate import launch_translate_app

    return launch_translate_app()


def _gui_animation():
    from definers.presentation.apps.animation import launch_animation_app

    return launch_animation_app()


def _gui_image():
    from definers.presentation.apps.image import launch_image_app

    return launch_image_app()


def _gui_chat():
    from definers.presentation.apps.chat_app import launch_chat_app

    return launch_chat_app()


def _gui_faiss():
    from definers.presentation.apps.faiss import launch_faiss_app

    return launch_faiss_app()


def _gui_video():
    from definers.presentation.apps.video import launch_video_app

    return launch_video_app()


def _gui_audio():
    from definers.presentation.apps.audio import launch_audio_app

    return launch_audio_app()


def _gui_train():
    from definers.presentation.apps.train import launch_train_app

    return launch_train_app()


GUI_LAUNCHERS = register_gui_launchers(
    {
        "translate": "_gui_translate",
        "animation": "_gui_animation",
        "image": "_gui_image",
        "chat": "_gui_chat",
        "faiss": "_gui_faiss",
        "video": "_gui_video",
        "audio": "_gui_audio",
        "train": "_gui_train",
    },
    namespace=globals(),
)


def music_video(audio_path, width=1920, height=1080, fps=30):
    import cv2
    import librosa
    import madmom
    from moviepy import AudioFileClip
    from moviepy.video.VideoClip import VideoClip

    from definers.system import cores
    from definers.video_gui import draw_star_of_david

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

    from definers.constants import MODELS

    print("Loading multilingual transcription model (stable-ts)...")
    MODELS["stable-whisper"] = stable_whisper.load_model("tiny", device="cpu")


def lyric_video(
    audio_path,
    background_path,
    lyrics_text,
    text_position,
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

    import definers.text as text
    from definers.constants import MODELS
    from definers.system import catch, cores, log, tmp

    def clean_word(text):
        return "".join(filter(str.isalnum, text.lower()))

    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration
    lyrics_text = strip_nikud(lyrics_text)
    detected_lang = text.language(lyrics_text)
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
    def on_missing(project_name: str):
        from definers.system import catch

        catch(f"Error: No project called '{project_name}' !")

    return create_gui_project_starter(
        globals(),
        on_missing,
        registry=GUI_LAUNCHERS,
    ).start(proj)
