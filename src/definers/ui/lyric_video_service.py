from __future__ import annotations

import gc
import re
from pathlib import Path


def strip_nikud(text: str) -> str:
    return "".join(char for char in text if not "֑" <= char <= "ׇ")


def init_stable_whisper():
    from definers.constants import MODELS
    from definers.model_installation import load_stable_whisper_model

    print("Loading multilingual transcription model (stable-ts)...")
    MODELS["stable-whisper"] = load_stable_whisper_model(device_name="cpu")


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
    from definers.system.download_activity import (
        create_activity_reporter,
        report_download_activity,
    )
    from definers.system.output_paths import managed_output_path

    def clean_word(text_value):
        return "".join(filter(str.isalnum, text_value.lower()))

    report = create_activity_reporter(5)
    report(
        1,
        "Load lyric media",
        detail="Loading the audio and lyric inputs.",
    )
    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration
    lyrics_text = strip_nikud(lyrics_text)
    report(
        2,
        "Detect lyric language",
        detail="Detecting the lyric language and preparing synchronization.",
    )
    detected_lang = text.language(lyrics_text)
    print(f"🌍 Detected language: {detected_lang}")
    print("🎤 Starting automatic lyric synchronization...")
    timed_lyrics = []
    lines = [
        line.strip() for line in lyrics_text.strip().split("\n") if line.strip()
    ]
    line_report = create_activity_reporter(len(lines) or 1)
    if not lines:
        print("Warning: Lyrics text is empty.")
    else:
        lyrics_text = "\n".join(lines)
        try:
            if MODELS["stable-whisper"] is None:
                report_download_activity(
                    "Load lyric alignment model",
                    detail="Initializing the transcription runtime.",
                    phase="model",
                )
                init_stable_whisper()
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
                line_report(
                    i + 1,
                    "Align lyric lines",
                    detail=f"Aligning lyric line {i + 1}/{len(lines)}.",
                )
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
        except Exception as error:
            catch(
                f"Could not automatically sync lyrics: {error}. Video will have no lyrics."
            )
            return None
    log("Timed Lyrics", timed_lyrics)
    print("✅ Synchronization complete.")
    report(
        3,
        "Build lyric overlays",
        detail="Preparing the lyric overlay clips and background.",
    )
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
    clip_report = create_activity_reporter(len(timed_lyrics) or 1)
    for clip_index, (start, end, line) in enumerate(timed_lyrics, start=1):
        clip_report(
            clip_index,
            "Prepare lyric overlays",
            detail=(
                f"Preparing lyric overlay {clip_index}/{len(timed_lyrics) or 1}."
            ),
        )
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
    output_path = managed_output_path(
        "mp4",
        section="video",
        stem=f"{Path(audio_path).stem}_lyrics",
    )
    print(f"Writing video to temporary file: {output_path}")
    report(
        4,
        "Render lyric video",
        detail="Encoding the lyric video frames.",
    )
    final_clip.write_videofile(
        output_path,
        codec="libx264",
        fps=20,
        audio_codec="aac",
        preset="ultrafast",
        threads=cores(),
    )
    print("Video rendering complete.")
    report(
        5,
        "Finalize lyric video",
        detail="Writing the lyric video output.",
    )
    return output_path
