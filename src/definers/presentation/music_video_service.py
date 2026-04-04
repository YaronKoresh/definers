from __future__ import annotations

import importlib
import random


def load_numeric_backend():
    try:
        return importlib.import_module("cupy")
    except Exception:
        return importlib.import_module("numpy")


def music_video(audio_path, width=1920, height=1080, fps=30):
    import cv2
    import librosa
    import madmom
    from moviepy import AudioFileClip
    from moviepy.video.VideoClip import VideoClip

    from definers.system import cores
    from definers.video_gui import draw_star_of_david

    np = load_numeric_backend()
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
        israel_blue = (0, 56, 184)
        metalic_black = (44, 44, 43)
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
        frame[gap_height : gap_height + stripe_height] = israel_blue
        frame[h - gap_height - stripe_height : h - gap_height] = israel_blue
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
            metalic_black,
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
