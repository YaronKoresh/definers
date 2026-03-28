from unittest.mock import patch

import definers.chat as chat


def test_music_video_facade_delegates_to_service():
    with patch(
        "definers.presentation.music_video_service.music_video",
        return_value="video.mp4",
    ) as mock_music_video:
        result = chat.music_video("song.wav", width=320, height=240, fps=15)

    assert result == "video.mp4"
    mock_music_video.assert_called_once_with(
        "song.wav",
        width=320,
        height=240,
        fps=15,
    )


def test_lyric_video_facade_delegates_to_service():
    with patch(
        "definers.presentation.lyric_video_service.lyric_video",
        return_value="lyrics.mp4",
    ) as mock_lyric_video:
        result = chat.lyric_video(
            "song.wav",
            "background.png",
            "lyrics",
            "bottom",
            max_dim=720,
            font_size=80,
            text_color="yellow",
            stroke_color="blue",
            stroke_width=4,
            fade_duration=0.75,
        )

    assert result == "lyrics.mp4"
    mock_lyric_video.assert_called_once_with(
        "song.wav",
        "background.png",
        "lyrics",
        "bottom",
        max_dim=720,
        font_size=80,
        text_color="yellow",
        stroke_color="blue",
        stroke_width=4,
        fade_duration=0.75,
    )


def test_init_stable_whisper_facade_delegates_to_service():
    with patch(
        "definers.presentation.lyric_video_service.init_stable_whisper",
        return_value=None,
    ) as mock_init_stable_whisper:
        chat.init_stable_whisper()

    mock_init_stable_whisper.assert_called_once_with()