from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import definers
from definers.data.loaders import load_audio_values


def test_load_audio_values_falls_back_when_sox_pipeline_fails():
    fake_transformer = MagicMock()
    fake_transformer.rate = MagicMock()
    fake_transformer.build_file = MagicMock()

    with (
        patch.object(
            definers,
            "system",
            SimpleNamespace(install_sox=MagicMock(return_value=True)),
            create=True,
        ),
        patch.object(
            definers,
            "sox",
            SimpleNamespace(
                Transformer=MagicMock(return_value=fake_transformer)
            ),
            create=True,
        ),
        patch.object(
            definers,
            "audio",
            SimpleNamespace(
                compact_audio=MagicMock(),
                remove_silence=MagicMock(),
                split_audio=MagicMock(
                    side_effect=[
                        ["segments_sox\\clip-01.mp3"],
                        ["segments_ffmpeg\\clip-01.mp3"],
                    ]
                ),
                extract_audio_features=MagicMock(
                    side_effect=[RuntimeError("boom"), [1.0, 2.0]]
                ),
            ),
            create=True,
        ),
        patch.object(
            definers,
            "data",
            SimpleNamespace(
                arrays=SimpleNamespace(
                    numpy_to_cupy=MagicMock(side_effect=lambda value: value)
                )
            ),
            create=True,
        ),
        patch(
            "definers.data.loaders.tmp",
            side_effect=[
                "temp_sox.wav",
                "temp_sox.mp3",
                "temp_ffmpeg.mp3",
                "temp_cleaned.mp3",
            ],
        ),
        patch(
            "tempfile.mkdtemp",
            side_effect=["segments_sox", "segments_ffmpeg"],
        ),
        patch("definers.data.loaders.catch") as mock_catch,
        patch("definers.data.loaders.delete") as mock_delete,
    ):
        result = load_audio_values("input.wav", training=True)

    assert result == [[1.0, 2.0]]
    mock_catch.assert_called_once()
    assert mock_delete.call_args_list == [
        call("temp_sox.wav"),
        call("temp_sox.mp3"),
        call("segments_sox"),
        call("temp_ffmpeg.mp3"),
        call("temp_cleaned.mp3"),
        call("segments_ffmpeg"),
    ]


def test_load_audio_values_uses_ffmpeg_path_when_sox_unavailable():
    with (
        patch.object(
            definers,
            "system",
            SimpleNamespace(install_sox=MagicMock(return_value=False)),
            create=True,
        ),
        patch.object(
            definers,
            "audio",
            SimpleNamespace(
                compact_audio=MagicMock(),
                remove_silence=MagicMock(),
                split_audio=MagicMock(return_value=[]),
                extract_audio_features=MagicMock(return_value=[3.0, 4.0]),
            ),
            create=True,
        ),
        patch.object(
            definers,
            "data",
            SimpleNamespace(
                arrays=SimpleNamespace(
                    numpy_to_cupy=MagicMock(side_effect=lambda value: value)
                )
            ),
            create=True,
        ),
        patch("definers.data.loaders.tmp", return_value="temp_ffmpeg.mp3"),
        patch("definers.data.loaders.delete"),
    ):
        result = load_audio_values("input.wav", training=False)

    assert result == [3.0, 4.0]
