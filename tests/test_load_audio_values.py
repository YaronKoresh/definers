from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import definers
from definers.data.loaders import load_audio_values


def test_load_audio_values_cleans_temp_files_after_failure():
    fake_transformer = MagicMock()
    fake_transformer.rate = MagicMock()
    fake_transformer.build_file = MagicMock()

    with (
        patch.object(
            definers,
            "sox",
            SimpleNamespace(
                Transformer=MagicMock(return_value=fake_transformer)
            ),
            create=True,
        ),
        patch.object(definers, "remove_silence", create=True),
        patch.object(
            definers,
            "split_mp3",
            return_value=("segments", None),
            create=True,
        ),
        patch.object(
            definers,
            "extract_audio_features",
            side_effect=RuntimeError("boom"),
            create=True,
        ),
        patch.object(
            definers,
            "numpy_to_cupy",
            side_effect=lambda value: value,
            create=True,
        ),
        patch(
            "definers.data.loaders.read",
            return_value=["segments\\clip-01.mp3"],
        ),
        patch(
            "definers.data.loaders.tmp",
            side_effect=["temp.wav", "temp.mp3"],
        ),
        patch("definers.data.loaders.delete") as mock_delete,
    ):
        try:
            load_audio_values("input.wav", training=True)
        except RuntimeError as error:
            assert str(error) == "boom"
        else:
            raise AssertionError("Expected RuntimeError")

    assert mock_delete.call_args_list == [
        call("temp.wav"),
        call("temp.mp3"),
        call("segments"),
    ]
