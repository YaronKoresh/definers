import math
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from definers import split_mp3


class TestSplitMp3(unittest.TestCase):

    @patch("definers.AudioSegment")
    @patch("definers.Path")
    def test_split_audio_evenly(
        self, mock_path_cls, mock_audio_segment_cls
    ):
        # Setup mock for a 30-second audio file, split into 5-second chunks
        mock_audio = MagicMock()
        mock_audio.__len__.return_value = 30000  # 30 seconds in ms

        # Mock the slicing to return distinct mock chunks
        mock_chunks = [MagicMock(name=f"chunk_{i}") for i in range(6)]
        mock_audio.__getitem__.side_effect = mock_chunks

        mock_audio_segment_cls.from_mp3.return_value = mock_audio

        mock_path_instance = MagicMock()
        mock_path_instance.mkdir.return_value = None
        mock_path_cls.return_value = mock_path_instance

        with patch("definers.random.random", return_value=0.12345):
            export_path, num_chunks = split_mp3("dummy.mp3", 5)

        self.assertTrue(
            export_path.startswith(f"{os.getcwd()}/mp3_segments_")
        )
        self.assertEqual(num_chunks, 6)

        mock_path_instance.mkdir.assert_called_once_with(
            parents=True, exist_ok=True
        )
        mock_audio_segment_cls.from_mp3.assert_called_with(
            "dummy.mp3"
        )

        # Verify that 6 chunks were created and exported
        self.assertEqual(mock_audio.__getitem__.call_count, 6)
        self.assertEqual(len(mock_chunks[0].export.call_args_list), 1)

    @patch("definers.AudioSegment")
    @patch("definers.Path")
    def test_split_audio_unevenly(
        self, mock_path_cls, mock_audio_segment_cls
    ):
        # 22 seconds, 5 second chunks -> 5 chunks (4 full, 1 partial)
        mock_audio = MagicMock()
        mock_audio.__len__.return_value = 22000

        mock_chunks = [MagicMock(name=f"chunk_{i}") for i in range(5)]
        mock_audio.__getitem__.side_effect = mock_chunks

        mock_audio_segment_cls.from_mp3.return_value = mock_audio

        mock_path_cls.return_value.mkdir.return_value = None

        export_path, num_chunks = split_mp3("dummy.mp3", 5)

        self.assertEqual(num_chunks, 5)
        # Check that export was called on all 5 mocked chunks
        for chunk in mock_chunks:
            chunk.export.assert_called_once()

    @patch("definers.AudioSegment")
    @patch("definers.Path")
    def test_audio_shorter_than_chunk(
        self, mock_path_cls, mock_audio_segment_cls
    ):
        # 3 seconds, 5 second chunks -> 1 chunk
        mock_audio = MagicMock()
        mock_audio.__len__.return_value = 3000

        mock_chunk = MagicMock(name="chunk_0")
        mock_audio.__getitem__.return_value = mock_chunk

        mock_audio_segment_cls.from_mp3.return_value = mock_audio

        mock_path_cls.return_value.mkdir.return_value = None

        export_path, num_chunks = split_mp3("dummy.mp3", 5)

        self.assertEqual(num_chunks, 1)
        mock_chunk.export.assert_called_once()


if __name__ == "__main__":
    unittest.main()
