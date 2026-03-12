import unittest
from unittest.mock import MagicMock, patch

from definers import autotune_song, stretch_audio


class TestAudioRunCalls(unittest.TestCase):
    @patch("definers._audio.run")
    def test_stretch_audio_uses_list(self, mock_run):
        import tempfile

        inp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        inp.close()
        out = inp.name.replace(".wav", "_out.wav")
        stretch_audio(inp.name, output_path=out, speed_factor=1.1)
        self.assertTrue(mock_run.called)
        args = mock_run.call_args[0][0]
        self.assertIsInstance(args, list)
        self.assertEqual(args[0], "rubberband")

    @patch("definers._audio.normalize_audio_to_peak", side_effect=lambda p: p)
    @patch("definers._audio.run")
    @patch("madmom.features.beats.BeatTrackingProcessor")
    @patch("madmom.features.beats.RNNBeatProcessor")
    @patch(
        "definers._audio.separate_stems",
        return_value=("vocals.wav", "instr.wav"),
    )
    @patch(
        "definers._audio.analyze_audio_features",
        return_value=("C", "major", 120),
    )
    @patch("definers._audio.exist", return_value=True)
    @patch("librosa.onset.onset_detect", return_value=[0.1, 0.2])
    @patch("librosa.load")
    def test_autotune_song_uses_list(
        self,
        mock_load,
        mock_onset,
        mock_exist,
        mock_analysis,
        mock_separate,
        mock_rnn_cls,
        mock_beat_cls,
        mock_run,
        mock_norm,
    ):

        import numpy as np

        mock_load.return_value = (np.zeros(1024), 22050)

        mock_rnn = MagicMock()
        mock_rnn.return_value = np.array([0.1, 0.2])
        mock_rnn_cls.return_value = mock_rnn
        mock_beat = MagicMock()
        mock_beat.return_value = np.array([0.1, 0.2])
        mock_beat_cls.return_value = mock_beat
        import tempfile
        import wave

        path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        with wave.open(path, "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(22050)
            w.writeframes((np.zeros(22050)).astype(np.int16).tobytes())
        try:
            autotune_song(path, output_path=path)
        except Exception:
            pass
        self.assertTrue(mock_run.called)
        for call_args in mock_run.call_args_list:
            cmd = call_args[0][0]
            if isinstance(cmd, list):
                return
        self.fail("Expected at least one list-based run invocation")

    @patch("definers.logger.exception")
    def test_get_audio_duration_logs_error(self, mock_logger_exc):

        from definers import get_audio_duration

        res = get_audio_duration("no_such_file.wav")
        self.assertIsNone(res)
        mock_logger_exc.assert_called_once()


if __name__ == "__main__":
    unittest.main()
