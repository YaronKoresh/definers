import sys
import types

import pytest


def test_filter_styles_returns_update():

    import definers.video.gui as video_gui

    result = video_gui.filter_styles("psy", "All")
    assert isinstance(result, dict)
    assert "choices" in result

    result2 = video_gui.filter_styles("nope", "All")
    assert result2.get("choices") == []


def test_normalize_arr_constant():
    import numpy as np

    import definers.video.gui as video_gui

    arr = np.full((3, 3), 5.0)
    out = video_gui.normalize_arr(arr)
    assert np.all(out == 0)

    arr2 = np.array([])
    out2 = video_gui.normalize_arr(arr2)
    assert out2.shape == arr2.shape


def test_generate_video_handler_reports_render_frame_activity(monkeypatch):
    import numpy as np

    import definers.video.gui as video_gui

    activity = []

    monkeypatch.setattr(
        "definers.system.download_activity.report_download_activity",
        lambda item_label=None, **kwargs: activity.append(
            (item_label, kwargs.get("detail"))
        ),
    )
    monkeypatch.setattr(
        video_gui,
        "prepare_common_resources",
        lambda *args, **kwargs: (64, 64, None, None),
    )
    monkeypatch.setattr(
        video_gui,
        "_render_composed_frame",
        lambda *args, **kwargs: np.zeros((64, 64, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "definers.audio.analyze_audio",
        lambda audio: {
            "duration": 2.0,
            "stft": np.zeros((1, 3)),
            "sr": 1,
            "hop_length": 1,
            "rms": np.array([0.0, 0.0, 0.0]),
            "beat_frames": [],
        },
    )
    monkeypatch.setattr(
        "definers.system.output_paths.managed_output_path",
        lambda *args, **kwargs: "video.mp4",
    )
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        types.SimpleNamespace(
            COLOR_BGR2RGB=1, cvtColor=lambda frame, *_args: frame
        ),
    )

    class FakeAudioFileClip:
        def __init__(self, path):
            self.path = path

    class FakeVideoClip:
        def __init__(self, make_frame, duration):
            self.make_frame = make_frame
            self.duration = duration

        def with_audio(self, audio_clip):
            return self

        def write_videofile(self, *_args, **_kwargs):
            self.make_frame(0.0)
            self.make_frame(self.duration)

    monkeypatch.setitem(
        sys.modules,
        "moviepy",
        types.SimpleNamespace(
            AudioFileClip=FakeAudioFileClip,
            VideoClip=FakeVideoClip,
        ),
    )

    result = video_gui.generate_video_handler(
        "audio.wav",
        None,
        "Psychedelic Geometry",
        "Landscape (16:9)",
        10,
        1,
        "Full",
        "default",
        [],
        [],
        "None",
        0.5,
        0.5,
        1.0,
        1.0,
        "",
        None,
    )

    assert result[0] == "video.mp4"
    labels = [item for item, _ in activity]
    assert labels[:4] == [
        "Validate media",
        "Analyze audio",
        "Prepare composition",
        "Render video frames",
    ]
    assert any("Rendering frame" in detail for _, detail in activity if detail)
    assert labels[-1] == "Finalize video"
