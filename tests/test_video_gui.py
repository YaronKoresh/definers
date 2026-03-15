import pytest


def test_filter_styles_returns_update():

    import definers.video_gui as video_gui

    result = video_gui.filter_styles("psy", "All")
    assert isinstance(result, dict)
    assert "choices" in result

    result2 = video_gui.filter_styles("nope", "All")
    assert result2.get("choices") == []


def test_normalize_arr_constant():
    import numpy as np

    import definers.video_gui as video_gui

    arr = np.full((3, 3), 5.0)
    out = video_gui.normalize_arr(arr)
    assert np.all(out == 0)

    arr2 = np.array([])
    out2 = video_gui.normalize_arr(arr2)
    assert out2.shape == arr2.shape
