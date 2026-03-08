import pytest


def test_filter_styles_returns_update():

    from definers import _video_gui

    result = _video_gui.filter_styles("psy", "All")
    assert isinstance(result, dict)
    assert "choices" in result

    result2 = _video_gui.filter_styles("nope", "All")
    assert result2.get("choices") == []


def test_normalize_arr_constant():
    import numpy as np

    from definers import _video_gui

    arr = np.full((3, 3), 5.0)
    out = _video_gui.normalize_arr(arr)
    assert np.all(out == 0)

    arr2 = np.array([])
    out2 = _video_gui.normalize_arr(arr2)
    assert out2.shape == arr2.shape
