import os
import pickle

import joblib
import pytest

from definers import init_custom_model, predict


def test_predict_untrusted(tmp_path, capsys):
    model = tmp_path / "safe.pkl"
    joblib.dump({"a": 1}, model)

    os.environ["DEFINERS_TRUSTED_PATHS"] = str(tmp_path / "other")

    result = predict("foo.txt", str(model))
    assert result is None

    from unittest.mock import patch

    with patch("definers.logger.exception") as mock_exc:
        predict("foo.txt", str(model))
        assert mock_exc.called

        msg = str(mock_exc.call_args[0][0]) if mock_exc.call_args else ""
        assert str(model) in msg


def test_init_custom_model_untrusted(tmp_path, capsys):
    p = tmp_path / "m.pkl"
    with open(p, "wb") as f:
        pickle.dump({"y": 2}, f)
    os.environ["DEFINERS_TRUSTED_PATHS"] = str(tmp_path / "other")

    assert init_custom_model("pkl", str(p)) is None
    captured = capsys.readouterr()

    assert captured.out == ""


def test_predict_linear_regression_untrusted(tmp_path, capsys):

    import torch

    model_path = tmp_path / "lr.pt"
    dummy = torch.nn.Linear(1, 1)
    torch.save(dummy.state_dict(), model_path)
    os.environ["DEFINERS_TRUSTED_PATHS"] = str(tmp_path / "unrelated")
    import numpy as np

    result = None
    try:
        from definers import predict_linear_regression

        result = predict_linear_regression(np.zeros((1, 1)), str(model_path))
    except Exception:
        pass
    assert result is None
    captured = capsys.readouterr()
    assert "Unsafe model path" in captured.out
