from dataclasses import dataclass

from definers.logger import init_logger
from definers.ml.inference import (
    predict_linear_regression as predict_runtime_linear_regression,
)
from definers.ml.training import (
    LinearRegressionTorch,
    initialize_linear_regression as initialize_runtime_linear_regression,
    linear_regression as runtime_linear_regression,
    train_linear_regression as train_runtime_linear_regression,
)

try:
    from definers.cuda import device
except Exception:

    def device():
        return "cpu"


logger = init_logger("definers.ml.regression_api")


@dataclass(frozen=True, slots=True)
class _LinearRegressionRuntime:
    device: object


@dataclass(frozen=True, slots=True)
class _TrainLinearRegressionRuntime:
    device: object
    initialize_linear_regression: object


def linear_regression(X, y, learning_rate=0.01, epochs=50):
    return runtime_linear_regression(
        X,
        y,
        learning_rate=learning_rate,
        epochs=epochs,
    )


def initialize_linear_regression(input_dim, model_path):
    return initialize_runtime_linear_regression(
        input_dim,
        model_path,
        runtime=_LinearRegressionRuntime(device=device),
        factory=LinearRegressionTorch,
        logger=logger.info,
    )


def train_linear_regression(X, y, model_path, learning_rate=0.01):
    return train_runtime_linear_regression(
        X,
        y,
        model_path,
        learning_rate=learning_rate,
        runtime=_TrainLinearRegressionRuntime(
            device=device,
            initialize_linear_regression=initialize_linear_regression,
        ),
    )


def predict_linear_regression(X_new, model_path):
    return predict_runtime_linear_regression(
        X_new,
        model_path,
        factory=LinearRegressionTorch,
    )


__all__ = [
    "initialize_linear_regression",
    "linear_regression",
    "predict_linear_regression",
    "train_linear_regression",
]
