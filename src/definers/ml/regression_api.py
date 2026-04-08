from definers.logger import init_logger
from definers.ml.facade_api import MlFacadeApi
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


def linear_regression(X, y, learning_rate=0.01, epochs=50):
    return MlFacadeApi.linear_regression(
        X,
        y,
        linear_regression_fn=runtime_linear_regression,
        learning_rate=learning_rate,
        epochs=epochs,
    )


def initialize_linear_regression(input_dim, model_path):
    return MlFacadeApi.initialize_linear_regression(
        input_dim,
        model_path,
        initialize_linear_regression_fn=initialize_runtime_linear_regression,
        factory=LinearRegressionTorch,
        device_fn=device,
        logger=logger.info,
    )


def train_linear_regression(X, y, model_path, learning_rate=0.01):
    return MlFacadeApi.train_linear_regression(
        X,
        y,
        model_path,
        train_linear_regression_fn=train_runtime_linear_regression,
        initialize_linear_regression_fn=initialize_linear_regression,
        device_fn=device,
        learning_rate=learning_rate,
    )


def predict_linear_regression(X_new, model_path):
    return MlFacadeApi.predict_linear_regression(
        X_new,
        model_path,
        predict_linear_regression_fn=predict_runtime_linear_regression,
        factory=LinearRegressionTorch,
    )


__all__ = [
    "initialize_linear_regression",
    "linear_regression",
    "predict_linear_regression",
    "train_linear_regression",
]
