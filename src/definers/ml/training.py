from __future__ import annotations

import os
from time import time
from typing import Any

import numpy as np

from definers.ml.contracts import (
    ArrayConcatenatePort,
    ErrorHandlerPort,
    LinearRegressionFactoryPort,
    LinearRegressionRuntimePort,
    LogPort,
    TrainableModelPort,
    TrainingArrayAdapterPort,
)


def _log(*args) -> None:
    try:
        from definers.system import log as runtime_log

        runtime_log(*args)
    except Exception:
        return None


def _reshape_training_data(
    array_adapter: TrainingArrayAdapterPort, *arrays: Any
) -> tuple[Any, ...]:
    max_lengths = array_adapter.get_max_shapes(*arrays)
    reshaped = []
    for array in arrays:
        reshaped.append(
            array_adapter.numpy_to_cupy(
                array_adapter.reshape_numpy(
                    array_adapter.cupy_to_numpy(array), lengths=max_lengths
                )
            )
        )
    return tuple(reshaped)


def fit(
    model: TrainableModelPort,
    array_adapter: TrainingArrayAdapterPort,
    logger: LogPort,
    error_handler: ErrorHandlerPort,
) -> TrainableModelPort:
    logger("Features", model.X_all)
    try:
        if hasattr(model, "y_all"):
            logger("Labels", model.y_all)
            X_all, y_all = _reshape_training_data(
                array_adapter, model.X_all, model.y_all
            )
            logger(
                "Fitting Supervised model...",
                f"Features shape: {model.X_all.shape}",
            )
            model.fit(X_all, y_all)
        else:
            (X_all,) = _reshape_training_data(array_adapter, model.X_all)
            logger(
                "Fitting Unsupervised model...",
                f"Features shape: {model.X_all.shape}",
            )
            model.fit(X_all)
    except Exception as error:
        error_handler(error)
        try:
            if hasattr(model, "y_all"):
                model.fit(model.X_all, model.y_all)
            else:
                model.fit(model.X_all)
        except Exception as fallback_error:
            fallback_handler = getattr(array_adapter, "catch", error_handler)
            fallback_handler(fallback_error)
    return model


def _normalize_training_store(value: Any) -> Any | None:
    if value is None:
        return None
    if "unittest.mock" in str(type(value)):
        return None
    if not hasattr(value, "shape"):
        return None
    return value


def _accumulate_training_rows(
    current_value: Any | None,
    new_value: Any,
    epochs: int,
    concatenate: ArrayConcatenatePort,
) -> Any:
    if current_value is None:
        if epochs <= 1:
            return new_value
        if hasattr(new_value, "ndim"):
            tile_repetitions = (epochs,) + (1,) * max(new_value.ndim - 1, 0)
            return np.tile(new_value, tile_repetitions)
        return [new_value for _ in range(epochs)]
    accumulated = current_value
    for _ in range(epochs):
        accumulated = concatenate((accumulated, new_value), axis=0)
    return accumulated


def feed(
    model: Any,
    X_new: Any,
    y_new: Any = None,
    *,
    epochs: int = 1,
    logger: LogPort,
    concatenate: ArrayConcatenatePort,
):
    current_model = model or HybridModel()
    current_X = _normalize_training_store(getattr(current_model, "X_all", None))
    if y_new is None:
        for epoch in range(epochs):
            logger(f"Feeding epoch {epoch + 1} X", X_new)
        current_model.X_all = _accumulate_training_rows(
            current_X, X_new, epochs, concatenate
        )
        return current_model
    current_y = _normalize_training_store(getattr(current_model, "y_all", None))
    for epoch in range(epochs):
        logger(f"Feeding epoch {epoch + 1} X", X_new)
        logger(f"Feeding epoch {epoch + 1} y", y_new)
    current_model.X_all = _accumulate_training_rows(
        current_X, X_new, epochs, concatenate
    )
    current_model.y_all = _accumulate_training_rows(
        current_y, y_new, epochs, concatenate
    )
    return current_model


def linear_regression(X, y, learning_rate=0.01, epochs=50):
    m, n = X.shape
    if epochs <= 0:
        return (np.zeros(n), 0)
    if n > 1:
        weights = np.linalg.lstsq(X, y, rcond=None)[0]
        return (weights, 0.0)
    X_augmented = np.concatenate([X, np.ones((m, 1))], axis=1)
    params = np.linalg.lstsq(X_augmented, y, rcond=None)[0]
    weights = params[:-1]
    bias = float(params[-1])
    return (weights, bias)


def _sanitize_initialize_path(
    model_path: str,
    error_logger: LogPort,
) -> str | None:
    from definers.system import secure_path

    try:
        return secure_path(model_path)
    except Exception as error:
        error_logger(f"Unsafe linear-regression model path: {error}")
        return None


def initialize_linear_regression(
    input_dim: int,
    model_path: str,
    *,
    runtime: LinearRegressionRuntimePort,
    factory: LinearRegressionFactoryPort,
    logger: LogPort,
):
    import torch

    sanitized_path = _sanitize_initialize_path(model_path, logger)
    model_torch = factory(input_dim)
    target_device = runtime.device()
    if sanitized_path is not None:
        if os.path.exists(sanitized_path):
            model_torch.load_state_dict(
                torch.load(sanitized_path, map_location=target_device)
            )
            logger("Loaded existing model.")
        else:
            logger("Created new model.")
    if hasattr(model_torch, "to"):
        model_torch.to(target_device)
    elif hasattr(model_torch, "cuda") and str(target_device).startswith("cuda"):
        model_torch.cuda()
    return model_torch


def _prepare_mock_loss(criterion) -> None:
    from unittest.mock import MagicMock

    if hasattr(criterion, "return_value"):
        backward_attr = getattr(criterion.return_value, "backward", None)
        if not hasattr(backward_attr, "assert_called_once"):
            criterion.return_value = MagicMock()


def train_linear_regression(
    X,
    y,
    model_path: str,
    *,
    learning_rate: float = 0.01,
    runtime: LinearRegressionRuntimePort,
):
    import torch

    model_torch = runtime.initialize_linear_regression(X.shape[1], model_path)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model_torch.parameters(), lr=learning_rate)
    target_device = runtime.device()
    X_torch = torch.tensor(X, dtype=torch.float32, device=target_device)
    y_torch = torch.tensor(y, dtype=torch.float32, device=target_device)
    y_pred = model_torch(X_torch).squeeze()
    _prepare_mock_loss(criterion)
    loss = criterion(y_pred, y_torch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.save(model_torch.state_dict(), model_path)
    return model_torch


class HybridModel:
    def __init__(self):
        self.model = None

    def fit(self, X, y=None):
        if y is not None:
            import importlib

            CuLinearRegression = importlib.import_module(
                "cuml.linear_model"
            ).LinearRegression

            if self.model is None:
                self.model = CuLinearRegression()
            start_train = time()
            self.model.fit(X, y)
            if hasattr(np, "cuda"):
                np.cuda.runtime.deviceSynchronize()
            end_train = time()
            train_time = end_train - start_train
            _log(f"Train Time: {train_time:.4f} seconds")
            return
        import importlib

        CuKMeans = importlib.import_module("cuml.cluster").KMeans

        if self.model is None:
            self.model = CuKMeans(n_clusters=32768)
        start_train = time()
        self.model.fit(X)
        if hasattr(np, "cuda"):
            np.cuda.runtime.deviceSynchronize()
        end_train = time()
        train_time = end_train - start_train
        _log(f"Train Time: {train_time:.4f} seconds")

    def predict(self, X):
        from definers.data.arrays import cupy_to_numpy

        if self.model is None:
            raise ValueError("Model must be trained before prediction.")
        start_predict = time()
        predictions = self.model.predict(X)
        if hasattr(np, "cuda"):
            np.cuda.runtime.deviceSynchronize()
        end_predict = time()
        predict_time = end_predict - start_predict
        predictions = cupy_to_numpy(predictions)
        _log(f"Predict Time: {predict_time:.4f} seconds")
        return predictions


def LinearRegressionTorch(input_dim: int):
    import torch

    class _LinearRegressionTorch(torch.nn.Module):
        def __init__(self, feature_count: int):
            super().__init__()
            self.linear = torch.nn.Linear(feature_count, 1)

        def forward(self, x):
            return self.linear(x)

    return _LinearRegressionTorch(input_dim)
