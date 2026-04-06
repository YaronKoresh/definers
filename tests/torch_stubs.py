import types
from unittest.mock import MagicMock

import numpy as np


class FakeTensor:
    def __init__(self, value, *, device="cpu", requires_grad=False):
        self.value = np.array(value, dtype=np.float32)
        self.device = device
        self.requires_grad = requires_grad

    def squeeze(self):
        return FakeTensor(
            np.squeeze(self.value),
            device=self.device,
            requires_grad=self.requires_grad,
        )

    def reshape(self, *shape):
        return FakeTensor(
            self.value.reshape(*shape),
            device=self.device,
            requires_grad=self.requires_grad,
        )

    def cpu(self):
        return FakeTensor(
            self.value,
            device="cpu",
            requires_grad=self.requires_grad,
        )

    def numpy(self):
        return np.array(self.value)


class FakeParameter:
    def __init__(self, device="cpu"):
        self.device = device


class FakeModel:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.parameter = FakeParameter()
        self.loaded_state_dict = None
        self.in_eval_mode = False

    def to(self, device):
        self.parameter.device = device
        return self

    def cuda(self):
        self.parameter.device = "cuda"
        return self

    def parameters(self):
        yield self.parameter

    def load_state_dict(self, state_dict):
        expected_input_dim = state_dict.get("input_dim")
        if (
            expected_input_dim is not None
            and expected_input_dim != self.input_dim
        ):
            raise ValueError("input dimension mismatch")
        self.loaded_state_dict = state_dict

    def state_dict(self):
        return {"input_dim": self.input_dim}

    def eval(self):
        self.in_eval_mode = True

    def __call__(self, tensor):
        values = tensor.value
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError("dimension mismatch")
        return FakeTensor(
            values.sum(axis=1, keepdims=True), device=tensor.device
        )


class FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def build_fake_torch(*, load_side_effect=None, load_return_value=None):
    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = np.float32
    fake_torch.device = lambda value: value
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.no_grad = FakeNoGrad
    fake_torch.randn = lambda *shape: np.zeros(shape, dtype=np.float32)

    def tensor(value, dtype=None, device="cpu", requires_grad=False):
        return FakeTensor(
            value,
            device=device,
            requires_grad=requires_grad,
        )

    def load(path, map_location="cpu"):
        if load_side_effect is not None:
            raise load_side_effect
        if load_return_value is not None:
            return load_return_value
        return {"loaded_from": path, "map_location": map_location}

    fake_torch.tensor = tensor
    fake_torch.load = MagicMock(side_effect=load)
    fake_torch.save = MagicMock()
    fake_torch.nn = types.SimpleNamespace()
    fake_torch.optim = types.SimpleNamespace()
    fake_torch.nn.MSELoss = MagicMock()
    fake_torch.nn.MSELoss.return_value = MagicMock()
    fake_torch.nn.MSELoss.return_value.return_value = MagicMock()
    fake_torch.nn.MSELoss.return_value.return_value.backward = MagicMock()
    fake_torch.optim.SGD = MagicMock(return_value=MagicMock())
    return fake_torch
