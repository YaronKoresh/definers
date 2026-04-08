import sys
import types
from unittest.mock import MagicMock

import numpy as np


class FakeTensor:
    def __init__(self, value, *, device="cpu", requires_grad=False):
        self.value = np.array(value)
        self.device = device
        self.requires_grad = requires_grad

    @property
    def shape(self):
        return self.value.shape

    @property
    def dtype(self):
        return self.value.dtype

    def size(self):
        return self.value.shape

    def __len__(self):
        return len(self.value)

    def __getitem__(self, index):
        return FakeTensor(
            self.value[index],
            device=self.device,
            requires_grad=self.requires_grad,
        )

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

    def to(self, target):
        if isinstance(target, str):
            return FakeTensor(
                self.value,
                device=target,
                requires_grad=self.requires_grad,
            )
        return FakeTensor(
            self.value.astype(target),
            device=self.device,
            requires_grad=self.requires_grad,
        )

    def is_floating_point(self):
        return np.issubdtype(self.value.dtype, np.floating)

    def max(self):
        return self.value.max()

    def min(self):
        return self.value.min()

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


class FakeTensorDataset:
    def __init__(self, *tensors):
        self.tensors = tuple(tensors)

    def __len__(self):
        if not self.tensors:
            return 0
        return len(self.tensors[0])

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)


def _stack_fake_tensors(values):
    return FakeTensor(np.stack([value.value for value in values], axis=0))


def _pad_sequence(sequences, batch_first=True):
    max_length = max(sequence.value.shape[0] for sequence in sequences)
    trailing_shape = sequences[0].value.shape[1:]
    padded = []
    for sequence in sequences:
        pad_shape = (max_length,) + trailing_shape
        target = np.zeros(pad_shape, dtype=sequence.value.dtype)
        target[: sequence.value.shape[0]] = sequence.value
        padded.append(target)
    axis = 0 if batch_first else 1
    return FakeTensor(np.stack(padded, axis=axis))


class FakeDataLoader:
    def __init__(
        self,
        dataset,
        pin_memory=False,
        num_workers=0,
        batch_size=1,
        shuffle=True,
        drop_last=False,
        sampler=None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler

    def __iter__(self):
        indices = (
            list(self.sampler)
            if self.sampler is not None
            else list(range(len(self.dataset)))
        )
        for start_index in range(0, len(indices), self.batch_size):
            batch_indices = indices[start_index : start_index + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                break
            batch_rows = [self.dataset[index] for index in batch_indices]
            if not batch_rows:
                continue
            columns = list(zip(*batch_rows))
            yield tuple(_stack_fake_tensors(column) for column in columns)

    def __len__(self):
        dataset_length = len(self.dataset)
        if dataset_length == 0:
            return 0
        full_batches, remainder = divmod(dataset_length, self.batch_size)
        if self.drop_last or remainder == 0:
            return full_batches
        return full_batches + 1


def build_fake_torch(*, load_side_effect=None, load_return_value=None):
    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = FakeTensor
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float16 = np.float16
    fake_torch.float32 = np.float32
    fake_torch.float64 = np.float64
    fake_torch.int = np.int32
    fake_torch.int8 = np.int8
    fake_torch.int16 = np.int16
    fake_torch.int32 = np.int32
    fake_torch.int64 = np.int64
    fake_torch.uint8 = np.uint8
    fake_torch.uint16 = np.uint16
    fake_torch.uint32 = np.uint32
    fake_torch.uint64 = np.uint64
    fake_torch.device = lambda value: value
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        is_bf16_supported=lambda: False,
    )
    fake_torch.no_grad = FakeNoGrad
    fake_torch.randn = lambda *shape: FakeTensor(
        np.zeros(shape, dtype=np.float32)
    )

    def tensor(value, dtype=None, device="cpu", requires_grad=False):
        tensor_dtype = np.float32 if dtype is None else dtype
        return FakeTensor(
            np.array(value, dtype=tensor_dtype),
            device=device,
            requires_grad=requires_grad,
        )

    def stack(values):
        return FakeTensor(np.stack([value.value for value in values], axis=0))

    def load(path, map_location="cpu"):
        if load_side_effect is not None:
            raise load_side_effect
        if load_return_value is not None:
            return load_return_value
        return {"loaded_from": path, "map_location": map_location}

    fake_torch.tensor = tensor
    fake_torch.as_tensor = tensor
    fake_torch.stack = MagicMock(side_effect=stack)
    fake_torch.equal = lambda left, right: np.array_equal(
        left.value,
        right.value,
    )
    fake_torch.is_floating_point = lambda tensor: tensor.is_floating_point()
    fake_torch.load = MagicMock(side_effect=load)
    fake_torch.save = MagicMock()
    fake_torch.nn = types.SimpleNamespace()
    fake_torch.optim = types.SimpleNamespace()
    fake_torch.nn.utils = types.SimpleNamespace(
        rnn=types.SimpleNamespace(pad_sequence=_pad_sequence)
    )
    fake_torch.nn.MSELoss = MagicMock()
    fake_torch.nn.MSELoss.return_value = MagicMock()
    fake_torch.nn.MSELoss.return_value.return_value = MagicMock()
    fake_torch.nn.MSELoss.return_value.return_value.backward = MagicMock()
    fake_torch.optim.SGD = MagicMock(return_value=MagicMock())
    return fake_torch


def build_fake_torch_modules(*, load_side_effect=None):
    fake_torch = build_fake_torch(load_side_effect=load_side_effect)
    fake_torch_utils = types.ModuleType("torch.utils")
    fake_torch_utils_data = types.ModuleType("torch.utils.data")
    fake_torch_utils_data.DataLoader = FakeDataLoader
    fake_torch_utils_data.TensorDataset = FakeTensorDataset
    fake_torch.utils = types.SimpleNamespace(data=fake_torch_utils_data)
    return fake_torch, fake_torch_utils, fake_torch_utils_data


def install_fake_torch_modules(monkeypatch, *, load_side_effect=None):
    fake_torch, fake_torch_utils, fake_torch_utils_data = (
        build_fake_torch_modules(load_side_effect=load_side_effect)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.utils", fake_torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", fake_torch_utils_data)
    return fake_torch, fake_torch_utils_data
