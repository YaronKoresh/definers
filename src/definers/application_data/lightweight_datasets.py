from __future__ import annotations

import math
import random

import numpy as np


class LightweightTensorDataset:
    def __init__(self, *tensors):
        self.tensors = tuple(tensors)

    def __len__(self):
        if not self.tensors:
            return 0
        try:
            return len(self.tensors[0])
        except Exception:
            return 1

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)


class LightweightSubset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def _stack_batch_column(column_values):
    try:
        return np.stack([np.asarray(value) for value in column_values], axis=0)
    except Exception:
        try:
            return np.asarray(column_values)
        except Exception:
            return list(column_values)


class LightweightDataLoader:
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
        self.batch_size = max(int(batch_size or 1), 1)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.sampler = sampler
        self.pin_memory = bool(pin_memory)
        self.num_workers = max(num_workers, 0)

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
            first_row = batch_rows[0]
            if isinstance(first_row, tuple):
                columns = list(zip(*batch_rows))
                yield tuple(
                    _stack_batch_column(column_values)
                    for column_values in columns
                )
                continue
            yield batch_rows

    def __len__(self):
        dataset_length = len(self.dataset)
        if dataset_length == 0:
            return 0
        full_batches, remainder = divmod(dataset_length, self.batch_size)
        if self.drop_last or remainder == 0:
            return full_batches
        return full_batches + 1


def resolve_tensor_dataset():
    try:
        from torch.utils.data import TensorDataset

        return TensorDataset
    except Exception:
        return LightweightTensorDataset


def resolve_subset():
    try:
        from torch.utils.data import Subset

        return Subset
    except Exception:
        return LightweightSubset


def resolve_data_loader():
    try:
        from torch.utils.data import DataLoader

        return DataLoader
    except Exception:
        return LightweightDataLoader


def split_indices(
    indices,
    *,
    test_size,
    random_state=None,
    stratify=None,
):
    del stratify
    ordered_indices = list(indices)
    if not ordered_indices:
        return [], []
    if isinstance(test_size, float):
        if test_size <= 0:
            return ordered_indices, []
        test_count = max(1, math.ceil(len(ordered_indices) * test_size))
    else:
        test_count = int(test_size)
    if test_count <= 0:
        return ordered_indices, []
    if test_count >= len(ordered_indices):
        return [], ordered_indices
    shuffled_indices = list(ordered_indices)
    random.Random(random_state).shuffle(shuffled_indices)
    test_selection = set(shuffled_indices[:test_count])
    train_indices = [
        index for index in ordered_indices if index not in test_selection
    ]
    test_indices = [
        index for index in ordered_indices if index in test_selection
    ]
    return train_indices, test_indices
