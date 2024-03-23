# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0
# --------------------------------------------------------------------------- #
# Copyright (c) 2016-present, Facebook Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# --------------------------------------------------------------------------- #
# Modified from:
# https://github.com/pytorch/pytorch/blob/main/torch/utils/data/sampler.py
# --------------------------------------------------------------------------- #

import math
from collections.abc import Iterator, Sized
from typing import Any, Protocol

import torch

from deepsight.typing import Stateful


class Sampler[T](Stateful, Protocol):
    def __iter__(self) -> Iterator[T]: ...

    def __len__(self) -> int: ...


class SequentialSampler(Sampler[int]):
    """A sampler that returns indices sequentially in order.

    For example, if the dataset has 10 samples, then the indices are returned in
    the following order: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].

    Different from PyTorch's SequentialSampler, this sampler is stateful. This means
    that it can be saved and loaded using the `state_dict` and `load_state_dict`
    methods, thus allowing to resume even in the middle of an epoch.
    """

    def __init__(self, data_source: Sized) -> None:
        self._data_source = data_source
        self._last_idx = None

    # ------------------------------------------------------------------------- #
    # Public Methods
    # ------------------------------------------------------------------------- #

    def state_dict(self) -> dict[str, Any]:
        return {"last_idx": self._last_idx}

    def load_state_dict(self, state_dict: dict[str, Any]) -> Any:
        self._last_idx = state_dict["last_idx"]

    # ------------------------------------------------------------------------- #
    # Magic methods
    # ------------------------------------------------------------------------- #

    def __iter__(self) -> Iterator[int]:
        self._last_idx = 0
        while self._last_idx < len(self._data_source):
            self._last_idx += 1
            yield self._last_idx - 1

    def __len__(self) -> int:
        return len(self._data_source)


class RandomSampler(Sampler[int]):
    """A sampler that returns indices randomly.

    Different from PyTorch's RandomSampler, this sampler is stateful. This means that it
    can be saved and loaded using the `state_dict` and `load_state_dict` methods, thus
    allowing to resume even in the middle of an epoch.
    """

    def __init__(self, data_source: Sized) -> None:
        self._data_source = data_source
        self._permutation = None
        self._last_idx = None

    # ------------------------------------------------------------------------- #
    # Public Methods
    # ------------------------------------------------------------------------- #

    def state_dict(self) -> dict[str, Any]:
        return {"permutation": self._permutation, "last_idx": self._last_idx}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._permutation = state_dict["permutation"]
        self._last_idx = state_dict["last_idx"]

    # ------------------------------------------------------------------------- #
    # Magic methods
    # ------------------------------------------------------------------------- #

    def __iter__(self) -> Iterator[int]:
        self._permutation = torch.randperm(len(self._data_source))
        self._last_idx = 0
        while self._last_idx < len(self._data_source):
            self._last_idx += 1
            yield int(self._permutation[self._last_idx - 1].item())

    def __len__(self) -> int:
        return len(self._data_source)


class BatchSampler(Sampler[list[int]]):
    def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool) -> None:
        if batch_size < 1:
            msg = f"batch_size must be greater than 0, got {batch_size}."
            raise ValueError(msg)

        self._sampler = sampler
        self._batch_size = batch_size
        self._drop_last = drop_last

    # ------------------------------------------------------------------------- #
    # Public Methods
    # ------------------------------------------------------------------------- #

    def state_dict(self) -> dict[str, Any]:
        return self._sampler.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._sampler.load_state_dict(state_dict)

    # ------------------------------------------------------------------------- #
    # Magic methods
    # ------------------------------------------------------------------------- #

    def __iter__(self) -> Iterator[list[int]]:
        batch = []
        for idx in self._sampler:
            batch.append(idx)
            if len(batch) == self._batch_size:
                yield batch
                batch = []

        if len(batch) > 0 and not self._drop_last:
            yield batch

    def __len__(self) -> int:
        if self._drop_last:
            return len(self._sampler) // self._batch_size
        else:
            return math.ceil(len(self._sampler) / self._batch_size)
