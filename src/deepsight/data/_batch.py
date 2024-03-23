# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

import typing
from collections.abc import Iterator, Sequence
from typing import Self

import torch

from deepsight.typing import Moveable


class Batch[T](Moveable):
    """A batch of samples."""

    # ------------------------------------------------------------------------- #
    # Constructor and Factory methods
    # ------------------------------------------------------------------------- #

    def __init__(self, samples: Sequence[T]) -> None:
        super().__init__()

        if len(samples) == 0:
            msg = "Batch must contain at least one sample."
            raise ValueError(msg)

        self._samples = samples

    @classmethod
    def concat(cls, batches: Sequence[Self]) -> Self:
        """Concatenate multiple batches into a single batch.

        !!! note

            If only one batch is provided, it is returned as is.

        Args:
            batches: The batches to concatenate.

        Returns:
            The concatenated batch.
        """
        if len(batches) == 0:
            msg = "Must provide at least one batch."
            raise ValueError(msg)
        if len(batches) == 1:
            return batches[0]

        return cls([sample for batch in batches for sample in batch])

    # ------------------------------------------------------------------------- #
    # Public methods
    # ------------------------------------------------------------------------- #

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> Self:
        """Move this batch to the given device.

        !!! warning

            This method is not implemented for all sample types. If a sample type
            does not implement this method, the batch will not be moved and `self`
            will be returned.

        !!! note

            If the batch is already on the given device, `self` is returned. Otherwise,
            a copy is returned. How the samples are moved depends on the implementation
            of `to_device` for the sample type.

        Args:
            device: The device to move to.
            non_blocking: Whether to perform the move asynchronously.

        Returns:
            The batch on the given device.
        """
        if not isinstance(self._samples[0], Moveable):
            return self

        return self.__class__([
            sample.to(device, non_blocking=non_blocking)
            if isinstance(sample, Moveable)
            else sample
            for sample in self._samples
        ])

    def split(self, num_splits: int) -> tuple[Self, ...]:
        """Split this batch into smaller batches.

        Args:
            num_splits: The number of batches to split into.

        Returns:
            The smaller batches.

        Raises:
            ValueError: If the batch size is not divisible by `num_splits`.
        """
        if len(self) % num_splits != 0:
            msg = f"Batch size {len(self)} is not divisible by {num_splits}."
            raise ValueError(msg)
        split_size = len(self) // num_splits

        return tuple(
            self.__class__(self._samples[i * split_size : (i + 1) * split_size])
            for i in range(num_splits)
        )

    # ------------------------------------------------------------------------- #
    # Magic methods
    # ------------------------------------------------------------------------- #

    def __len__(self) -> int:
        """Return the number of samples in this batch."""
        return len(self._samples)

    @typing.overload
    def __getitem__(self, index: int) -> T: ...

    @typing.overload
    def __getitem__(self, index: slice) -> Self: ...

    def __getitem__(self, index: int | slice) -> T | Self:
        """Return the sample at the given index or a new batch of samples.

        Args:
            index: The index or slice to get.

        Returns:
            The sample at the given index or a new batch of samples
            if a slice was given.
        """
        if isinstance(index, int):
            return self._samples[index]

        return self.__class__(self._samples[index])

    def __iter__(self) -> Iterator[T]:
        """Return an iterator over the samples in this batch."""
        return iter(self._samples)

    # ------------------------------------------------------------------------- #
    # Private fields
    # ------------------------------------------------------------------------- #

    __slots__ = ("_samples",)
