##
##
##

import typing
from collections.abc import Iterable, Iterator
from typing import Generic, TypeVar

import torch
from typing_extensions import Self

from deepsight.typing import Moveable

T = TypeVar("T", bound=Moveable)


class Batch(Generic[T], Moveable):
    """A batch of samples."""

    # ------------------------------------------------------------------------- #
    # Constructor and Factory methods
    # ------------------------------------------------------------------------- #

    def __init__(self, samples: Iterable[T]) -> None:
        samples = tuple(samples)

        if len(samples) == 0:
            raise ValueError("Batch must contain at least one sample.")

        if any(sample.device != samples[0].device for sample in samples):
            raise ValueError("All samples must be on the same device.")

        self._samples = samples

    # ------------------------------------------------------------------------- #
    # Properties
    # ------------------------------------------------------------------------- #

    @property
    def device(self) -> torch.device:
        """The device the samples are on."""
        return self._samples[0].device

    # ------------------------------------------------------------------------- #
    # Public methods
    # ------------------------------------------------------------------------- #

    def move(self, device: torch.device, non_blocking: bool = False) -> Self:
        """Move this batch to the given device.

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
        if self.device == device:
            return self

        return self.__class__(
            [sample.move(device, non_blocking) for sample in self._samples]
        )

    def split(
        self,
        *,
        num_splits: int | None = None,
        split_size: int | None = None,
    ) -> Iterable[Self]:
        """Split this batch into smaller batches.

        Args:
            num_splits: The number of batches to split into. This must be specified if
                `split_size` is not.
            split_size: The size of each batch. This must be specified if `num_splits`
                is not.

        Returns:
            An iterable of batches.

        Raises:
            ValueError: If neither `num_splits` nor `split_size` is specified.
            ValueError: If `num_splits` and `split_size` are both specified and
                inconsistent.
            ValueError: If `num_splits` does not divide the batch size.
            ValueError: If `split_size` does not divide the batch size.
        """
        match num_splits, split_size:
            case None, None:
                raise ValueError("Either num_splits or split_size must be specified.")
            case _, None:
                if len(self) % num_splits != 0:  # type: ignore
                    raise ValueError(
                        "num_splits does not divide the batch size evenly."
                    )
                split_size = len(self) // num_splits  # type: ignore
            case None, _:
                if len(self) % split_size != 0:
                    raise ValueError(
                        "split_size does not divide the batch size evenly."
                    )
            case _, _:
                if num_splits * split_size != len(self):
                    raise ValueError(
                        "num_splits and split_size are inconsistent with the "
                        "batch size."
                    )

        for i in range(num_splits):  # type: ignore
            yield self.__class__(self._samples[i * split_size : (i + 1) * split_size])  # type: ignore

    # ------------------------------------------------------------------------- #
    # Magic methods
    # ------------------------------------------------------------------------- #

    def __len__(self) -> int:
        """Return the number of samples in this batch."""
        return len(self._samples)

    @typing.overload
    def __getitem__(self, index: int) -> T:
        ...

    @typing.overload
    def __getitem__(self, index: slice) -> Self:
        ...

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
