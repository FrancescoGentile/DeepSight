##
##
##

from collections.abc import Iterator
from typing import Generic, TypeVar

from torch.utils.data import DataLoader as _DataLoader

from deepsight.structures import Batch
from deepsight.tasks import Dataset

S = TypeVar("S")
A = TypeVar("A")
T = TypeVar("T")


class DataLoader(Generic[S, A, T]):
    """A wrapper around PyTorch's DataLoader to batch samples from a dataset."""

    # ------------------------------------------------------------------------- #
    # Constructor
    # ------------------------------------------------------------------------- #

    def __init__(
        self,
        dataset: Dataset[S, A, T],
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
        drop_last: bool = False,
    ) -> None:
        self._loader = _DataLoader(
            dataset,  # type: ignore
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=self._collate_fn,
        )

    # ------------------------------------------------------------------------- #
    # Properties
    # ------------------------------------------------------------------------- #

    @property
    def dataset(self) -> Dataset[S, A, T]:
        """The dataset used to create the batches."""
        return self._loader.dataset  # type: ignore

    # ------------------------------------------------------------------------- #
    # Magic methods
    # ------------------------------------------------------------------------- #

    def __len__(self) -> int:
        """The number of batches."""
        return len(self._loader)

    def __iter__(self) -> Iterator[tuple[Batch[S], Batch[A], Batch[T]]]:
        """An iterator over the batches."""
        return iter(self._loader)

    # ------------------------------------------------------------------------- #
    # Private methods
    # ------------------------------------------------------------------------- #

    def _collate_fn(
        self, batch: list[tuple[S, A, T]]
    ) -> tuple[Batch[S], Batch[A], Batch[T]]:
        states, actions, targets = zip(*batch, strict=True)
        return Batch(states), Batch(actions), Batch(targets)
