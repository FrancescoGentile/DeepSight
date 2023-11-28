##
##
##

from collections.abc import Iterator
from typing import Any

from torch.utils.data import DataLoader as _DataLoader

from deepsight.core import Batch, Dataset
from deepsight.typing import StateDict, Stateful

from ._sampler import BatchSampler, RandomSampler, SequentialSampler


class DataLoader[S, A, T](Stateful):
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
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self._loader = _DataLoader(
            dataset,  # type: ignore
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
        )

    # ------------------------------------------------------------------------- #
    # Properties
    # ------------------------------------------------------------------------- #

    @property
    def dataset(self) -> Dataset[S, A, T]:
        """The dataset used to create the batches."""
        return self._loader.dataset  # type: ignore

    @property
    def batch_size(self) -> int:
        """The batch size."""
        return self._loader.batch_size  # type: ignore

    # ------------------------------------------------------------------------- #
    # Public methods
    # ------------------------------------------------------------------------- #

    def num_batches(self) -> int:
        """Get the number of batches."""
        return len(self)

    def num_samples(self) -> int:
        """Get the number of samples."""
        if self._loader.drop_last:
            return self.num_batches() * self.batch_size
        else:
            return len(self.dataset)

    def state_dict(self) -> StateDict:
        return self._loader.batch_sampler.state_dict()  # type: ignore

    def load_state_dict(self, state_dict: StateDict) -> Any:
        self._loader.batch_sampler.load_state_dict(state_dict)  # type: ignore

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
