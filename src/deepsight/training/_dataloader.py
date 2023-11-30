##
##
##

from collections.abc import Iterator
from typing import Any

from torch.utils.data import DataLoader as _DataLoader

from deepsight import utils
from deepsight.core import Batch, Dataset
from deepsight.typing import Configs, Configurable, StateDict, Stateful

from ._sampler import BatchSampler, RandomSampler, SequentialSampler


class DataLoader[S, A, T](Stateful, Configurable):
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

        # by passing to the dataloaders only the samplers and not also these values,
        # these are not saved
        self._batch_size = batch_size
        self._drop_last = drop_last
        self._shuffle = shuffle

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
        return self._batch_size

    @property
    def num_batches(self) -> int:
        """The number of batches."""
        return len(self)

    @property
    def num_samples(self) -> int:
        """The number of samples."""
        if self._drop_last:
            return self.num_batches * self.batch_size
        else:
            return len(self.dataset)

    @property
    def drop_last(self) -> bool:
        """Whether to drop the last batch if it is not full."""
        return self._drop_last

    @property
    def shuffle(self) -> bool:
        """Whether to shuffle the samples."""
        return self._shuffle

    # ------------------------------------------------------------------------- #
    # Public methods
    # ------------------------------------------------------------------------- #

    def state_dict(self) -> StateDict:
        return self._loader.batch_sampler.state_dict()  # type: ignore

    def load_state_dict(self, state_dict: StateDict) -> Any:
        self._loader.batch_sampler.load_state_dict(state_dict)  # type: ignore

    def get_configs(self, recursive: bool) -> Configs:
        configs: Configs = {
            "batch_size": self._batch_size,
            "shuffle": self._shuffle,
            "num_workers": self._loader.num_workers,
            "drop_last": self._drop_last,
        }

        if recursive:
            configs["dataset"] = utils.get_configs(self.dataset, recursive)

        return configs

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
