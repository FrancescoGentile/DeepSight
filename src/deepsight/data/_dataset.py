##
##
##

from typing import Protocol


class Dataset[S, A, P](Protocol):
    """Interface for all datasets."""

    def __len__(self) -> int:
        """The number of samples in the dataset."""
        ...

    def __getitem__(self, index: int) -> tuple[S, A, P]:
        """Get a sample, its annotations and the ground truth predictions."""
        ...
