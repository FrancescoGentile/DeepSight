##
##
##

from typing import Generic, Protocol, TypeVar

S = TypeVar("S", covariant=True)
A = TypeVar("A", covariant=True)
P = TypeVar("P", covariant=True)


class Dataset(Protocol, Generic[S, A, P]):
    """Interface for all datasets."""

    def __len__(self) -> int:
        """The number of samples in the dataset."""
        ...

    def __getitem__(self, index: int) -> tuple[S, A, P]:
        """Get a sample, its annotations and the ground truth predictions."""
        ...
