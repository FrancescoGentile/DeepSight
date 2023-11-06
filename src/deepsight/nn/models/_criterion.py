##
##
##

from collections.abc import Iterable
from typing import Annotated, Generic, Protocol, TypeVar

from torch import Tensor

from deepsight.structures import Batch

O = TypeVar("O", contravariant=True)  # noqa: E741
A = TypeVar("A")


class Criterion(Generic[O, A], Protocol):
    @property
    def losses(self) -> Iterable[str]:
        """The names of the losses computed by the criterion."""
        ...

    def compute(
        self,
        output: O,
        annotations: Batch[A],
    ) -> dict[str, Annotated[Tensor, "", float]]:
        """Compute the losses.

        Args:
            output: The output of the model.
            annotations: The ground truth annotations for the samples.

        Returns:
            The losses computed by the criterion. Each loss must be a scalar
            tensor.
        """
        ...
