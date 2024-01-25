##
##
##

from dataclasses import dataclass
from typing import Protocol

from deepsight.data import Batch
from deepsight.typing import Losses


class Criterion[O, A](Protocol):
    """Interface for criteria used to compute the losses of a model."""

    @dataclass(frozen=True)
    class LossInfo:
        """The info of a loss computed by a criterion."""

        name: str
        weight: float

    def get_losses_info(self) -> tuple[LossInfo, ...]:
        """Get the info of the losses computed by the criterion.

        Returns:
            The losses info.
        """
        ...

    def compute(self, output: O, annotations: Batch[A]) -> Losses:
        """Compute the losses.

        Args:
            output: The output of the model.
            annotations: The ground truth annotations for the samples.

        Returns:
            The losses computed by the criterion.
        """
        ...
