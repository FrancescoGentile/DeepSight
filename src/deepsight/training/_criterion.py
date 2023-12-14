##
##
##

from dataclasses import dataclass
from typing import Protocol

from deepsight.typing import Losses

from ._batch import Batch


@dataclass(frozen=True)
class LossInfo:
    name: str
    weight: float


class Criterion[O, A](Protocol):
    """Interface for criteria used to compute the losses of a model."""

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
