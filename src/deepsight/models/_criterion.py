# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Literal, Protocol

from deepsight.data import Batch
from deepsight.typing import Tensor


@dataclass(frozen=True)
class LossInfo:
    """Info of a loss computed by a criterion for a model."""

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

    def compute(
        self, output: O, annotations: Batch[A]
    ) -> dict[str, Tensor[Literal[""], float]]:
        """Compute the losses.

        Args:
            output: The output of the model.
            annotations: The ground truth annotations for the samples.

        Returns:
            The losses computed by the criterion.
        """
        ...
