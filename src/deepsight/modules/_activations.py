# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

"""Activation functions."""

from typing import Literal, Protocol

from deepsight.typing import Tensor


class Activation(Protocol):
    """Interface for activation functions.

    This interface is used to type hint activation functions that
    accept a tensor as input and return a tensor of the same shape
    as output.
    """

    def __call__(
        self, x: Tensor[Literal["..."], float]
    ) -> Tensor[Literal["..."], float]:
        """Applies the activation function to the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        ...
