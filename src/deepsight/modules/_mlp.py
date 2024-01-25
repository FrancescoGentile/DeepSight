##
##
##

from collections.abc import Callable
from typing import Literal

from torch import nn

from deepsight.typing import Tensor

from ._activations import Activation
from ._module import Module


class FFN(Module):
    """A position-wise feed-forward network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        num_layers: int = 2,
        activation: Callable[[], Activation and nn.Module] = nn.GELU,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        """Initialize a position-wise feed-forward network.

        Args:
            input_dim: The dimension of the input.
            hidden_dim: The dimension of the hidden layers.
            output_dim: The dimension of the output. If `None`, defaults to
                `input_dim`.
            num_layers: The number of layers.
            activation: The activation function to use between hidden layers.
            dropout: The dropout rate.
            bias: Whether to include a bias term in the linear layers.
        """
        super().__init__()

        if output_dim is None:
            output_dim = input_dim

        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, output_dim, bias=bias))
        layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)

    def __call__(
        self, x: Tensor[Literal["... L D"], float]
    ) -> Tensor[Literal["... L D"], float]:
        """Apply the feed-forward network to the input.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        return self.layers(x)
