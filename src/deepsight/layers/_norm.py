##
##
##

from typing import Any, Literal, Protocol

import torch
from torch import nn

from deepsight.typing import StateDict, Tensor

# --------------------------------------------------------------------------- #
# Sequence normalization layers
# --------------------------------------------------------------------------- #


class SequenceNorm(Protocol):
    """Interface for sequence normalization layers.

    This interface is used to type hint normalization layers that
    accept a sequence of vectors as input. It does not specify the
    way in which the normalization is performed.
    """

    def __call__(
        self, x: Tensor[Literal["... L D"], float], /
    ) -> Tensor[Literal["... L D"], float]: ...


# --------------------------------------------------------------------------- #
# Image normalization layers
# --------------------------------------------------------------------------- #


class ImageNorm(Protocol):
    """Interface for image normalization layers.

    This interface is used to type hint normalization layers that
    accept an image as input. It does not specify the way in which
    the normalization is performed.
    """

    def __call__(
        self, x: Tensor[Literal["B C H W"], float], /
    ) -> Tensor[Literal["B C H W"], float]: ...


class LayerNorm2D(nn.Module, ImageNorm):
    r"""Layer normalization for images.

    Applies layer normalization over a mini-batch of images as in the
    formula:

    $$
    y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} * \gamma + \beta
    $$

    The mean and standard-deviation are calculated over the channel dimension.
    The standard-deviation is calculated via the unbiased estimator, equivalent to
    `torch.var(input, unbiased=False)`. $\gamma$ and $\beta$ are learnable affine
    transform parameters of size $C$ if `elementwise_affine` is `True`.

    !!! note

        Applying this layer to a batch of images is equivalent to permute the images
        from the shape `(B, C, H, W)` to `(B, H, W, C)`, apply `torch.nn.LayerNorm` with
        `normalized_shape=C` and permute the result back to `B C H W`. This layer is
        just a more efficient implementation of the above procedure.
    """

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            if bias:
                self.bias = nn.Parameter(torch.zeros(num_channels))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        # for type hinting
        self.weight: torch.Tensor | None
        self.bias: torch.Tensor | None

        self.eps = eps

    def forward(
        self, x: Tensor[Literal["B C H W"], float]
    ) -> Tensor[Literal["B C H W"], float]:
        mean = x.mean(1, keepdim=True)
        # compute the var using unbiased estimator
        var = x.var(1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)

        if self.weight is not None:
            x = x * self.weight.view(1, -1, 1, 1)
        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1)

        return x


# Source code taken from torchvision
# https://github.com/pytorch/vision/blob/main/torchvision/ops/misc.py


class FrozenBatchNorm2d(nn.Module, ImageNorm):
    """BatchNorm2d where the batch statistics and the affine parameters are fixed."""

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        """Initialize the layer.

        Args:
            num_features: The number of channels in the input tensor.
            eps: A value added to the denominator for numerical stability.
        """
        super().__init__()

        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

        # for type hinting
        self.weight: torch.Tensor
        self.bias: torch.Tensor
        self.running_mean: torch.Tensor
        self.running_var: torch.Tensor

    # ----------------------------------------------------------------------- #
    # Public methods
    # ----------------------------------------------------------------------- #

    def forward(
        self, x: Tensor[Literal["B C H W"], float]
    ) -> Tensor[Literal["B C H W"], float]:
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    # ----------------------------------------------------------------------- #
    # Private methods
    # ----------------------------------------------------------------------- #

    def _load_from_state_dict(
        self,
        state_dict: StateDict,
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
