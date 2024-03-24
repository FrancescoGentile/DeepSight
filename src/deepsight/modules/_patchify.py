# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0
# --------------------------------------------------------------------------- #
# Copyright 2020 Ross Wightman
# SPDX-License-Identifier: Apache-2.0
# --------------------------------------------------------------------------- #
# Modified from:
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py
# --------------------------------------------------------------------------- #

from collections.abc import Callable

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from deepsight import utils
from deepsight.structures import BatchedImages

from ._module import Module


class ConvPatchify(Module):
    """Extract patches from images using a convolutional layer."""

    def __init__(
        self,
        patch_size: int | tuple[int, int],
        in_channels: int,
        embed_dim: int,
        norm_layer: Callable[[int], nn.Module] | None = None,
        bias: bool = True,
    ) -> None:
        """Initializes the patchify layer.

        Args:
            patch_size: The spatial size (h, w) of the patches to extract. If an integer
                is provided, the same patch size will be used for both spatial
                dimensions.
            in_channels: The number of input channels.
            embed_dim: The embedding dimension.
            norm_layer: A callable that takes the embedding dimension as input and
                returns a normalization layer. If `None`, no normalization will be
                applied.
            bias: Whether to add a bias term to the convolutional layer.
        """
        super().__init__()

        self._patch_size = utils.to_2tuple(patch_size)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=self._patch_size,
            stride=self._patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def patch_size(self) -> tuple[int, int]:
        """The spatial size (h, w) of the patches to extract."""
        return self._patch_size

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def compute_output_shape(
        self, input_shape: tuple[int, int] | torch.Size
    ) -> tuple[int, int]:
        return (
            (input_shape[0] + self.patch_size[0] - 1) // self.patch_size[0],
            (input_shape[1] + self.patch_size[1] - 1) // self.patch_size[1],
        )

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __call__(self, x: BatchedImages) -> BatchedImages:
        H, W = x.shape[-2:]  # noqa: N806
        pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
        pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]

        data = x.data
        if pad_h > 0 or pad_w > 0:
            data = F.pad(data, (0, pad_w, 0, pad_h))

        data = self.proj(data)  # (B, D, h, w)
        data = self.norm(data)  # (B, D, h, w)

        return BatchedImages(
            data,
            image_sizes=tuple(
                self.compute_output_shape(image_size) for image_size in x.image_sizes
            ),
        )
