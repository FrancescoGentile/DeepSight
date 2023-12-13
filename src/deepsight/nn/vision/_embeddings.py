##
##
##

from typing import Literal, overload

import torch.nn.functional as F  # noqa: N812
from torch import nn

from deepsight import utils
from deepsight.structures.vision import BatchedImages
from deepsight.typing import Tensor


class PatchEmbedding(nn.Module):
    """Extracts patches from images and embeds them."""

    def __init__(
        self,
        patch_size: int | tuple[int, int],
        in_channels: int,
        embed_dim: int,
        layer_norm_eps: float | None = 1e-6,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self._patch_size = utils.to_2tuple(patch_size)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=self._patch_size,
            stride=self._patch_size,
            bias=bias,
        )
        self.norm = (
            nn.LayerNorm(embed_dim, eps=layer_norm_eps)
            if layer_norm_eps
            else nn.Identity()
        )

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def patch_size(self) -> tuple[int, int]:
        return self._patch_size

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def forward(
        self,
        x: Tensor[Literal["B C H W"], float] | BatchedImages,
    ) -> Tensor[Literal["B D h w"], float] | BatchedImages:
        H, W = x.shape[-2:]  # noqa: N806
        pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
        pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]

        data = x.data if isinstance(x, BatchedImages) else x
        if pad_h > 0 or pad_w > 0:
            data = F.pad(data, (0, pad_w, 0, pad_h))

        data = self.proj(data)  # (B, D, h, w)
        data = self.norm(data)  # (B, D, h, w)

        if isinstance(x, BatchedImages):
            new_image_sizes = tuple(
                ((h + pad_h) // self.patch_size[0], (w + pad_w) // self.patch_size[1])
                for h, w in x.image_sizes
            )
            return BatchedImages(data, image_sizes=new_image_sizes)

        return data

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    @overload
    def __call__(
        self,
        x: Tensor[Literal["B C H W"], float],
    ) -> Tensor[Literal["B D h w"], float]: ...

    @overload
    def __call__(
        self,
        x: BatchedImages,
    ) -> BatchedImages: ...

    def __call__(
        self,
        x: Tensor[Literal["B C H W"], float] | BatchedImages,
    ) -> Tensor[Literal["B D h w"], float] | BatchedImages:
        return super().__call__(x)
