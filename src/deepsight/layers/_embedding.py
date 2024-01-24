##
##
##

from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from deepsight import utils
from deepsight.structures import BatchedImages
from deepsight.typing import Tensor


class LearnedImagePositionEmbedding(nn.Module):
    """Learned position embedding for images."""

    def __init__(
        self,
        embed_dim: int,
        num_patches: int | tuple[int, int],
        init_scale: float = 1.0,
        interpolation: Literal["bilinear", "bicubic"] = "bicubic",
        antialias: bool = True,
    ) -> None:
        """Initialize the learned positional embedding.

        Args:
            embed_dim: The embedding dimension.
            num_patches: The number of patches to embed for each spatial dimension
                (h, w). If an integer is provided, the same number of patches will be
                used for both spatial dimensions.
            init_scale: The scale factor to use when initializing the positional
                embeddings.
            interpolation: The interpolation mode to use when resizing the positional
                embeddings.
            antialias: Whether to use antialiasing when resizing the positional
                embeddings.
        """
        super().__init__()

        self.embed_dim = embed_dim
        num_h_patches, num_w_patches = utils.to_2tuple(num_patches)
        self.interpolation = interpolation
        self.antialias = antialias

        embeds = torch.randn(1, embed_dim, num_h_patches, num_w_patches)
        self.embeddings = nn.Parameter(embeds * init_scale)

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def forward(self, x: BatchedImages) -> BatchedImages:
        if not x.is_padded():
            h, w = x.shape[-2:]
            pos_embeds = self._resize((h, w)).expand(len(x), -1, -1, -1)
            return x.new_with(data=pos_embeds)
        else:
            pos_embeds_cache: dict[tuple[int, int], torch.Tensor] = {}
            pos_embeds_list = []
            for h, w in x.image_sizes:
                if (h, w) not in pos_embeds_cache:
                    pos_embeds_cache[(h, w)] = self._resize((h, w))[0]

                pos_embeds_list.append(pos_embeds_cache[(h, w)])

            return BatchedImages.batch(pos_embeds_list)

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __call__(self, x: BatchedImages) -> BatchedImages:
        return super().__call__(x)

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _resize(self, new_size: tuple[int, int]) -> Tensor[Literal["1 C H W"], float]:
        if new_size == self.embeddings.shape[-2:]:
            return self.embeddings

        orig_dtype = self.embeddings.dtype
        pos_embeds = self.embeddings.float()

        pos_embeds = F.interpolate(
            pos_embeds,
            size=new_size,
            mode=self.interpolation,
            antialias=self.antialias,
        )

        pos_embeds = pos_embeds.to(orig_dtype)
        return pos_embeds


# --------------------------------------------------------------------------- #
# Sinusoidal Embedding
# --------------------------------------------------------------------------- #

# Source code taken and modified from the detrex repository:
# https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/position_embedding.py


class SinusoidalImagePositionEmbedding(nn.Module):
    """Sinusoidal position embedding for images."""

    def __init__(
        self,
        embed_dim: int,
        temperature: float = 10000,
        normalize: bool = False,
        offset: float = 0.0,
        eps: float = 1e-6,
        scale: float = 2 * torch.pi,
        xy_order: Literal["xy", "yx"] = "xy",
    ) -> None:
        """Initialize the sinusoidal positional embedding.

        Args:
            embed_dim: The embedding dimension. Note that this is the total
                embedding dimension, not the dimension for each spatial dimension
                (the dimension for each spatial dimension will be embed_dim // 2).
            temperature: The temperature to use for the sinusoidal function.
            normalize: Whether to normalize the positional embeddings.
            offset: An offset to add to the positional embeddings. This is used
                only if `normalize` is True.
            eps: A small value to add to the denominator when normalizing the
                positional embeddings. This is used only if `normalize` is True.
            scale: A scale factor applied to the positional embeddings. This is
                used only if `normalize` is True.
            xy_order: The order of the positional embeddings. If "xy", then the
                positional embeddings will be in the order (x, y). If "yx", then
                the positional embeddings will be in the order (y, x).
        """
        super().__init__()

        if embed_dim % 2 != 0:
            raise ValueError(
                f"The embedding dimension must be divisible by 2, but got {embed_dim}."
            )

        self.embed_dim = embed_dim
        self.temperature = temperature
        self.scale = scale
        self.eps = eps
        self.offset = offset
        self.normalize = normalize
        self.xy_order: Literal["xy", "yx"] = xy_order

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def forward(self, images: BatchedImages) -> BatchedImages:
        B, _, H, W = images.shape  # noqa: N806

        if not images.is_padded():
            y_embed = torch.arange(H, device=images.device, dtype=images.dtype) + 1
            y_embed = y_embed.view(1, -1, 1).expand(B, -1, W)

            x_embed = torch.arange(W, device=images.device, dtype=images.dtype) + 1
            x_embed = x_embed.view(1, 1, -1).expand(B, H, -1)
        else:
            mask = images.padding_mask  # (B, H, W)
            y_embed = mask.cumsum(dim=1, dtype=images.dtype)
            x_embed = mask.cumsum(dim=2, dtype=images.dtype)

        if self.normalize:
            y_embed = y_embed + self.offset
            y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale

            x_embed = x_embed + self.offset
            x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale

        pos_dim = self.embed_dim // 2
        dim_t = torch.arange(pos_dim, device=images.device, dtype=images.dtype)
        dim_t = 2 * dim_t.div(2, rounding_mode="floor") / pos_dim
        dim_t = self.temperature**dim_t

        x = x_embed.unsqueeze(-1) / dim_t
        y = y_embed.unsqueeze(-1) / dim_t

        x = torch.stack((x[:, :, :, 0::2].sin(), x[:, :, :, 1::2].cos()), dim=4)
        x = x.view(B, H, W, -1)

        y = torch.stack((y[:, :, :, 0::2].sin(), y[:, :, :, 1::2].cos()), dim=4)
        y = y.view(B, H, W, -1)

        match self.xy_order:
            case "xy":
                out = torch.cat((x, y), dim=3)
            case "yx":
                out = torch.cat((y, x), dim=3)

        out = out.permute(0, 3, 1, 2)
        return images.new_with(data=out)

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __call__(self, images: BatchedImages) -> BatchedImages:
        return super().__call__(images)
