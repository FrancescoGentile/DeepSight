##
##
##

from typing import Literal, overload

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from deepsight import utils
from deepsight.structures.vision import BatchedImages, BatchedSequences
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

    def compute_output_shape(
        self, input_shape: tuple[int, int] | torch.Size
    ) -> tuple[int, int]:
        return (
            (input_shape[0] + self.patch_size[0] - 1) // self.patch_size[0],
            (input_shape[1] + self.patch_size[1] - 1) // self.patch_size[1],
        )

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
            return BatchedImages(
                data,
                image_sizes=tuple(
                    self.compute_output_shape(image_size)
                    for image_size in x.image_sizes
                ),
            )

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


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embedding."""

    def __init__(
        self,
        embed_dim: int,
        num_patches: int | tuple[int, int],
        num_prefix_embedding: int,
        pos_dropout: float = 0.0,
        interpolation: Literal["bilinear", "bicubic"] = "bicubic",
        antialias: bool = True,
    ) -> None:
        """Initializes the learned positional embedding.

        Args:
            embed_dim: The embedding dimension.
            num_patches: The number of patches to embed for each spatial
                dimension (h, w). If an integer is provided, the same number of patches
                will be used for both spatial dimensions.
            num_prefix_embedding: The number of prefix embeddings to use. Prefix
                embeddings can be used to embed tokens that are prepended to the
                input patch sequence, like class tokens or register tokens.
            pos_dropout: The dropout probability to use on the positional
                embeddings.
            interpolation: The interpolation mode to use when resizing the
                positional embeddings.
            antialias: Whether to use antialiasing when resizing the positional
                embeddings.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_h_patches, self.num_w_patches = utils.to_2tuple(num_patches)
        self.num_prefix_embedding = num_prefix_embedding
        self.interpolation = interpolation
        self.antialias = antialias

        total_embeds = self.num_h_patches * self.num_w_patches
        total_embeds += self.num_prefix_embedding
        self.embeddings = nn.Parameter(torch.randn(1, total_embeds, embed_dim) * 0.02)
        self.pos_dropout = nn.Dropout(pos_dropout)

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def forward(
        self,
        x: Tensor[Literal["B D h w"], float] | BatchedImages,
        prefix_tokens: Tensor[Literal["1 _ D"], float]
        | list[Tensor[Literal["1 _ D"], float]]
        | None = None,
    ) -> Tensor[Literal["B L D"], float] | BatchedSequences:
        if isinstance(x, torch.Tensor):
            h, w = x.shape[-2:]
            pos_embeds = self._resize((h, w))
            out = x.flatten(2).transpose(1, 2)  # (B, hw, D)
        else:
            pos_embeds_cache: dict[tuple[int, int], torch.Tensor] = {}
            pos_embeds_list = []
            for h, w in x.image_sizes:
                if (h, w) not in pos_embeds_cache:
                    pos_embeds_cache[(h, w)] = self._resize((h, w))

                pos_embeds_list.append(pos_embeds_cache[(h, w)])

            if len(pos_embeds_cache) > 1:
                pos_embeds = pad_sequence(
                    [embed[0] for embed in pos_embeds_list], batch_first=True
                )
                out = pad_sequence(
                    [image.data.flatten(1).T for image in x],
                    batch_first=True,
                )
            else:
                # all the images have the same size, so we can avoid padding
                pos_embeds = pos_embeds_list[0]
                out = x.data.flatten(2).transpose(1, 2)

        match prefix_tokens:
            case None:
                num_prefix_tokens = 0
                prefix_tokens = []
            case torch.Tensor():
                num_prefix_tokens = prefix_tokens.shape[1]
                prefix_tokens = [prefix_tokens.expand(out.shape[0], -1, -1)]
            case list():
                num_prefix_tokens = sum(t.shape[1] for t in prefix_tokens)
                prefix_tokens = [t.expand(out.shape[0], -1, -1) for t in prefix_tokens]

        if self.num_prefix_embedding > 0:
            out = torch.cat(prefix_tokens + [out], dim=1)
            out = out + pos_embeds
        else:
            out = out + pos_embeds
            out = torch.cat(prefix_tokens + [out], dim=1)

        out = self.pos_dropout(out)
        if isinstance(x, torch.Tensor):
            return out
        else:
            seq_lengths = tuple(num_prefix_tokens + h * w for h, w in x.image_sizes)
            return BatchedSequences(out, sequence_lengths=seq_lengths)

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    @overload
    def __call__(
        self,
        x: Tensor[Literal["B D h w"], float],
        prefix_tokens: Tensor[Literal["1 _ D"], float]
        | list[Tensor[Literal["1 _ D"], float]]
        | None = None,
    ) -> Tensor[Literal["B L D"], float]: ...

    @overload
    def __call__(
        self,
        x: BatchedImages,
        prefix_tokens: Tensor[Literal["1 _ D"], float]
        | list[Tensor[Literal["1 _ D"], float]]
        | None = None,
    ) -> BatchedSequences: ...

    def __call__(
        self,
        x: Tensor[Literal["B D h w"], float] | BatchedImages,
        prefix_tokens: Tensor[Literal["1 _ D"], float]
        | list[Tensor[Literal["1 _ D"], float]]
        | None = None,
    ) -> Tensor[Literal["B L D"], float] | BatchedSequences:
        return super().__call__(x, prefix_tokens)

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _resize(self, new_size: tuple[int, int]) -> Tensor[Literal["1 L D"], float]:
        if new_size == (self.num_h_patches, self.num_w_patches):
            return self.embeddings

        if self.num_prefix_embedding > 0:
            prefix = self.embeddings[:, : self.num_prefix_embedding]
            pos_embeds = self.embeddings[:, self.num_prefix_embedding :]
        else:
            pos_embeds = self.embeddings
            prefix = None

        embed_dim = pos_embeds.shape[-1]
        orig_dtype = pos_embeds.dtype
        pos_embeds = pos_embeds.float()

        pos_embeds = pos_embeds.reshape(
            1, self.num_h_patches, self.num_w_patches, embed_dim
        )
        pos_embeds = pos_embeds.permute(0, 3, 1, 2)
        pos_embeds = F.interpolate(
            pos_embeds,
            size=new_size,
            mode=self.interpolation,
            antialias=self.antialias,
        )

        pos_embeds = pos_embeds.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
        pos_embeds = pos_embeds.to(orig_dtype)

        if prefix is not None:
            pos_embeds = torch.cat([prefix, pos_embeds], dim=1)

        return pos_embeds
