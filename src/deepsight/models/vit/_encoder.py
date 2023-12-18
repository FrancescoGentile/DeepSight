##
##
##

import math
from typing import Literal, overload

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from deepsight import utils
from deepsight.layers import LayerScale
from deepsight.layers.vision import PatchEmbedding
from deepsight.structures.vision import BatchedImages, BatchedSequences
from deepsight.typing import Tensor

from ._config import Configs


class Encoder(nn.Module):
    """Encoder of the original ViT."""

    def __init__(self, configs: Configs) -> None:
        super().__init__()

        image_size = utils.to_2tuple(configs.image_size)
        patch_size = utils.to_2tuple(configs.patch_size)

        self.num_h_patches = math.ceil(image_size[0] / patch_size[0])
        self.num_w_patches = math.ceil(image_size[1] / patch_size[1])
        self.no_class_embedding = configs.no_class_embedding
        self.embed_dim = configs.embed_dim

        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            in_channels=configs.in_channels,
            embed_dim=configs.embed_dim,
            layer_norm_eps=None,
            bias=not configs.pre_normalize,
        )

        if configs.use_class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, configs.embed_dim))
        else:
            self.cls_token = None

        num_patches = self.num_h_patches * self.num_w_patches
        num_embeds = num_patches
        if configs.use_class_token and not configs.no_class_embedding:
            num_embeds += 1

        self.pos_embed = nn.Parameter(torch.zeros(1, num_embeds, configs.embed_dim))
        self.pos_dropout = nn.Dropout(configs.pos_embed_dropout)
        self.pre_layernorm = (
            nn.LayerNorm(configs.embed_dim, eps=configs.layer_norm_eps)
            if configs.pre_normalize
            else nn.Identity()
        )

        self.layers = nn.ModuleList([Layer(configs) for _ in range(configs.num_layers)])

        self.post_layernorm = nn.LayerNorm(
            configs.embed_dim, eps=configs.layer_norm_eps
        )

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def output_channels(self) -> int:
        """The number of output channels."""
        return self.embed_dim

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def forward(
        self, x: BatchedImages | Tensor[Literal["B C H W"], float]
    ) -> BatchedImages | Tensor[Literal["B D h w"], float]:
        x = self.patch_embed(x)  # (B, D, h, w)
        x_embed = self._add_pos_embeds(x)
        if isinstance(x_embed, BatchedSequences):
            out = x_embed.data
            mask = x_embed.mask[:, None, None]  # (B, 1, 1, (1) + hw)
        else:
            out = x_embed
            mask = None

        out: torch.Tensor = self.pre_layernorm(out)
        for layer in self.layers:
            out = layer(out, mask)

        out = self.post_layernorm(out)

        if self.cls_token is not None:
            out = out[:, 1:]  # remove cls token
        out = out.transpose(1, 2).reshape(*x.shape)  # (B, D, h, w)

        if isinstance(x, BatchedImages):
            return x.replace(data=out)

        return out

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    @overload
    def __call__(
        self, x: Tensor[Literal["B C H W"], float]
    ) -> Tensor[Literal["B D h w"], float]: ...

    @overload
    def __call__(self, x: BatchedImages) -> BatchedImages: ...

    def __call__(
        self, x: BatchedImages | Tensor[Literal["B C H W"], float]
    ) -> BatchedImages | Tensor[Literal["B D h w"], float]:
        return super().__call__(x)

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _add_pos_embeds(
        self, x: Tensor[Literal["B D h w"], float] | BatchedImages
    ) -> Tensor[Literal["B hw D"], float] | BatchedSequences:
        if isinstance(x, torch.Tensor):
            h, w = x.shape[-2:]
            pos_embed = _resize_pos_embeds(
                self.pos_embed,
                (self.num_h_patches, self.num_w_patches),
                (h, w),
                num_prefix_tokens=int(self.cls_token is not None),
            )
            sequences = None
            out = x.flatten(2).transpose(1, 2)  # (B, hw, D)
        else:
            pos_embeds = []
            for h, w in x.image_sizes:
                embeds = _resize_pos_embeds(
                    self.pos_embed,
                    (self.num_h_patches, self.num_w_patches),
                    (h, w),
                    num_prefix_tokens=int(self.cls_token is not None),
                )
                pos_embeds.append(embeds[0])

            pos_embed = pad_sequence(pos_embeds, batch_first=True)

            sequences = x.to_sequences()
            out = sequences.data

        to_cat = []
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(out.shape[0], -1, -1)
            to_cat.append(cls_token)

        if self.no_class_embedding:
            out = out + pos_embed
            out = torch.cat(to_cat + [out], dim=1)
        else:
            out = torch.cat(to_cat + [out], dim=1)
            out = out + pos_embed

        out = self.pos_dropout(out)
        if sequences is not None:
            if self.cls_token is not None:
                mask = sequences.mask  # (B, hw)
                mask = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1)
                out = BatchedSequences(out, mask=mask)
            else:
                out = sequences.replace(data=out)

        return out


class Layer(nn.Module):
    def __init__(self, configs: Configs) -> None:
        super().__init__()

        # Self-Attention
        self.sa_layernorm = nn.LayerNorm(configs.embed_dim, eps=configs.layer_norm_eps)
        self.sa = SelfAttention(configs)
        self.sa_layerscale = (
            LayerScale(configs.embed_dim, configs.layer_scale_init_value)
            if configs.layer_scale_init_value is not None
            else nn.Identity()
        )

        # Feed-Forward
        self.ffn_layernorm = nn.LayerNorm(configs.embed_dim, eps=configs.layer_norm_eps)
        hidden_dim = int(configs.embed_dim * configs.ffn_hidden_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(configs.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(configs.ffn_dropout),
            nn.Linear(hidden_dim, configs.embed_dim),
            nn.Dropout(configs.ffn_dropout),
        )
        self.ffn_layerscale = (
            LayerScale(configs.embed_dim, configs.layer_scale_init_value)
            if configs.layer_scale_init_value is not None
            else nn.Identity()
        )

    def forward(
        self,
        x: Tensor[Literal["B N D"], float],
        mask: Tensor[Literal[" B 1 1 N"], bool] | None,
    ) -> Tensor[Literal["B N D"], float]:
        sa_x = self.sa_layernorm(x)
        sa_x = self.sa(sa_x, mask)
        sa_x = self.sa_layerscale(sa_x)
        x = x + sa_x

        ffn_x = self.ffn_layernorm(x)
        ffn_x = self.ffn(ffn_x)
        ffn_x = self.ffn_layerscale(ffn_x)
        x = x + ffn_x

        return x


class SelfAttention(nn.Module):
    # ----------------------------------------------------------------------- #
    # Constructor
    # ----------------------------------------------------------------------- #

    def __init__(self, configs: Configs) -> None:
        super().__init__()

        self.num_heads = configs.num_heads
        self.head_dim = configs.embed_dim // configs.num_heads
        self.attn_dropout = configs.attn_dropout

        self.qkv_proj = nn.Linear(
            configs.embed_dim, configs.embed_dim * 3, bias=configs.qkv_bias
        )

        qk_layer_norm_eps = configs.layer_norm_eps if configs.qk_normalize else None
        self.q_norm = (
            nn.LayerNorm(configs.embed_dim, eps=qk_layer_norm_eps)
            if qk_layer_norm_eps is not None
            else nn.Identity()
        )
        self.k_norm = (
            nn.LayerNorm(configs.embed_dim, eps=qk_layer_norm_eps)
            if qk_layer_norm_eps is not None
            else nn.Identity()
        )
        self.qkv_dropout = nn.Dropout(configs.qkv_dropout)

        self.proj = nn.Linear(configs.embed_dim, configs.embed_dim)
        self.proj_dropout = nn.Dropout(configs.proj_dropout)

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def forward(
        self,
        x: Tensor[Literal["B N D"], float],
        mask: Tensor[Literal[" B 1 1 N"], bool] | None,
    ) -> Tensor[Literal["B N D"], float]:
        B, N, D = x.shape  # noqa

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, Dh)
        q, k, v = qkv.unbind(dim=0)  # (B, H, N, Dh)
        q, k = self.q_norm(q), self.k_norm(k)

        q = self.qkv_dropout(q)
        k = self.qkv_dropout(k)
        v = self.qkv_dropout(v)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )

        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        return self.proj_dropout(out)


# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #


def _resize_pos_embeds(
    pos_embeds: torch.Tensor,
    old_size: tuple[int, int],
    new_size: tuple[int, int],
    num_prefix_tokens: int = 1,
    interpolation: Literal["bilinear", "bicubic"] = "bicubic",
    antialias: bool = True,
) -> torch.Tensor:
    num_old_tokens = pos_embeds.shape[1]
    num_new_tokens = (new_size[0] * new_size[1]) + num_prefix_tokens
    if num_old_tokens == num_new_tokens:
        return pos_embeds

    if num_prefix_tokens > 0:
        prefix = pos_embeds[:, :num_prefix_tokens]
        pos_embeds = pos_embeds[:, num_prefix_tokens:]
    else:
        prefix = None

    embed_dim = pos_embeds.shape[-1]
    orig_dtype = pos_embeds.dtype
    pos_embeds = pos_embeds.float()

    pos_embeds = pos_embeds.reshape(1, old_size[0], old_size[1], embed_dim)
    pos_embeds = pos_embeds.permute(0, 3, 1, 2)
    pos_embeds = F.interpolate(
        pos_embeds, size=new_size, mode=interpolation, antialias=antialias
    )

    pos_embeds = pos_embeds.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    pos_embeds = pos_embeds.to(orig_dtype)

    if prefix is not None:
        pos_embeds = torch.cat([prefix, pos_embeds], dim=1)

    return pos_embeds
