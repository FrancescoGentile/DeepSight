##
##
##

import math
from collections.abc import Iterable
from typing import Literal, overload

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from deepsight import utils
from deepsight.layers import LayerScale
from deepsight.layers.vision import LearnedPositionalEmbedding, PatchEmbedding
from deepsight.structures import BatchedImages, BatchedSequences
from deepsight.typing import Tensor

from ._config import EncoderConfig


class Encoder(nn.Module):
    """Vision Transformer Encoder."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()

        image_size = utils.to_2tuple(config.image_size)
        patch_size = utils.to_2tuple(config.patch_size)

        self.embed_dim = config.embed_dim

        self.num_h_patches = math.ceil(image_size[0] / patch_size[0])
        self.num_w_patches = math.ceil(image_size[1] / patch_size[1])
        self.num_prefix_tokens = 1 if config.use_class_token else 0
        self.num_prefix_tokens += config.num_register_tokens

        self.use_prefix_embedding = config.use_prefix_embedding

        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            layer_norm_eps=None,
            bias=not config.pre_normalize,
        )

        if config.use_class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        else:
            self.cls_token = None

        if config.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, config.num_register_tokens, config.embed_dim)
            )
        else:
            self.register_tokens = None

        self.pos_embed = LearnedPositionalEmbedding(
            embed_dim=config.embed_dim,
            num_patches=(self.num_h_patches, self.num_w_patches),
            num_prefix_embedding=self.num_prefix_tokens
            if config.use_prefix_embedding
            else 0,
            pos_dropout=config.pos_embed_dropout,
        )

        self.pre_layernorm = (
            nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
            if config.pre_normalize
            else nn.Identity()
        )

        self.layers = nn.ModuleList([Layer(config) for _ in range(config.num_layers)])

        self.post_layernorm = (
            nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
            if config.post_normalize
            else nn.Identity()
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

    @overload
    def get_intermediate_outputs(
        self,
        images: BatchedImages,
        return_layers: int | Iterable[int] = -1,
        apply_post_layernorm: bool = True,
        remove_prefix_tokens: Literal[True] = True,
    ) -> tuple[BatchedImages, ...]: ...

    @overload
    def get_intermediate_outputs(
        self,
        images: BatchedImages,
        return_layers: int | Iterable[int] = -1,
        apply_post_layernorm: bool = True,
        remove_prefix_tokens: Literal[False] = False,
    ) -> tuple[BatchedSequences, ...]: ...

    def get_intermediate_outputs(
        self,
        images: BatchedImages,
        return_layers: int | Iterable[int] = -1,
        apply_post_layernorm: bool = True,
        remove_prefix_tokens: bool = True,
    ) -> tuple[BatchedImages, ...] | tuple[BatchedSequences, ...]:
        """Get the intermediate outputs of the encoder.

        Args:
            images: The input images.
            return_layers: The indices of the layers to return. The indices can be
                negative, in which case they are interpreted as relative to the end of
                the list of layers.
            apply_post_layernorm: Whether to apply post layer normalization. If multiple
                layers are returned, this will be applied to each layer.
            remove_prefix_tokens: Whether to remove the prefix tokens from the output.
                If the prefix tokens are not removed, the output of each layer will
                consist of a sequence of vectors corresponding to the patches features
                and, possibly, the class and register tokens. If the prefix tokens are
                removed, the patches features are reshaped to a 4D tensor (B, C, H, W)
                with the same spatial proportions as the input images.

        Returns:
            The intermediate outputs of the encoder.
        """
        return_layers = utils.to_tuple(return_layers)
        take_indices = {i if i >= 0 else len(self.layers) + i for i in return_layers}

        x = self.patch_embed(images)  # (B, D, h, w)

        prefix_tokens = []
        if self.cls_token is not None:
            prefix_tokens.append(self.cls_token)
        if self.register_tokens is not None:
            prefix_tokens.append(self.register_tokens)

        x_embed = self.pos_embed(x, prefix_tokens)  # (B, (cls + reg) + hw, D)
        if not x_embed.is_padded():
            out = x_embed.data
            mask = None  # since there are no padding tokens, no mask is needed
        else:
            out = x_embed.data
            mask = x_embed.padding_mask[:, None, None]  # (B, 1, 1, (cls + reg) + hw)

        layer_output: torch.Tensor = self.pre_layernorm(out)
        torch_outputs: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            layer_output = layer(layer_output, mask)
            if i in take_indices:
                torch_outputs.append(layer_output)

        if apply_post_layernorm:
            torch_outputs = [self.post_layernorm(out) for out in torch_outputs]

        outputs = tuple(x_embed.replace(data=out) for out in torch_outputs)
        if remove_prefix_tokens:
            image_sizes = tuple(
                self.patch_embed.compute_output_shape(image_size)
                for image_size in images.image_sizes
            )

            outputs = tuple(
                self._extract_feature_map(seq, image_sizes=image_sizes)
                for seq in outputs
            )

        return outputs

    def forward(
        self,
        images: BatchedImages,
    ) -> BatchedSequences:
        return self.get_intermediate_outputs(
            images,
            return_layers=-1,
            apply_post_layernorm=True,
            remove_prefix_tokens=False,
        )[-1]

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __call__(self, x: BatchedImages) -> BatchedSequences:
        return super().__call__(x)

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _extract_feature_map(
        self,
        sequences: BatchedSequences,
        image_sizes: tuple[tuple[int, int], ...],
    ) -> BatchedImages:
        if not sequences.is_padded():
            h, w = image_sizes[0]
            data = sequences.data[:, self.num_prefix_tokens :]
            data = data.view(-1, h, w, self.embed_dim)
            data = data.permute(0, 3, 1, 2)

            return BatchedImages(data)
        else:
            images = []
            for seq, (h, w) in zip(sequences.unbatch(), image_sizes, strict=True):
                data = seq[self.num_prefix_tokens :].T.view(-1, h, w)
                images.append(data)

            return BatchedImages.batch(images)


class Layer(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()

        # Self-Attention
        self.sa_layernorm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.sa = SelfAttention(config)
        self.sa_layerscale = (
            LayerScale(config.embed_dim, config.layer_scale_init_value)
            if config.layer_scale_init_value is not None
            else nn.Identity()
        )

        # Feed-Forward
        self.ffn_layernorm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        hidden_dim = int(config.embed_dim * config.ffn_hidden_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.ffn_dropout),
            nn.Linear(hidden_dim, config.embed_dim),
            nn.Dropout(config.ffn_dropout),
        )
        self.ffn_layerscale = (
            LayerScale(config.embed_dim, config.layer_scale_init_value)
            if config.layer_scale_init_value is not None
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

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()

        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.attn_dropout = config.attn_dropout

        self.qkv_proj = nn.Linear(
            config.embed_dim, config.embed_dim * 3, bias=config.qkv_bias
        )

        qk_layer_norm_eps = config.layer_norm_eps if config.qk_normalize else None
        self.q_norm = (
            nn.LayerNorm(config.embed_dim, eps=qk_layer_norm_eps)
            if qk_layer_norm_eps is not None
            else nn.Identity()
        )
        self.k_norm = (
            nn.LayerNorm(config.embed_dim, eps=qk_layer_norm_eps)
            if qk_layer_norm_eps is not None
            else nn.Identity()
        )
        self.qkv_dropout = nn.Dropout(config.qkv_dropout)

        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.proj_dropout = nn.Dropout(config.proj_dropout)

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
