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
from deepsight.structures.vision import BatchedImages, BatchedSequences
from deepsight.typing import Tensor

from ._config import Config


class Encoder(nn.Module):
    """Vision Transformer Encoder."""

    def __init__(self, config: Config) -> None:
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

        self.post_layernorm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)

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
    def forward(
        self,
        x: Tensor[Literal["B C H W"], float],
        return_layers: int | Iterable[int],
        apply_post_layernorm: bool,
    ) -> list[Tensor[Literal["B L D"], float]]: ...

    @overload
    def forward(
        self,
        x: BatchedImages,
        return_layers: int | Iterable[int],
        apply_post_layernorm: bool,
    ) -> list[BatchedSequences]: ...

    def forward(
        self,
        x: BatchedImages | Tensor[Literal["B C H W"], float],
        return_layers: int | Iterable[int] = -1,
        apply_post_layernorm: bool = True,
    ) -> list[BatchedSequences] | list[Tensor[Literal["B L D"], float]]:
        """Forward pass through the encoder.

        Args:
            x: The input data.
            return_layers: The indices of the layers to return.
            apply_post_layernorm: Whether to apply post layer normalization. If multiple
                layers are returned, this will be applied to each layer.
        """
        return_layers = utils.to_tuple(return_layers)
        take_indices = {i if i >= 0 else len(self.layers) + i for i in return_layers}

        x = self.patch_embed(x)  # (B, D, h, w)

        prefix_tokens = []
        if self.cls_token is not None:
            prefix_tokens.append(self.cls_token)
        if self.register_tokens is not None:
            prefix_tokens.append(self.register_tokens)

        x_embed = self.pos_embed(x, prefix_tokens)  # (B, (cls + reg) + hw, D)
        if isinstance(x_embed, BatchedSequences):
            out = x_embed.data
            # The attention mask in BatchedSequences has True for the padding tokens
            # and False for the valid tokens. The boolean attention mask used by
            # `scaled_dot_product_attention` instead should be the opposite (True for
            # elements that take part in the attention and False for elements that are
            # masked out). Therefore, we need to invert the mask.
            mask = ~x_embed.mask[:, None, None]
        else:
            out = x_embed
            mask = None

        tmp: torch.Tensor = self.pre_layernorm(out)

        outputs: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            tmp = layer(tmp, mask)
            if i in take_indices:
                outputs.append(tmp)

        if apply_post_layernorm:
            outputs = [self.post_layernorm(out) for out in outputs]

        if isinstance(x_embed, BatchedSequences):
            return [x_embed.replace(data=out) for out in outputs]
        else:
            return outputs

    @overload
    def extract_feature_maps(
        self,
        inputs: Tensor[Literal["B C H W"], float],
        layer_outputs: Iterable[Tensor[Literal["B L D"], float]],
    ) -> list[Tensor[Literal["B D h w"], float]]: ...

    @overload
    def extract_feature_maps(
        self,
        inputs: BatchedImages,
        layer_outputs: Iterable[BatchedSequences],
    ) -> list[BatchedImages]: ...

    def extract_feature_maps(
        self,
        inputs: BatchedImages | Tensor[Literal["B C H W"], float],
        layer_outputs: Iterable[BatchedSequences]
        | Iterable[Tensor[Literal["B L D"], float]],
    ) -> list[BatchedImages] | list[Tensor[Literal["B D h w"], float]]:
        """Extract the feature maps from the encoder.

        The output of each encoder layer is a sequence of vectors corresponding to the
        patches features and, possibly, the class and register tokens. This method
        removes the class and register tokens and reshapes the output to a 4D tensor
        with the same spatial proportions as the input image.

        Args:
            inputs: The original input data to the encoder.
            layer_outputs: The outputs of the encoder layers.
        """
        if isinstance(inputs, BatchedImages):
            image_sizes = tuple(
                self.patch_embed.compute_output_shape(image_size)
                for image_size in inputs.image_sizes
            )

            outputs = []
            for layer_output in layer_outputs:
                assert isinstance(layer_output, BatchedSequences)
                sequences = layer_output.unbatch()
                images = [
                    seq[self.num_prefix_tokens :].T.view(-1, h, w)
                    for seq, (h, w) in zip(sequences, image_sizes, strict=True)
                ]
                outputs.append(BatchedImages.batch(images))

            return outputs
        else:
            output_size = self.patch_embed.compute_output_shape(inputs.shape[-2:])
            h, w = output_size

            outputs = []
            for layer_output in layer_outputs:
                assert isinstance(layer_output, torch.Tensor)
                data = layer_output[:, self.num_prefix_tokens :]
                data = data.view(-1, h, w, self.embed_dim)
                data = data.permute(0, 3, 1, 2)

                outputs.append(data)

            return outputs

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    @overload
    def __call__(
        self, x: Tensor[Literal["B C H W"], float]
    ) -> Tensor[Literal["B L D"], float]: ...

    @overload
    def __call__(self, x: BatchedImages) -> BatchedSequences: ...

    def __call__(
        self, x: BatchedImages | Tensor[Literal["B C H W"], float]
    ) -> BatchedSequences | Tensor[Literal["B L D"], float]:
        return super().__call__(x)[-1]


class Layer(nn.Module):
    def __init__(self, configs: Config) -> None:
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

    def __init__(self, configs: Config) -> None:
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
