##
##
##

from typing import Literal

from torch import nn

from deepsight.modules import FFN, Module, attention
from deepsight.typing import Tensor

from ._config import Config


class Encoder(Module):
    """The visual encoder of DETR model."""

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.num_encoder_layers)
        ])

        self.post_layer_norm = (
            nn.LayerNorm(config.embedding_dim)
            if config.encoder_post_norm
            else nn.Identity()
        )

    def __call__(
        self,
        features: Tensor[Literal["B HW D"], float],
        pos_embed: Tensor[Literal["B HW D"], float],
        mask: attention.Mask | None = None,
    ) -> Tensor[Literal["B HW D"], float]:
        """Forward pass through the encoder.

        Args:
            features: The image features extracted by the backbone.
            pos_embed: The positional embeddings for the image features.
            mask: The attention mask.

        Returns:
            The encoded image features.
        """
        for layer in self.layers:
            features = layer(features, pos_embed, mask=mask)

        return self.post_layer_norm(features)


class EncoderLayer(Module):
    """Encoder layer of DETR model."""

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.attn = attention.MultiHeadAttentionWithPos(
            qkv_generator=attention.LinearQKVGeneratorWithPrePosAddition(
                query_dim=config.embedding_dim,
                key_dim=config.embedding_dim,
                value_dim=config.embedding_dim,
                hidden_dim=config.embedding_dim,
                num_heads=config.num_heads,
            ),
            mechanism=attention.ScaledDotProductAttention(dropout=config.attn_dropout),
            out_dim=config.embedding_dim,
            qkv_dropout=0.0,
            q_norm=None,
            k_norm=None,
            out_dropout=config.proj_dropout,
        )
        self.attn_norm = nn.LayerNorm(config.embedding_dim)

        self.ffn = FFN(
            input_dim=config.embedding_dim,
            hidden_dim=config.ffn_dim,
            output_dim=config.embedding_dim,
            num_layers=config.ffn_num_layers,
            dropout=config.ffn_dropout,
            activation=nn.ReLU,
        )
        self.ffn_norm = nn.LayerNorm(config.embedding_dim)

    def __call__(
        self,
        x: Tensor[Literal["B L D"], float],
        pos_embed: Tensor[Literal["B L D"], float],
        mask: attention.Mask | None,
    ) -> Tensor[Literal["B L D"], float]:
        attn_x = self.attn(
            query=x,
            key=x,
            value=x,
            query_pos=pos_embed,
            key_pos=pos_embed,
            mask=mask,
        )
        x = self.attn_norm(x + attn_x)

        ffn_x = self.ffn(x)
        x = self.ffn_norm(x + ffn_x)

        return x
