##
##
##

from typing import Literal

import torch
from torch import nn

from deepsight.layers import FFN, attention
from deepsight.typing import Tensor

from ._config import Config


class Decoder(nn.Module):
    """The transformer decoder of DETR model."""

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.num_decoder_layers)
        ])

        self.post_layer_norm = (
            nn.LayerNorm(config.embedding_dim)
            if config.decoder_post_norm
            else nn.Identity()
        )

    def forward(
        self,
        query: Tensor[Literal["B Q D"], float],
        memory: Tensor[Literal["B HW D"], float],
        query_pos: Tensor[Literal["B Q D"], float],
        memory_pos: Tensor[Literal["B HW D"], float],
        mask: attention.Mask | None = None,
    ) -> Tensor[Literal["L B Q D"], float]:
        outputs = []
        for layer in self.layers:
            query = layer(query, memory, query_pos, memory_pos, mask=mask)
            outputs.append(query)

        return self.post_layer_norm(torch.stack(outputs))

    def __call__(
        self,
        query: Tensor[Literal["B Q D"], float],
        memory: Tensor[Literal["B HW D"], float],
        query_pos: Tensor[Literal["B Q D"], float],
        memory_pos: Tensor[Literal["B HW D"], float],
        mask: attention.Mask | None = None,
    ) -> Tensor[Literal["L B Q D"], float]:
        """Forward pass through the decoder.

        Args:
            query: The object queries.
            memory: The encoded image features.
            query_pos: The positional embeddings for the object queries.
            memory_pos: The positional embeddings for the image features.
            mask: The attention mask.

        Returns:
            The decoder outputs stacked along the first dimension. Each decoder output
            consists of the object queries after each decoder layer.
        """
        return super().__call__(query, memory, query_pos, memory_pos, mask=mask)


class DecoderLayer(nn.Module):
    """Decoder layer of DETR model."""

    def __init__(self, config: Config) -> None:
        super().__init__()

        # Self-attention.
        self.self_attn = attention.MultiHeadAttentionWithPos(
            qkv_generator=attention.LinearQKVGeneratorWithPrePosAddition(
                query_dim=config.embedding_dim,
                key_dim=config.embedding_dim,
                value_dim=config.embedding_dim,
                hidden_dim=config.embedding_dim,
                num_heads=config.num_heads,
            ),
            mechanism=attention.ScaledDotProductAttention(dropout=config.attn_dropout),
            out_dim=config.embedding_dim,
            out_dropout=config.proj_dropout,
        )
        self.self_attn_norm = nn.LayerNorm(config.embedding_dim)

        # Cross-attention.
        self.cross_attn = attention.MultiHeadAttentionWithPos(
            qkv_generator=attention.LinearQKVGeneratorWithPrePosAddition(
                query_dim=config.embedding_dim,
                key_dim=config.embedding_dim,
                value_dim=config.embedding_dim,
                hidden_dim=config.embedding_dim,
                num_heads=config.num_heads,
            ),
            mechanism=attention.ScaledDotProductAttention(dropout=config.attn_dropout),
            out_dim=config.embedding_dim,
            out_dropout=config.proj_dropout,
        )
        self.cross_attn_norm = nn.LayerNorm(config.embedding_dim)

        # Feed-forward network.
        self.ffn = FFN(
            input_dim=config.embedding_dim,
            hidden_dim=config.ffn_dim,
            output_dim=config.embedding_dim,
            num_layers=config.ffn_num_layers,
            dropout=config.ffn_dropout,
            activation=nn.ReLU,
        )
        self.ffn_norm = nn.LayerNorm(config.embedding_dim)

    def forward(
        self,
        query: Tensor[Literal["B Q D"], float],
        memory: Tensor[Literal["B HW D"], float],
        query_pos: Tensor[Literal["B Q D"], float],
        memory_pos: Tensor[Literal["B HW D"], float],
        mask: attention.Mask | None = None,
    ) -> Tensor[Literal["B Q D"], float]:
        sa_query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            mask=mask,
        )
        query = self.self_attn_norm(query + sa_query)

        ca_query = self.cross_attn(
            query=query,
            key=memory,
            value=memory,
            query_pos=query_pos,
            key_pos=memory_pos,
            mask=mask,
        )
        query = self.cross_attn_norm(query + ca_query)

        ffn_query = self.ffn(query)
        query = self.ffn_norm(query + ffn_query)

        return query

    def __call__(
        self,
        query: Tensor[Literal["B Q D"], float],
        memory: Tensor[Literal["B HW D"], float],
        query_pos: Tensor[Literal["B Q D"], float],
        memory_pos: Tensor[Literal["B HW D"], float],
        mask: attention.Mask | None = None,
    ) -> Tensor[Literal["B Q D"], float]:
        return super().__call__(query, memory, query_pos, memory_pos, mask=mask)
