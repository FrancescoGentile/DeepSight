##
##
##

from typing import Annotated

import torch
import torch.nn.functional as F  # noqa
from torch import Tensor, nn

from deepsight.structures import BatchedSequences

# ----------------------------------------------------------------------- #
# Encoder
# ----------------------------------------------------------------------- #


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.dropout = dropout

        self.qkv_prok = nn.Linear(embed_dim, embed_dim * 3)
        self.qk_pos_proj = nn.Linear(2 * embed_dim, 2 * embed_dim)
        self.sa_dropout = nn.Dropout(dropout)
        self.sa_layer_norm = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, embed_dim),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_layernorm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        inputs: BatchedSequences,
        positional_encodings: Annotated[Tensor, "B N 2D"],
    ) -> BatchedSequences:
        qkv = self.qkv_prok(inputs.data)  # (B, N, 3 * D)
        q, k, v = qkv.unbind(dim=-1)  # (B, N, D)

        qk_pos = self.qk_pos_proj(positional_encodings)  # (B, N, 2 * D)
        q_pos, k_pos = qk_pos.unbind(dim=-1)  # (B, N, D)

        q = q + q_pos
        k = k + k_pos

        attn_output = _multihead_attention(
            q, k, v, inputs.mask, self.num_heads, self.dropout, self.training
        )

        attn_output = self.sa_dropout(attn_output)
        attn_output = self.sa_layer_norm(inputs.data + attn_output)

        ffn_output = self.ffn(attn_output)
        ffn_output = self.ffn_dropout(ffn_output)
        ffn_output = self.ffn_layernorm(attn_output + ffn_output)

        return inputs.replace(ffn_output)


class TranformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(embed_dim, embed_dim * 4, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        inputs: BatchedSequences,
        positional_encodings: Annotated[Tensor, "B N 2D"],
    ) -> BatchedSequences:
        for layer in self.layers:
            inputs = layer(inputs, positional_encodings)

        return inputs

    def __call__(
        self,
        inputs: BatchedSequences,
        positional_encodings: Annotated[Tensor, "B N 2D"],
    ) -> BatchedSequences:
        return super().__call__(inputs, positional_encodings)


# ----------------------------------------------------------------------- #
# Decoder
# ----------------------------------------------------------------------- #


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        ffn_hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.dropout = dropout

        self.sa_qkv_proj = nn.Linear(q_dim, q_dim * 3)
        self.sa_qk_pos_proj = nn.Linear(4 * kv_dim, 2 * q_dim)
        self.sa_dropout = nn.Dropout(dropout)
        self.sa_layer_norm = nn.LayerNorm(q_dim)

        self.ca_q_proj = nn.Linear(q_dim, q_dim)
        self.ca_kv_proj = nn.Linear(kv_dim, 2 * q_dim)
        self.ca_q_pos_proj = nn.Linear(2 * kv_dim, q_dim)
        self.ca_k_pos_proj = nn.Linear(kv_dim, q_dim)
        self.ca_dropout = nn.Dropout(dropout)
        self.ca_layer_norm = nn.LayerNorm(q_dim)

        self.ffn = nn.Sequential(
            nn.Linear(q_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, q_dim),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_layernorm = nn.LayerNorm(q_dim)

    def forward(
        self,
        queries: BatchedSequences,  # (B, Q, D)
        box_positional_encodings: Annotated[Tensor, "B Q 4C"],
        center_positional_encodings: Annotated[Tensor, "B Q 2C"],
        key_values: BatchedSequences,
        key_value_positional_encodings: Annotated[Tensor, "B K C"],
    ) -> BatchedSequences:
        # Self-Attention
        qkv = self.sa_qkv_proj(queries.data)  # (B, Q, 3 * D)
        q, k, v = qkv.unbind(dim=-1)

        qk_pos = self.sa_qk_pos_proj(box_positional_encodings)  # (B, Q, 2 * D)
        q_pos, k_pos = qk_pos.unbind(dim=-1)

        q = q + q_pos
        k = k + k_pos

        sa_output = _multihead_attention(
            q, k, v, queries.mask, self.num_heads, self.dropout, self.training
        )

        sa_output = self.sa_dropout(sa_output)
        sa_output = self.sa_layer_norm(queries.data + sa_output)

        # Cross-Attention
        q = self.ca_q_proj(sa_output)  # (B, Q, D)
        kv = self.ca_kv_proj(key_values.data)  # (B, K, 2 * D)
        k, v = kv.unbind(dim=-1)

        q_pos = self.ca_q_pos_proj(center_positional_encodings)  # (B, Q, D)
        k_pos = self.ca_k_pos_proj(key_value_positional_encodings)  # (B, K, D)

        B, Q = q.shape[:2]  # noqa
        _, K = k.shape[:2]  # noqa

        q = q.view(B, Q, self.num_heads, -1)
        q_pos = q_pos.view(B, Q, self.num_heads, -1)
        q = torch.cat([q, q_pos], dim=-1).view(B, Q, -1)  # (B, Q, 2 * D)

        k = k.view(B, K, self.num_heads, -1)
        k_pos = k_pos.view(B, K, self.num_heads, -1)
        k = torch.cat([k, k_pos], dim=-1).view(B, K, -1)  # (B, K, 2 * D)

        ca_output = _multihead_attention(
            q, k, v, key_values.mask, self.num_heads, self.dropout, self.training
        )

        ca_output = self.ca_dropout(ca_output)
        ca_output = self.ca_layer_norm(sa_output + ca_output)

        # Feed-Forward
        ffn_output = self.ffn(ca_output)
        ffn_output = self.ffn_dropout(ffn_output)
        ffn_output = self.ffn_layernorm(ca_output + ffn_output)

        return queries.replace(ffn_output)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(q_dim, kv_dim, q_dim * 4, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        queries: BatchedSequences,
        box_positional_encodings: Annotated[Tensor, "B Q 4D"],
        center_positional_encodings: Annotated[Tensor, "B Q 2D"],
        key_values: BatchedSequences,
        key_value_positional_encodings: Annotated[Tensor, "B K D"],
    ) -> BatchedSequences:
        for layer in self.layers:
            queries = layer(
                queries,
                box_positional_encodings,
                center_positional_encodings,
                key_values,
                key_value_positional_encodings,
            )

        return queries

    def __call__(
        self,
        queries: BatchedSequences,
        box_positional_encodings: Annotated[Tensor, "B Q 4D"],
        center_positional_encodings: Annotated[Tensor, "B Q 2D"],
        key_values: BatchedSequences,
        key_value_positional_encodings: Annotated[Tensor, "B K D"],
    ) -> BatchedSequences:
        return super().__call__(
            queries,
            box_positional_encodings,
            center_positional_encodings,
            key_values,
            key_value_positional_encodings,
        )


# ----------------------------------------------------------------------- #
# Private helper functions
# ----------------------------------------------------------------------- #


def _multihead_attention(
    queries: Annotated[Tensor, "B Q D"],
    keys: Annotated[Tensor, "B K D", float],
    values: Annotated[Tensor, "B K D", float],
    key_padding_mask: Annotated[Tensor, "B K", bool] | None,
    num_heads: int,
    dropout: float,
    training: bool,
) -> Annotated[Tensor, "B Q D", float]:
    B, Q = queries.shape[:2]  # noqa
    _, K = keys.shape[:2]  # noqa

    queries = queries.view(B, Q, num_heads, -1).transpose(1, 2)  # (B, H, Q, Dh)
    keys = keys.view(B, K, num_heads, -1).transpose(1, 2)  # (B, H, K, Dh)
    values = values.view(B, K, num_heads, -1).transpose(1, 2)  # (B, H, K, Dh)

    attn = queries @ keys.transpose(-2, -1)  # (B, H, Q, K)
    attn = attn / (keys.shape[-1] ** 0.5)

    if key_padding_mask is not None:
        attn = attn.masked_fill_(key_padding_mask[:, None, None], -torch.inf)
    attn = attn.softmax(dim=-1)
    attn = F.dropout(attn, p=dropout, training=training)  # (B, H, Q, K)

    output = attn @ values  # (B, H, Q, Dh)
    output = output.transpose(1, 2).contiguous()  # (B, Q, H, Dh)
    output = output.view(B, Q, -1)  # (B, Q, H * Dh) = (B, Q, D)

    return output
