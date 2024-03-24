# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Literal

import torch.nn.functional as F  # noqa: N812
from torch import nn

from deepsight.modules import Module
from deepsight.typing import Tensor

from ._mask import Mask
from ._mechanism import AttentionMechanism
from ._qkv import QKVGenerator, QKVGeneratorWithPos

# -------------------------------------------------------------------------- #
# Multi-head attention modules without positional embeddings
# -------------------------------------------------------------------------- #


class MultiHeadAttention(Module):
    """Multi-head attention module."""

    # ---------------------------------------------------------------------- #
    # Constructor
    # ---------------------------------------------------------------------- #

    def __init__(
        self,
        qkv_generator: QKVGenerator,
        mechanism: AttentionMechanism,
        out_dim: int | None = None,
        qkv_dropout: float = 0.0,
        q_norm: int | Callable[[int], nn.Module] | None = None,
        k_norm: int | Callable[[int], nn.Module] | None = None,
        out_dropout: float = 0.0,
    ) -> None:
        """Initialize the multi-head attention module.

        Args:
            qkv_generator: The query, key, and value generator.
            mechanism: The module used to compute the attention scores and
                aggregate the values to generate the output.
            out_dim: The output dimension. If passed, a linear projection is
                applied to the output of the attention module to generate the
                output tensor. If `None`, no projection is applied.
            qkv_dropout: The dropout rate applied to the output of the query,
                key, and value generator.
            q_norm: The query normalization module. If `None`, no normalization is
                applied. If an integer is passed, Lp norm with the given integer is
                applied. If a callable is passed, the callable is used to generate the
                normalization module by passing the input dimension.
            k_norm: The key normalization module. See `q_norm` for details.
            out_dropout: The dropout rate applied to the output of the
                attention module. If a projection is applied, the dropout is
                applied to the output of the projection.
        """
        super().__init__()

        self.qkv_generator = qkv_generator
        self.attention = mechanism
        self.out_proj = (
            nn.Linear(qkv_generator.head_dim * qkv_generator.num_heads, out_dim)
            if out_dim is not None
            else nn.Identity()
        )
        self.qkv_dropout = nn.Dropout(qkv_dropout)
        self.out_dropout = nn.Dropout(out_dropout)

        self.q_norm = q_norm(qkv_generator.head_dim) if callable(q_norm) else q_norm
        self.k_norm = k_norm(qkv_generator.head_dim) if callable(k_norm) else k_norm

    # ---------------------------------------------------------------------- #
    # Magic Methods
    # ---------------------------------------------------------------------- #

    def __call__(
        self,
        query: Tensor[Literal["B Q Dq"], float],
        key: Tensor[Literal["B K Dk"], float],
        value: Tensor[Literal["B K Dv"], float],
        mask: Mask | None = None,
    ) -> Tensor[Literal["B Q D"], float]:
        """Performs a forward pass through the multi-head attention module.

        Args:
            query: The input tensor used to generate the query.
            key: The input tensor used to generate the key.
            value: The input tensor used to generate the value.
            mask: The attention mask to apply to the attention scores. If `None`,
                no mask is applied.

        Returns:
            The output tensor.
        """
        query, key, value = self.qkv_generator(query, key, value)

        match self.q_norm:
            case None:
                pass
            case int():
                query = F.normalize(query, p=self.q_norm, dim=-1)
            case _:
                query = self.q_norm(query)

        match self.k_norm:
            case None:
                pass
            case int():
                key = F.normalize(key, p=self.k_norm, dim=-1)
            case _:
                key = self.k_norm(key)

        query, key, value = (
            self.qkv_dropout(query),
            self.qkv_dropout(key),
            self.qkv_dropout(value),
        )

        output = self.attention(query, key, value, mask=mask)
        B, H, K, D = output.shape  # noqa: N806
        output = output.transpose(1, 2).reshape(B, K, H * D)
        output = self.out_proj(output)
        output = self.out_dropout(output)

        return output


# -------------------------------------------------------------------------- #
# Multi-head attention modules with positional embeddings
# -------------------------------------------------------------------------- #


class MultiHeadAttentionWithPos(Module):
    """Multi-head attention module with positional embeddings."""

    # ---------------------------------------------------------------------- #
    # Constructor
    # ---------------------------------------------------------------------- #

    def __init__(
        self,
        qkv_generator: QKVGeneratorWithPos,
        mechanism: AttentionMechanism,
        out_dim: int | None = None,
        qkv_dropout: float = 0.0,
        q_norm: int | Callable[[int], nn.Module] | None = None,
        k_norm: int | Callable[[int], nn.Module] | None = None,
        out_dropout: float = 0.0,
    ) -> None:
        """Initialize the multi-head attention module.

        Args:
            qkv_generator: The query, key, and value generator.
            mechanism: The module used to compute the attention scores and
                aggregate the values to generate the output.
            out_dim: The output dimension. If passed, a linear projection is
                applied to the output of the attention module to generate the
                output tensor. If `None`, no projection is applied.
            qkv_dropout: The dropout rate applied to the output of the query,
                key, and value generator.
            q_norm: The query normalization module. If `None`, no normalization is
                applied. If an integer is passed, Lp norm with the given integer is
                applied. If a callable is passed, the callable is used to generate the
                normalization module by passing the input dimension.
            k_norm: The key normalization module. See `q_norm` for details.
            out_dropout: The dropout rate applied to the output of the
                attention module. If a projection is applied, the dropout is
                applied to the output of the projection.
        """
        super().__init__()

        self.qkv_generator = qkv_generator
        self.attention = mechanism
        self.out_proj = (
            nn.Linear(qkv_generator.head_dim * qkv_generator.num_heads, out_dim)
            if out_dim is not None
            else nn.Identity()
        )
        self.qkv_dropout = nn.Dropout(qkv_dropout)
        self.out_dropout = nn.Dropout(out_dropout)

        self.q_norm = q_norm(qkv_generator.head_dim) if callable(q_norm) else q_norm
        self.k_norm = k_norm(qkv_generator.head_dim) if callable(k_norm) else k_norm

    # ---------------------------------------------------------------------- #
    # Magic Methods
    # ---------------------------------------------------------------------- #

    def __call__(
        self,
        query: Tensor[Literal["B Q Dq"], float],
        key: Tensor[Literal["B K Dk"], float],
        value: Tensor[Literal["B K Dv"], float],
        query_pos: Tensor[Literal["B Q Dqp"], float],
        key_pos: Tensor[Literal["B K Dkp"], float],
        mask: Mask | None = None,
    ) -> Tensor[Literal["B Q D"], float]:
        """Performs a forward pass through the multi-head attention module.

        Args:
            query: The input tensor used to generate the query.
            key: The input tensor used to generate the key.
            value: The input tensor used to generate the value.
            query_pos: The positional embedding of the query.
            key_pos: The positional embedding of the key.
            mask: The attention mask to apply to the attention scores. If `None`,
                no mask is applied.

        Returns:
            The output tensor.
        """
        query, key, value = self.qkv_generator(query, key, value, query_pos, key_pos)

        match self.q_norm:
            case None:
                pass
            case int():
                query = F.normalize(query, p=self.q_norm, dim=-1)
            case _:
                query = self.q_norm(query)

        match self.k_norm:
            case None:
                pass
            case int():
                key = F.normalize(key, p=self.k_norm, dim=-1)
            case _:
                key = self.k_norm(key)

        query, key, value = (
            self.qkv_dropout(query),
            self.qkv_dropout(key),
            self.qkv_dropout(value),
        )

        output = self.attention(query, key, value, mask=mask)
        B, H, K, D = output.shape  # noqa: N806
        output = output.transpose(1, 2).reshape(B, K, H * D)
        output = self.out_proj(output)
        output = self.out_dropout(output)

        return output
