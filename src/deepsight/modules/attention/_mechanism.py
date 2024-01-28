# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0
# --------------------------------------------------------------------------- #
# Copyright (C) 2021-2023 ExplosionAI GmbH
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------- #
# Modified from:
# https://github.com/explosion/curated-transformers/blob/main/curated_transformers/layers/attention.py
# --------------------------------------------------------------------------- #

import abc
from typing import Literal

import torch.nn.functional as F  # noqa: N812

from deepsight.modules import Module
from deepsight.typing import Tensor

from ._mask import Mask


class AttentionMechanism(Module, abc.ABC):
    """Interface for classes implementing attention mechanisms.

    The classes implementing this interface are responsible for computing
    the attention scores between the query and the key tensors (and applying
    the mask if given) and then aggregating the values tensor using the
    computed attention scores to produce the output tensor.
    """

    @abc.abstractmethod
    def __call__(
        self,
        query: Tensor[Literal["B H Q D"], float],
        key: Tensor[Literal["B H K D"], float],
        value: Tensor[Literal["B H K D"], float],
        mask: Mask | None = None,
    ) -> Tensor[Literal["B H Q D"], float]: ...


class ScaledDotProductAttention(AttentionMechanism):
    """Scaled dot product attention module.

    This module computes the scaled dot product attention between the query
    and the key tensors (and applies the mask if given) to generate the
    attention scores. Then, a weighted sum of the value tensor is computed
    using the attention scores to generate the output tensor.
    """

    def __init__(self, dropout: float = 0.0, scale: float | None = None) -> None:
        """Initialize the scaled dot product attention module.

        Args:
            dropout: The dropout probability applied to the attention scores.
            scale: The scale factor applied to the dot product of the query
                and the key tensors. If `None`, the attention scores are
                divided by the square root of the dimension of the query
                and the key tensors.
        """
        super().__init__()
        self.dropout = dropout
        self.scale = scale

    def __call__(
        self,
        query: Tensor[Literal["B H Q D"], float],
        key: Tensor[Literal["B H K D"], float],
        value: Tensor[Literal["B H K D"], float],
        mask: Mask | None = None,
    ) -> Tensor[Literal["B H Q D"], float]:
        out = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=mask.tensor if mask is not None else None,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )

        return out
