# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

import abc
from typing import Literal

import torch
from torch import nn

from deepsight.modules import Module
from deepsight.typing import Tensor

# --------------------------------------------------------------------------- #
# QKV without positional embeddings
# --------------------------------------------------------------------------- #


class QKVGenerator(Module, abc.ABC):
    """Interface for a generator of queries, keys and values.

    The classes implementing this protocol are used to generate queries, keys and
    values using the input tensors.

    If you need positional embeddings, you can define a new class that implements the
    `QKVGeneratorWithPos` protocol instead or use an existing one.
    """

    @property
    @abc.abstractmethod
    def num_heads(self) -> int:
        """The number of heads."""
        ...

    @property
    @abc.abstractmethod
    def head_dim(self) -> int:
        """The dimension of the hidden space of each head."""
        ...

    @abc.abstractmethod
    def __call__(
        self,
        query: Tensor[Literal["B Q Dq"], float],
        key: Tensor[Literal["B K Dk"], float],
        value: Tensor[Literal["B K Dv"], float],
    ) -> tuple[
        Tensor[Literal["B H Q Dh"], float],
        Tensor[Literal["B H K Dh"], float],
        Tensor[Literal["B H K Dh"], float],
    ]:
        """Generate queries, keys and values.

        Args:
            query: The input tensor used to generate the query tensor.
            key: The input tensor used to generate the key tensor.
            value: The input tensor used to generate the value tensor.

        Returns:
            The generated queries, keys and values.
        """
        ...


class LinearQKVGenerator(QKVGenerator):
    """A generator of queries, keys and values.

    This class implements the `QKVGenerator` protocol. It generates queries, keys and
    values using the input tensors by linearly projecting them to the hidden space.
    """

    # ----------------------------------------------------------------------- #
    # Constructor
    # ----------------------------------------------------------------------- #

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        hidden_dim: int,
        num_heads: int,
    ) -> None:
        """Initialize the query, key, and value generator.

        Args:
            query_dim: The dimension of the input tensor used to generate the query
                tensor.
            key_dim: The dimension of the input tensor used to generate the key tensor.
            value_dim: The dimension of the input tensor used to generate the value
                tensor.
            hidden_dim: The dimension of the hidden space.
            num_heads: The number of heads.
        """
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"The hidden dimension ({hidden_dim}) must be divisible by the number "
                f"of heads ({num_heads})."
            )

        self._num_heads = num_heads
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(value_dim, hidden_dim)

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self.query_proj.out_features // self.num_heads

    # ----------------------------------------------------------------------- #
    # Magic methods
    # ----------------------------------------------------------------------- #

    def __call__(
        self,
        query: Tensor[Literal["B Q Dq"], float],
        key: Tensor[Literal["B K Dk"], float],
        value: Tensor[Literal["B K Dv"], float],
    ) -> tuple[
        Tensor[Literal["B H Q Dh"], float],
        Tensor[Literal["B H K Dh"], float],
        Tensor[Literal["B H K Dh"], float],
    ]:
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        query = _split_heads(query, self.num_heads)
        key = _split_heads(key, self.num_heads)
        value = _split_heads(value, self.num_heads)

        return query, key, value


# --------------------------------------------------------------------------- #
# QKV with positional embeddings
# --------------------------------------------------------------------------- #


class QKVGeneratorWithPos(Module, abc.ABC):
    """Interface for a generator of queries, keys and values with positional embeddings.

    The classes implementing this protocol are used to generate queries, keys and values
    using the input tensors and positional embeddings. For example, positional
    embeddings are used by DETR-like models to encode the bounding box coordinates of
    each input query and the absolute coordinates of the image patches.

    If you do not need positional embeddings, you can define a new class that implements
    the `QKVGenerator` protocol instead or use an existing one.
    """

    @property
    @abc.abstractmethod
    def num_heads(self) -> int:
        """The number of heads."""

    @property
    @abc.abstractmethod
    def head_dim(self) -> int:
        """The dimension of the hidden space of each head."""

    @abc.abstractmethod
    def __call__(
        self,
        query: Tensor[Literal["B Q Dq"], float],
        key: Tensor[Literal["B K Dk"], float],
        value: Tensor[Literal["B K Dv"], float],
        query_pos: Tensor[Literal["B Q Dqp"], float],
        key_pos: Tensor[Literal["B K Dkp"], float],
    ) -> tuple[
        Tensor[Literal["B H Q Dh"], float],
        Tensor[Literal["B H K Dh"], float],
        Tensor[Literal["B H K Dh"], float],
    ]:
        """Generate queries, keys and values with positional embeddings.

        Args:
            query: The input tensor used to generate the query tensor.
            key: The input tensor used to generate the key tensor.
            value: The input tensor used to generate the value tensor.
            query_pos: The positional embeddings for the queries.
            key_pos: The positional embeddings for the keys.

        Returns:
            The generated queries, keys and values.
        """


class LinearQKVGeneratorWithPrePosAddition(QKVGeneratorWithPos):
    """A generator of queries, keys and values with positional embeddings.

    This class implements the `QKVGeneratorWithPos` protocol. It generates queries,
    keys and values using the input tensors and positional embeddings. The positional
    embeddings are added to the input tensors before the linear projection to the
    hidden space.
    """

    # ----------------------------------------------------------------------- #
    # Constructor
    # ----------------------------------------------------------------------- #

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        hidden_dim: int,
        num_heads: int,
    ) -> None:
        """Initialize the query, key, and value generator.

        Args:
            query_dim: The dimension of the input tensor used to generate the query
                tensor.
            key_dim: The dimension of the input tensor used to generate the key tensor.
            value_dim: The dimension of the input tensor used to generate the value
                tensor.
            hidden_dim: The dimension of the hidden space.
            num_heads: The number of heads.
        """
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"The hidden dimension ({hidden_dim}) must be divisible by the number "
                f"of heads ({num_heads})."
            )

        self._num_heads = num_heads
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(value_dim, hidden_dim)

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self.query_proj.out_features // self.num_heads

    # ----------------------------------------------------------------------- #
    # Magic methods
    # ----------------------------------------------------------------------- #

    def __call__(
        self,
        query: Tensor[Literal["B Q Dq"], float],
        key: Tensor[Literal["B K Dk"], float],
        value: Tensor[Literal["B K Dv"], float],
        query_pos: Tensor[Literal["B Q Dqp"], float],
        key_pos: Tensor[Literal["B K Dkp"], float],
    ) -> tuple[
        Tensor[Literal["B H Q Dh"], float],
        Tensor[Literal["B H K Dh"], float],
        Tensor[Literal["B H K Dh"], float],
    ]:
        query = self.query_proj(query + query_pos)
        key = self.key_proj(key + key_pos)
        value = self.value_proj(value)

        query = _split_heads(query, self.num_heads)
        key = _split_heads(key, self.num_heads)
        value = _split_heads(value, self.num_heads)

        return query, key, value


class LinearQKVGeneratorWithPostPosAddition(QKVGeneratorWithPos):
    """A generator of queries, keys and values with positional embeddings.

    This class implements the `QKVGeneratorWithPos` protocol. It generates queries,
    keys and values using the input tensors and positional embeddings. This class
    first linearly projects the input tensors to the hidden space and then adds the
    positional embeddings to the projected tensors.
    """

    # ----------------------------------------------------------------------- #
    # Constructor
    # ----------------------------------------------------------------------- #

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        hidden_dim: int,
        num_heads: int,
        query_pos_dim: int | None = None,
        key_pos_dim: int | None = None,
    ) -> None:
        """Initialize the query, key, and value generator.

        Args:
            query_dim: The dimension of the input tensor used to generate the query
                tensor.
            key_dim: The dimension of the input tensor used to generate the key tensor.
            value_dim: The dimension of the input tensor used to generate the value
                tensor.
            hidden_dim: The dimension of the hidden space.
            num_heads: The number of heads.
            query_pos_dim: The dimension of the query positional embeddings. If given,
                a linear projection is applied to the query positional embeddings before
                they are added to the query tensor. If `None`, the query positional
                embeddings are directly added to the query tensor after the projection
                to the hidden space (the query positional embeddings must then have the
                same dimension as the hidden space).
            key_pos_dim: The dimension of the key positional embeddings. See
                `query_pos_dim` for details.
        """
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"The hidden dimension ({hidden_dim}) must be divisible by the number "
                f"of heads ({num_heads})."
            )

        self._num_heads = num_heads
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(value_dim, hidden_dim)

        self.query_pos_proj = (
            nn.Linear(query_pos_dim, hidden_dim)
            if query_pos_dim is not None
            else nn.Identity()
        )

        self.key_pos_proj = (
            nn.Linear(key_pos_dim, hidden_dim)
            if key_pos_dim is not None
            else nn.Identity()
        )

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self.query_proj.out_features // self.num_heads

    # ----------------------------------------------------------------------- #
    # Magic methods
    # ----------------------------------------------------------------------- #

    def __call__(
        self,
        query: Tensor[Literal["B Q Dq"], float],
        key: Tensor[Literal["B K Dk"], float],
        value: Tensor[Literal["B K Dv"], float],
        query_pos: Tensor[Literal["B Q Dqp"], float],
        key_pos: Tensor[Literal["B K Dkp"], float],
    ) -> tuple[
        Tensor[Literal["B H Q Dh"], float],
        Tensor[Literal["B H K Dh"], float],
        Tensor[Literal["B H K Dh"], float],
    ]:
        query = self.query_proj(query) + self.query_pos_proj(query_pos)
        key = self.key_proj(key) + self.key_pos_proj(key_pos)
        value = self.value_proj(value)

        query = _split_heads(query, self.num_heads)
        key = _split_heads(key, self.num_heads)
        value = _split_heads(value, self.num_heads)

        return query, key, value


class LinearQKVGeneratorWithPrePosConcat(QKVGeneratorWithPos):
    """A generator of queries, keys and values with positional embeddings.

    This class implements the `QKVGeneratorWithPos` protocol. It generates queries,
    keys and values using the input tensors and positional embeddings. The positional
    embeddings are concatenated to the input tensors before the linear projection to
    the hidden space.
    """

    # ----------------------------------------------------------------------- #
    # Constructor
    # ----------------------------------------------------------------------- #

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        hidden_dim: int,
        num_heads: int,
        query_pos_dim: int,
        key_pos_dim: int,
    ) -> None:
        """Initialize the query, key, and value generator.

        Args:
            query_dim: The dimension of the input tensor used to generate the query
                tensor.
            key_dim: The dimension of the input tensor used to generate the key tensor.
            value_dim: The dimension of the input tensor used to generate the value
                tensor.
            hidden_dim: The dimension of the hidden space.
            num_heads: The number of heads.
            query_pos_dim: The dimension of the query positional embeddings.
            key_pos_dim: The dimension of the key positional embeddings.
        """
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"The hidden dimension ({hidden_dim}) must be divisible by the number "
                f"of heads ({num_heads})."
            )

        self._num_heads = num_heads
        self.query_proj = nn.Linear(query_dim + query_pos_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim + key_pos_dim, hidden_dim)
        self.value_proj = nn.Linear(value_dim, hidden_dim)

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self.query_proj.out_features // self.num_heads

    # ----------------------------------------------------------------------- #
    # Magic methods
    # ----------------------------------------------------------------------- #

    def __call__(
        self,
        query: Tensor[Literal["B Q Dq"], float],
        key: Tensor[Literal["B K Dk"], float],
        value: Tensor[Literal["B K Dv"], float],
        query_pos: Tensor[Literal["B Q Dqp"], float],
        key_pos: Tensor[Literal["B K Dkp"], float],
    ) -> tuple[
        Tensor[Literal["B H Q Dh"], float],
        Tensor[Literal["B H K Dh"], float],
        Tensor[Literal["B H K Dh"], float],
    ]:
        query = self.query_proj(torch.cat([query, query_pos], dim=-1))
        key = self.key_proj(torch.cat([key, key_pos], dim=-1))
        value = self.value_proj(value)

        query = _split_heads(query, self.num_heads)
        key = _split_heads(key, self.num_heads)
        value = _split_heads(value, self.num_heads)

        return query, key, value


class LinearQKVGeneratorWithPostPosConcat(QKVGeneratorWithPos):
    """A generator of queries, keys and values with positional embeddings.

    This class implements the `QKVGeneratorWithPos` protocol. It generates queries,
    keys and values using the input tensors and positional embeddings. This class
    first linearly projects the input tensors to the hidden space and then concatenates
    the positional embeddings to the projected tensors.
    """

    # ----------------------------------------------------------------------- #
    # Constructor
    # ----------------------------------------------------------------------- #

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        hidden_dim: int,
        num_heads: int,
        query_pos_dim: int | None = None,
        key_pos_dim: int | None = None,
    ) -> None:
        """Initialize the query, key, and value generator.

        Args:
            query_dim: The dimension of the input tensor used to generate the query
                tensor.
            key_dim: The dimension of the input tensor used to generate the key tensor.
            value_dim: The dimension of the input tensor used to generate the value
                tensor.
            hidden_dim: The dimension of the hidden space.
            num_heads: The number of heads.
            query_pos_dim: The dimension of the query positional embeddings. If given,
                a linear projection is applied to the query positional embeddings before
                they are concatenated to the query tensor. If `None`, the query
                positional embeddings are directly concatenated to the query tensor
                after the projection to the hidden space.
            key_pos_dim: The dimension of the key positional embeddings. See
                `query_pos_dim` for details.
        """
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"The hidden dimension ({hidden_dim}) must be divisible by the number "
                f"of heads ({num_heads})."
            )

        self._num_heads = num_heads
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(value_dim, hidden_dim)

        self.query_pos_proj = (
            nn.Linear(query_pos_dim, hidden_dim)
            if query_pos_dim is not None
            else nn.Identity()
        )

        self.key_pos_proj = (
            nn.Linear(key_pos_dim, hidden_dim)
            if key_pos_dim is not None
            else nn.Identity()
        )

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self.query_proj.out_features // self.num_heads

    # ----------------------------------------------------------------------- #
    # Magic methods
    # ----------------------------------------------------------------------- #

    def __call__(
        self,
        query: Tensor[Literal["B Q Dq"], float],
        key: Tensor[Literal["B K Dk"], float],
        value: Tensor[Literal["B K Dv"], float],
        query_pos: Tensor[Literal["B Q Dqp"], float],
        key_pos: Tensor[Literal["B K Dkp"], float],
    ) -> tuple[
        Tensor[Literal["B H Q Dh"], float],
        Tensor[Literal["B H K Dh"], float],
        Tensor[Literal["B H K Dh"], float],
    ]:
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        query = _split_heads(query, self.num_heads)
        key = _split_heads(key, self.num_heads)
        value = _split_heads(value, self.num_heads)

        query_pos = self.query_pos_proj(query_pos)
        key_pos = self.key_pos_proj(key_pos)

        query_pos = _split_heads(query_pos, self.num_heads)
        key_pos = _split_heads(key_pos, self.num_heads)

        query = torch.cat([query, query_pos], dim=-1)
        key = torch.cat([key, key_pos], dim=-1)

        return query, key, value


# --------------------------------------------------------------------------- #
# Private helper functions
# --------------------------------------------------------------------------- #


def _split_heads(
    x: Tensor[Literal["B L D"], float], num_heads: int
) -> Tensor[Literal["B H L (D/H)"], float]:
    """Split the hidden dimension of a tensor into multiple heads."""
    B, L, D = x.shape  # noqa: N806
    head_dim = D // num_heads

    x = x.reshape(B, L, num_heads, head_dim)
    x = x.transpose(1, 2)  # (B, H, L, Dh)

    return x
