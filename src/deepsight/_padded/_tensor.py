# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, LiteralString, Self

from .._tensor import Tensor


class PaddedTensor[S: LiteralString, DT](Tensor[S, DT]):
    def __new__(
        cls,
        data: Any,
        mask: Any | None = None,
    ) -> Self: ...

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #
