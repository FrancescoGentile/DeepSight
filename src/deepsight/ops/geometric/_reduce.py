# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

import enum

from deepsight.typing import str_enum


@str_enum
class ReduceOp(enum.Enum):
    """Reduce operation."""

    SUM = "sum"
    MUL = "mul"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
