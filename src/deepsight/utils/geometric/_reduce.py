##
##
##

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
