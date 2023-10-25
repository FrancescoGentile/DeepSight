##
##
##

import enum

from deepsight.typing import str_enum


@str_enum
class Split(enum.Enum):
    """Common dataset splits."""

    TRAIN = "train"
    EVAL = "eval"
    TEST = "test"
