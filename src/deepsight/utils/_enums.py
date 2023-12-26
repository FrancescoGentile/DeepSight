##
##
##

import enum

from deepsight.typing import str_enum


@str_enum
class InterpolationMode(enum.Enum):
    NEAREST = "nearest"
    NEAREST_EXACT = "nearest_exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
