##
##
##

import enum
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Self

from torch import nn

from deepsight.modules import Activation, ImageNorm
from deepsight.typing import str_enum

from ._blocks import BasicBlock, Bottleneck


@str_enum
class Variant(enum.Enum):
    """Predefined ResNet variants."""

    RESNET18 = "resnet18"
    RESNET50 = "resnet50"


@dataclass(frozen=True)
class EncoderConfig:
    """Configuration for the encoder.

    Attributes:
        in_channels: Number of input channels.
        block: Type of the residual block.
        blocks_per_layer: Number of blocks in each layer.
        act_layer: A callable that returns an activation layer.
        norm_layer: A callable accepting the number of channels and returning a
            normalization layer.
    """

    in_channels: int = 3
    block: type[BasicBlock | Bottleneck] = Bottleneck
    blocks_per_layer: tuple[int, int, int, int] = (3, 4, 6, 3)
    act_layer: Callable[[], Activation] = partial(nn.ReLU, inplace=True)
    norm_layer: Callable[[int], ImageNorm and nn.Module] = nn.BatchNorm2d

    @classmethod
    def from_variant(cls, variant: Variant | str) -> Self:
        """Returns a configuration for the given variant."""
        match Variant(variant):
            case Variant.RESNET18:
                return cls(block=BasicBlock, blocks_per_layer=(2, 2, 2, 2))
            case Variant.RESNET50:
                return cls(block=Bottleneck, blocks_per_layer=(3, 4, 6, 3))
