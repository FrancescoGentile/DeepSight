##
##
##

import enum
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Self

from deepsight.layers import FrozenBatchNorm2d
from deepsight.models import resnet
from deepsight.structures import BatchedImages
from deepsight.typing import str_enum


@str_enum
class Variant(enum.Enum):
    """Predefined variants of DETR model."""

    DETR_RESNET50 = "detr_resnet50"


@dataclass(frozen=True)
class Config:
    """Configuration for DETR model."""

    backbone: Callable[[BatchedImages], BatchedImages]
    feature_dim: int
    num_classes: int
    num_queries: int = 100
    embedding_dim: int = 256
    pos_temperature: float = 10000
    pos_normalize: bool = True
    num_encoder_layers: int = 6
    encoder_post_norm: bool = False
    num_decoder_layers: int = 6
    decoder_post_norm: bool = True
    num_heads: int = 8
    attn_dropout: float = 0.1
    proj_dropout: float = 0.0
    ffn_dropout: float = 0.1
    ffn_num_layers: int = 2
    ffn_dim: int = 2048
    threshold: float = 0.0

    @classmethod
    def from_variant(
        cls,
        variant: Variant | str,
        num_classes: int,
        threshold: float = 0.0,
    ) -> Self:
        match Variant(variant):
            case Variant.DETR_RESNET50:
                res_config = resnet.EncoderConfig.from_variant(resnet.Variant.RESNET50)
                res_config = replace(res_config, norm_layer=FrozenBatchNorm2d)
                backbone = resnet.Encoder(res_config)
                return cls(
                    backbone=backbone,
                    feature_dim=2048,
                    num_classes=num_classes,
                    threshold=threshold,
                )
