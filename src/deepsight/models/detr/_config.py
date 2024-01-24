##
##
##

from collections.abc import Callable
from dataclasses import dataclass

from deepsight.structures import BatchedImages


@dataclass(frozen=True)
class Config:
    """Configuration for DETR model."""

    backbone: Callable[[BatchedImages], BatchedImages]
    feature_dim: int
    num_queries: int
    num_classes: int
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
