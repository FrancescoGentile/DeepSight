##
##
##

import enum
from dataclasses import dataclass
from typing import Self

from deepsight.typing import str_enum


@str_enum
class Variant(enum.Enum):
    """Predefined ViT variants."""

    OG_BASE_PATCH32_IMG224 = "og_base_patch32_img224"
    OG_BASE_PATCH32_IMG384 = "og_base_patch32_img384"
    DINOV2_BASE_PATCH14_IMG518 = "dinov2_base_patch14_img518"
    DINOV2_BASE_PATCH14_REG4_IMG518 = "dinov2_base_patch14_reg4_img518"


@dataclass(frozen=True)
class Config:
    """The Vision Transformer (ViT) configuration."""

    image_size: int | tuple[int, int]
    patch_size: int | tuple[int, int]
    in_channels: int = 3
    embed_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    ffn_hidden_ratio: float = 4
    qkv_bias: bool = True
    qk_normalize: bool = False
    layer_scale_init_value: float | None = None
    layer_norm_eps: float = 1e-6
    use_class_token: bool = True
    num_register_tokens: int = 0
    use_prefix_embedding: bool = True
    pre_normalize: bool = False
    pos_embed_dropout: float = 0.0
    qkv_dropout: float = 0.0
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    ffn_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})."
            )

        if self.num_register_tokens < 0:
            raise ValueError(
                f"num_register_tokens ({self.num_register_tokens}) must be "
                f"non-negative."
            )

    @classmethod
    def from_variant(cls, variant: Variant) -> Self:
        """Constructs a ViT config from a predefined variant."""
        match variant:
            case Variant.OG_BASE_PATCH32_IMG224:
                return cls(
                    patch_size=32,
                    image_size=224,
                    embed_dim=768,
                    num_layers=12,
                    num_heads=12,
                )
            case Variant.OG_BASE_PATCH32_IMG384:
                return cls(
                    patch_size=32,
                    image_size=384,
                    embed_dim=768,
                    num_layers=12,
                    num_heads=12,
                )
            case Variant.DINOV2_BASE_PATCH14_IMG518:
                return cls(
                    patch_size=14,
                    image_size=518,
                    embed_dim=768,
                    num_layers=12,
                    num_heads=12,
                    layer_scale_init_value=1e-5,
                )
            case Variant.DINOV2_BASE_PATCH14_REG4_IMG518:
                return cls(
                    patch_size=14,
                    image_size=518,
                    embed_dim=768,
                    num_layers=12,
                    num_heads=12,
                    num_register_tokens=4,
                    layer_scale_init_value=1e-5,
                    use_prefix_embedding=False,
                )
