##
##
##

import enum
from dataclasses import dataclass
from typing import Literal, Self

from deepsight.typing import str_enum


@str_enum
class Variant(enum.Enum):
    """Predefined ViT variants."""

    TINY = "tiny"
    SMALL = "small"
    BASE = "base"
    LARGE = "large"
    HUGE = "huge"
    GIANT = "giant"
    GIGANTIC = "gigantic"


@dataclass(frozen=True)
class Configs:
    """The ViT configs."""

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
    no_class_embedding: bool = False
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

    @classmethod
    def from_variant(
        cls,
        variant: Variant,
        patch_size: Literal[8, 14, 16, 32],
        image_size: Literal[224, 384],
    ) -> Self:
        """Constructs a ViT config from a predefined variant.

        Args:
            variant: The predefined variant.
            patch_size: The patch size.
            image_size: The image size.

        Returns:
            The ViT config.
        """
        ffn_hidden_ratio = 4
        match variant:
            case Variant.TINY:
                embed_dim = 192
                num_layers = 12
                num_heads = 3
            case Variant.SMALL:
                embed_dim = 384
                num_layers = 12
                num_heads = 6
            case Variant.BASE:
                embed_dim = 768
                num_layers = 12
                num_heads = 12
            case Variant.LARGE:
                embed_dim = 1024
                num_layers = 24
                num_heads = 16
            case Variant.HUGE:
                embed_dim = 1280
                num_layers = 32
                num_heads = 16
            case Variant.GIANT:
                embed_dim = 1408
                num_layers = 40
                num_heads = 16
                ffn_hidden_ratio = 48 / 11
            case Variant.GIGANTIC:
                embed_dim = 1664
                num_layers = 48
                num_heads = 16
                ffn_hidden_ratio = 64 / 13

        return cls(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_hidden_ratio=ffn_hidden_ratio,
        )
