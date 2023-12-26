##
##
##

from dataclasses import dataclass

from deepsight.models import vit


@dataclass(frozen=True)
class Config:
    """onfiguration for the GAHOI model."""

    human_class_id: int
    num_entity_classes: int
    num_interaction_classes: int
    allow_human_human: bool
    encoder_variant: vit.Variant = vit.Variant.OG_BASE_PATCH32_IMG384
    num_decoder_layers: int = 6
    node_dim: int = 256
    edge_dim: int = 256
    cpb_hidden_dim: int = 256
    num_heads: int = 8
    negative_slope: float = 0.2
    add_dropout_to_vit: bool = True
    patch_dropout: float = 0.1
    qkv_dropout: float = 0.1
    attn_dropout: float = 0.1
    proj_dropout: float = 0.1
    ffn_dropout: float = 0.1
    classifier_dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.node_dim % self.num_heads != 0:
            raise ValueError(
                f"node_dim ({self.node_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
