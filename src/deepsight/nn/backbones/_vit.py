##
##
##

from __future__ import annotations

import enum
import math
from typing import Annotated, Literal, TypedDict, overload

import timm
import torch
import torch.nn.functional as F  # noqa
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from typing_extensions import Self

from deepsight import utils
from deepsight.structures import BatchedImages, BatchedSequences
from deepsight.typing import str_enum


class ViTEncoder(nn.Module):
    """The original implementation of the Vision Transformer."""

    @str_enum
    class Variant(enum.Enum):
        TINY = "tiny"
        SMALL = "small"
        BASE = "base"
        LARGE = "large"
        HUGE = "huge"
        GIANT = "giant"
        GIGANTIC = "gigantic"

    # ----------------------------------------------------------------------- #
    # Constructor and Factory Methods
    # ----------------------------------------------------------------------- #

    def __init__(
        self,
        image_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        ffn_hidden_ratio: float = 4,
        qkv_bias: bool = True,
        qk_normalize: bool = False,
        init_scale: float | None = None,
        use_class_token: bool = True,
        no_class_embedding: bool = False,
        pre_normalize: bool = False,
        pos_embed_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        """Initialize the Vision Transformer."""
        super().__init__()
        image_size = utils.to_2tuple(image_size)
        patch_size = utils.to_2tuple(patch_size)

        self.num_h_patches = math.ceil(image_size[0] / patch_size[0])
        self.num_w_patches = math.ceil(image_size[1] / patch_size[1])
        self.no_class_embedding = no_class_embedding

        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            bias=not pre_normalize,
        )

        if use_class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        num_patches = self.num_h_patches * self.num_w_patches
        num_embeds = num_patches + int(use_class_token and not no_class_embedding)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_embeds, embed_dim))
        self.pos_dropout = nn.Dropout(pos_embed_dropout)
        # in timm, this is called "norm_pre"
        self.pre_layernorm = (
            nn.LayerNorm(embed_dim, eps=1e-6) if pre_normalize else nn.Identity()
        )

        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path, num_layers)]
        # in timm, this is called "blocks"
        self.layers = nn.ModuleList(
            [
                Layer(
                    embed_dim,
                    num_heads,
                    ffn_hidden_ratio,
                    qkv_bias,
                    qk_normalize,
                    attn_dropout,
                    proj_dropout,
                    init_scale,
                    drop_path_rates[i],
                )
                for i in range(num_layers)
            ]
        )

        # in timm, this is called "norm"
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=1e-6)

    @classmethod
    def build(
        cls,
        variant: Variant = Variant.BASE,
        patch_size: Literal[8, 14, 16, 32] = 16,
        image_size: Literal[224, 384] = 224,
        pretrained: bool = True,
    ) -> Self:
        model_name = f"vit_{variant.value}_patch{patch_size}_{image_size}"
        if pretrained and not timm.is_model_pretrained(model_name):
            raise ValueError(
                f"Pretrained weights for {model_name} are not available. "
                f"Please choose a different variant or set pretrained=False."
            )

        configs = _get_configs(variant)
        model = cls(
            image_size=image_size,
            patch_size=patch_size,
            **configs,
        )

        if pretrained:
            # This is a temporary workaround. In future versions, we will directly
            # load the state dict from the original source without delegating to timm.
            timm_model = timm.create_model(model_name, pretrained=True)
            state_dict = timm_model.state_dict()
            state_dict = _convert_weights(state_dict)
            model.load_state_dict(state_dict)
            del timm_model

        return model

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def forward(
        self, x: BatchedImages | Annotated[Tensor, "B C H W", float]
    ) -> BatchedImages | Annotated[Tensor, "B D h w", float]:
        x = self.patch_embed(x)  # (B, D, h, w)
        x_embed = self._add_pos_embeds(x)
        if isinstance(x_embed, BatchedSequences):
            out = x_embed.data
            mask = x_embed.mask
        else:
            out = x_embed
            mask = None

        out: Tensor = self.pre_layernorm(out)
        for layer in self.layers:
            out = layer(out, mask)

        out = self.post_layernorm(out)

        if self.cls_token is not None:
            out = out[:, 1:]  # remove cls token
        out = out.transpose(1, 2).reshape(*x.shape)  # (B, D, h, w)

        if isinstance(x, BatchedImages):
            return x.replace(data=out)
        else:
            return out

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    @overload
    def __call__(
        self, x: Annotated[Tensor, "B C H W", float]
    ) -> Annotated[Tensor, "B D h w", float]: ...

    @overload
    def __call__(self, x: BatchedImages) -> BatchedImages: ...

    def __call__(
        self, x: BatchedImages | Annotated[Tensor, "B C H W", float]
    ) -> BatchedImages | Annotated[Tensor, "B D h w", float]:
        return super().__call__(x)

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _add_pos_embeds(
        self, x: Annotated[Tensor, "B D h w", float] | BatchedImages
    ) -> Annotated[Tensor, "B hw D", float] | BatchedSequences:
        if isinstance(x, Tensor):
            h, w = x.shape[-2:]
            pos_embed = _resize_pos_embeds(
                self.pos_embed,
                (self.num_h_patches, self.num_w_patches),
                (h, w),
                num_prefix_tokens=int(self.cls_token is not None),
            )
            sequences = None
            out = x.flatten(2).transpose(1, 2)  # (B, hw, D)
        else:
            pos_embeds = []
            for h, w in x.image_sizes:
                embeds = _resize_pos_embeds(
                    self.pos_embed,
                    (self.num_h_patches, self.num_w_patches),
                    (h, w),
                    num_prefix_tokens=int(self.cls_token is not None),
                )
                pos_embeds.append(embeds[0])

            pos_embed = pad_sequence(pos_embeds, batch_first=True)

            sequences = x.to_sequences()
            out = sequences.data

        to_cat = []
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(out.shape[0], -1, -1)
            to_cat.append(cls_token)

        if self.no_class_embedding:
            out = out + pos_embed
            out = torch.cat(to_cat + [out], dim=1)
        else:
            out = torch.cat(to_cat + [out], dim=1)
            out = out + pos_embed

        out = self.pos_dropout(out)
        if sequences is not None:
            out = sequences.replace(data=out)

        return out


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        patch_size: int | tuple[int, int] = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        normalize: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.patch_size = utils.to_2tuple(patch_size)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6) if normalize else nn.Identity()

    def forward(
        self, x: Annotated[Tensor, "B C H W", float] | BatchedImages
    ) -> Annotated[Tensor, "B D h w", float] | BatchedImages:
        H, W = x.shape[-2:]  # noqa
        pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
        pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]

        data = x.data if isinstance(x, BatchedImages) else x
        if pad_h > 0 or pad_w > 0:
            data = F.pad(data, (0, pad_w, 0, pad_h))

        data = self.proj(data)  # (B, D, h, w)
        data = self.norm(data)  # (B, D, h, w)

        if isinstance(x, BatchedImages):
            new_image_sizes = tuple(
                ((h + pad_h) // self.patch_size[0], (w + pad_w) // self.patch_size[1])
                for h, w in x.image_sizes
            )
            return BatchedImages(data, image_sizes=new_image_sizes)
        else:
            return data


class Layer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_hidden_ratio: float = 4,
        qkv_bias: bool = False,
        qk_normalize: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        layer_scale_init: float | None = None,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        # Self-Attention
        self.sa_layernorm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.sa = SelfAttention(
            embed_dim, num_heads, qkv_bias, qk_normalize, attn_dropout, proj_dropout
        )
        self.sa_layerscale = (
            LayerScale(embed_dim, layer_scale_init)
            if layer_scale_init is not None
            else nn.Identity()
        )
        self.sa_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Feed-Forward
        self.ffn_layernorm = nn.LayerNorm(embed_dim, eps=1e-6)
        hidden_dim = int(embed_dim * ffn_hidden_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(proj_dropout),
        )
        self.ffn_layerscale = (
            LayerScale(embed_dim, layer_scale_init)
            if layer_scale_init is not None
            else nn.Identity()
        )
        self.ffn_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: Annotated[Tensor, "B N D", float],
        mask: Tensor | None,
    ) -> Annotated[Tensor, "B N D", float]:
        x = x + self.sa_drop_path(
            self.sa_layerscale(self.sa(self.sa_layernorm(x), mask))
        )
        x = x + self.ffn_drop_path(self.ffn_layerscale(self.ffn(self.ffn_layernorm(x))))

        return x


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_normalize: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_dropout = attn_dropout

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.q_norm = (
            nn.LayerNorm(embed_dim, eps=1e-6) if qk_normalize else nn.Identity()
        )
        self.k_norm = (
            nn.LayerNorm(embed_dim, eps=1e-6) if qk_normalize else nn.Identity()
        )
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(
        self,
        x: BatchedSequences | Annotated[Tensor, "B N D", float],
        mask: Annotated[Tensor, " B 1 1 N", bool] | None,
    ) -> Annotated[Tensor, "B N D", float]:
        B, N, D = x.shape  # noqa

        qkv = self.qkv_proj(x.data)  # (B, N, 3 * D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, Dh)
        q, k, v = qkv.unbind(dim=0)  # (B, H, N, Dh)
        q, k = self.q_norm(q), self.k_norm(k)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )

        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_dropout(out)

        return out


class LayerScale(nn.Module):
    def __init__(
        self, dim: int, init_value: float = 1e-5, inplace: bool = False
    ) -> None:
        super().__init__()

        self.inplace = inplace
        self.gamma = nn.Parameter(torch.full((dim,), init_value))

    def forward(
        self, x: Annotated[Tensor, "B N D", float]
    ) -> Annotated[Tensor, "B N D", float]:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()

        self.drop_prob = drop_prob

    def forward(self, x: BatchedSequences) -> BatchedSequences:
        raise NotImplementedError
        if self.training and self.drop_prob > 0.0:
            keep_prob = 1.0 - self.drop_prob
            mask = torch.rand(x.shape[:2], device=x.device) < keep_prob
            mask = mask[:, :, None].expand_as(x.data)
            x = x.replace(data=x.data.masked_fill(mask, 0.0) / keep_prob)
        return x


# --------------------------------------------------------------------------- #
# Private helper functions
# --------------------------------------------------------------------------- #


class Configs(TypedDict):
    embed_dim: int
    num_layers: int
    num_heads: int
    ffn_hidden_ratio: float


def _get_configs(variant: ViTEncoder.Variant) -> Configs:
    ffn_hidden_ratio = 4
    match variant:
        case ViTEncoder.Variant.TINY:
            embed_dim = 192
            num_layers = 12
            num_heads = 3
        case ViTEncoder.Variant.SMALL:
            embed_dim = 384
            num_layers = 12
            num_heads = 6
        case ViTEncoder.Variant.BASE:
            embed_dim = 768
            num_layers = 12
            num_heads = 12
        case ViTEncoder.Variant.LARGE:
            embed_dim = 1024
            num_layers = 24
            num_heads = 16
        case ViTEncoder.Variant.HUGE:
            embed_dim = 1280
            num_layers = 32
            num_heads = 16
        case ViTEncoder.Variant.GIANT:
            embed_dim = 1408
            num_layers = 40
            num_heads = 16
            ffn_hidden_ratio = 48 / 11
        case ViTEncoder.Variant.GIGANTIC:
            embed_dim = 1664
            num_layers = 48
            num_heads = 16
            ffn_hidden_ratio = 64 / 13

    return {
        "embed_dim": embed_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "ffn_hidden_ratio": ffn_hidden_ratio,
    }


def _convert_weights(state: dict[str, Tensor]) -> dict[str, Tensor]:  # noqa: C901
    new_state = state.copy()
    for key in state.keys():
        if key.startswith("norm_pre"):
            new_key = key.replace("norm_pre", "pre_layernorm")
        elif key.startswith("norm"):
            new_key = key.replace("norm", "post_layernorm")
        elif key.startswith("attn_pool"):
            new_key = None
        elif key.startswith("fc_norm"):
            new_key = None
        elif key.startswith("head"):
            new_key = None
        elif key.startswith("blocks."):
            new_key = key.replace("blocks.", "layers.")

            if "attn" in new_key:
                new_key = new_key.replace("attn", "sa")

            if "qkv" in new_key:
                new_key = new_key.replace("qkv", "qkv_proj")

            if "norm1" in new_key:
                new_key = new_key.replace("norm1", "sa_layernorm")

            if "ls1" in new_key:
                new_key = new_key.replace("ls1", "sa_layerscale")

            if "drop_path1" in new_key:
                new_key = new_key.replace("drop_path1", "sa_drop_path")

            if "norm2" in new_key:
                new_key = new_key.replace("norm2", "ffn_layernorm")

            if "ls2" in new_key:
                new_key = new_key.replace("ls2", "ffn_layerscale")

            if "drop_path2" in new_key:
                new_key = new_key.replace("drop_path2", "ffn_drop_path")

            if "mlp.fc1" in new_key:
                new_key = new_key.replace("mlp.fc1", "ffn.0")

            if "mlp.fc2" in new_key:
                new_key = new_key.replace("mlp.fc2", "ffn.3")
        else:
            continue

        if new_key is None:
            del new_state[key]
        else:
            new_state[new_key] = new_state.pop(key)

    return new_state


def _resize_pos_embeds(
    pos_embeds: Tensor,
    old_size: tuple[int, int],
    new_size: tuple[int, int],
    num_prefix_tokens: int = 1,
    interpolation: Literal["bilinear", "bicubic"] = "bicubic",
    antialias: bool = True,
) -> Tensor:
    num_old_tokens = pos_embeds.shape[1]
    num_new_tokens = (new_size[0] * new_size[1]) + num_prefix_tokens
    if num_old_tokens == num_new_tokens:
        return pos_embeds

    if num_prefix_tokens > 0:
        prefix = pos_embeds[:, :num_prefix_tokens]
        pos_embeds = pos_embeds[:, num_prefix_tokens:]
    else:
        prefix = None

    embed_dim = pos_embeds.shape[-1]
    orig_dtype = pos_embeds.dtype
    pos_embeds = pos_embeds.float()

    pos_embeds = pos_embeds.reshape(1, old_size[0], old_size[1], embed_dim)
    pos_embeds = pos_embeds.permute(0, 3, 1, 2)
    pos_embeds = F.interpolate(
        pos_embeds, size=new_size, mode=interpolation, antialias=antialias
    )

    pos_embeds = pos_embeds.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    pos_embeds = pos_embeds.to(orig_dtype)

    if prefix is not None:
        pos_embeds = torch.cat([prefix, pos_embeds], dim=1)

    return pos_embeds
