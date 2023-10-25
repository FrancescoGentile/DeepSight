##
##
##

from collections.abc import Iterable
from typing import Annotated

import timm
import torch
import torchvision.transforms.v2 as T  # noqa
from timm.models import VisionTransformer
from torch import Tensor, nn

import deepsight.utils as utils
from deepsight.structures import Image


class ViTEncoder(nn.Module):
    def __init__(
        self,
        name: str,
        image_size: int | tuple[int, int],
        out_channels: int,
    ) -> None:
        """Initialize the ViT encoder.

        Args:
            name: The name of the ViT model to use. The name is the same as
                the one used in timm.
            image_size: The size of the input image. If an integer is given,
                the image is assumed to be square.
            out_channels: The number of output channels. If the number of
                output channels is different from the number of channels
                of the ViT model, a linear projection is applied.
        """
        super().__init__()

        if not name.startswith("vit"):
            raise ValueError(f"Invalid ViT encoder name: {name}.")

        image_size = utils.to_2tuple(image_size)

        self.transform = T.Compose(
            [
                T.Resize(size=image_size),
                T.ToDtype(torch.float, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.vit: VisionTransformer = timm.create_model(
            name, pretrained=True, image_size=image_size
        )

        if self.vit.embed_dim != out_channels:
            self.proj = nn.Linear(self.vit.embed_dim, out_channels)
        else:
            self.proj = nn.Identity()

        patch_size = self.vit.patch_embed.patch_size
        self.h: int = image_size[0] // patch_size[0]
        self.w: int = image_size[1] // patch_size[1]

    def forward(self, images: Iterable[Image]) -> Annotated[Tensor, "B D h w", float]:
        """Forward pass of the ViT encoder."""
        images_list = [self.transform(image.to_tensor()) for image in images]
        features = torch.stack(images_list)
        features: Tensor = self.vit.forward_features(features)
        features = self.proj(features[:, 1:])  # remove the CLS token

        B, _, D = features.shape  # noqa
        features = features.reshape(B, self.h, self.w, D).permute(0, 3, 1, 2)

        return features

    def __call__(self, images: Iterable[Image]) -> Annotated[Tensor, "B D h w", float]:
        return self.forward(images)
