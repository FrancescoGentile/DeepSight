## Source code taken and modified from torchvision
## https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
##

import math
from collections.abc import Iterable

from torch import nn

from deepsight import utils
from deepsight.structures import BatchedImages

from ._blocks import BasicBlock, Bottleneck
from ._config import EncoderConfig


class Encoder(nn.Module):
    """ResNet encoder."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()

        stem_width = 64
        self.conv1 = nn.Conv2d(
            config.in_channels,
            stem_width,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.norm1 = config.norm_layer(stem_width)
        self.act1 = config.act_layer()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList()

        in_channels = stem_width
        out_channels = [64, 128, 256, 512]
        for num_blocks, out_channel in zip(
            config.blocks_per_layer, out_channels, strict=True
        ):
            blocks = self._make_layer(
                config,
                num_blocks,
                in_channels,
                out_channel,
                stride=1 if in_channels == out_channel else 2,
            )
            self.layers.append(blocks)
            in_channels = out_channel * config.block.expansion

    # ----------------------------------------------------------------------- #
    # Public methods
    # ----------------------------------------------------------------------- #

    def get_intermediate_outputs(
        self,
        images: BatchedImages,
        return_layers: int | Iterable[int] = -1,
    ) -> tuple[BatchedImages, ...]:
        """Returns intermediate outputs of the encoder.

        Args:
            images: Batched images.
            return_layers: Indices of layers to return. The indices can be negative,
                in which case the layers are counted from the end.

        Returns:
            Intermediate outputs of the encoder.
        """
        return_layers = utils.to_tuple(return_layers)
        indices = {idx if idx >= 0 else len(self.layers) + idx for idx in return_layers}
        for idx in indices:
            if idx < 0 or idx >= len(self.layers):
                raise IndexError(
                    f"Index {idx} is out of range for the encoder with "
                    f"{len(self.layers)} layers."
                )

        outputs = []
        x = images.data
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        if not images.is_padded():
            out = BatchedImages(x)
        else:
            new_image_sizes = tuple(
                (math.ceil(h / 4), math.ceil(w / 4)) for h, w in images.image_sizes
            )
            out = BatchedImages(x, new_image_sizes)

        for idx, layer in enumerate(self.layers):
            out = layer(out)
            if idx in indices:
                outputs.append(out)

        return tuple(outputs)

    def forward(self, images: BatchedImages) -> BatchedImages:
        return self.get_intermediate_outputs(images, return_layers=-1)[-1]

    # ----------------------------------------------------------------------- #
    # Magic methods
    # ----------------------------------------------------------------------- #

    def __call__(self, images: BatchedImages) -> BatchedImages:
        return super().__call__(images)

    # ----------------------------------------------------------------------- #
    # Private methods
    # ----------------------------------------------------------------------- #

    def _make_layer(
        self,
        config: EncoderConfig,
        num_blocks: int,
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> nn.Sequential:
        blocks = []
        in_channels = in_channels
        for idx in range(num_blocks):
            stride = stride if idx == 0 else 1
            if config.block == BasicBlock:
                blocks.append(
                    BasicBlock(
                        in_channels,
                        out_channels,
                        stride=stride,
                        act_layer=config.act_layer,
                        norm_layer=config.norm_layer,
                    )
                )
            elif config.block == Bottleneck:
                blocks.append(
                    Bottleneck(
                        in_channels,
                        out_channels,
                        stride=stride,
                        act_layer=config.act_layer,
                        norm_layer=config.norm_layer,
                    )
                )
            else:
                # This should never happen
                raise NotImplementedError

            in_channels = out_channels * config.block.expansion

        return nn.Sequential(*blocks)
