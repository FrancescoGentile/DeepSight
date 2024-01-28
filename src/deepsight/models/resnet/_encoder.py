# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0
# --------------------------------------------------------------------------- #
# Copyright (c) Soumith Chintala 2016. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# --------------------------------------------------------------------------- #
# Modified from:
# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
# --------------------------------------------------------------------------- #

import math
from collections.abc import Iterable

from torch import nn

from deepsight.modules import Backbone
from deepsight.structures import BatchedImages

from ._blocks import BasicBlock, Bottleneck
from ._config import EncoderConfig


class Encoder(Backbone):
    """ResNet encoder."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()

        self.stem_width = 64
        self.conv1 = nn.Conv2d(
            config.in_channels,
            self.stem_width,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.norm1 = config.norm_layer(self.stem_width)
        self.act1 = config.act_layer()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList()

        in_channels = self.stem_width
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

    def get_stages_info(self) -> tuple[Backbone.StageInfo, ...]:
        stages = [Backbone.StageInfo("stem", self.stem_width)]
        for idx, layer in enumerate(self.layers):
            stages.append(Backbone.StageInfo(f"layer{idx + 1}", layer[-1].out_channels))  # type: ignore

        return tuple(stages)

    # ----------------------------------------------------------------------- #
    # Magic methods
    # ----------------------------------------------------------------------- #

    def __call__(
        self,
        images: BatchedImages,
        *,
        return_stages: Iterable[str | int] = (-1,),
    ) -> tuple[BatchedImages, ...]:
        stages = self.get_stages_info()
        return_stages = [
            stage if isinstance(stage, str) else stages[stage].name
            for stage in return_stages
        ]
        return_stages_set = set(return_stages)

        stages = {}

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

        stages["stem"] = out
        for idx, layer in enumerate(self.layers):
            out = layer(out)
            name = f"layer{idx + 1}"
            # we make this check to avoid storing in memory the output of layers
            # that are not returned
            if name in return_stages_set:
                stages[name] = out

        outputs = []
        for stage in return_stages:
            outputs.append(stages[stage])

        return tuple(outputs)

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
