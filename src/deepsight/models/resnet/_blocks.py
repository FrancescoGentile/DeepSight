# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

import math
from collections.abc import Callable

from torch import nn

from deepsight.modules import Activation, ImageNorm
from deepsight.structures import BatchedImages

# --------------------------------------------------------------------------- #
# Basic block
# --------------------------------------------------------------------------- #


class BasicBlock(nn.Module):
    """Basic block of ResNet."""

    expansion = 1

    # ----------------------------------------------------------------------- #
    # Constructor
    # ----------------------------------------------------------------------- #

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm_layer: Callable[[int], ImageNorm and nn.Module] = nn.BatchNorm2d,
        act_layer: Callable[[], Activation] = nn.ReLU,
    ) -> None:
        super().__init__()

        self.stride = stride
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = norm_layer(out_channels)
        self.act1 = act_layer()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = norm_layer(out_channels)
        self.act2 = act_layer()

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm_layer(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    # ----------------------------------------------------------------------- #
    # Public methods
    # ----------------------------------------------------------------------- #

    def forward(self, x: BatchedImages) -> BatchedImages:
        out = self.conv1(x.data)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        identity = self.downsample(x.data)

        out += identity
        out = self.act2(out)

        if out.shape[-2:] != x.shape[-2:]:
            if not x.is_padded():
                return BatchedImages(out)

            new_sizes = tuple(
                (math.ceil(h / self.stride), math.ceil(w / self.stride))
                for h, w in x.image_sizes
            )
            return BatchedImages(out, new_sizes)
        else:
            return x.new_with(data=out)

    # ----------------------------------------------------------------------- #
    # Magic methods
    # ----------------------------------------------------------------------- #

    def __call__(self, x: BatchedImages) -> BatchedImages:
        return super().__call__(x)


# --------------------------------------------------------------------------- #
# Bottleneck block
# --------------------------------------------------------------------------- #


class Bottleneck(nn.Module):
    """Bottleneck block of ResNet."""

    expansion = 4

    # ----------------------------------------------------------------------- #
    # Constructor
    # ----------------------------------------------------------------------- #

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        base_channels: int = 64,
        dilation: int = 1,
        norm_layer: Callable[[int], ImageNorm and nn.Module] = nn.BatchNorm2d,
        act_layer: Callable[[], Activation] = nn.ReLU,
    ) -> None:
        super().__init__()

        hidden_channels = int(out_channels * (base_channels / 64.0)) * groups

        self.stride = stride
        self.out_channels = out_channels * self.expansion

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.norm1 = norm_layer(hidden_channels)
        self.act1 = act_layer()

        self.conv2 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.norm2 = norm_layer(hidden_channels)
        self.act2 = act_layer()

        self.conv3 = nn.Conv2d(
            hidden_channels,
            out_channels * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.norm3 = norm_layer(out_channels * self.expansion)
        self.act3 = act_layer()

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm_layer(out_channels * self.expansion),
            )
        else:
            self.downsample = nn.Identity()

    # ----------------------------------------------------------------------- #
    # Public methods
    # ----------------------------------------------------------------------- #

    def forward(self, x: BatchedImages) -> BatchedImages:
        out = self.conv1(x.data)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.norm3(out)

        identity = self.downsample(x.data)

        out += identity
        out = self.act3(out)

        if out.shape[-2:] != x.shape[-2:]:
            if not x.is_padded():
                return BatchedImages(out)

            new_sizes = tuple(
                (math.ceil(h / self.stride), math.ceil(w / self.stride))
                for h, w in x.image_sizes
            )
            return BatchedImages(out, new_sizes)
        else:
            return x.new_with(data=out)

    # ----------------------------------------------------------------------- #
    # Magic methods
    # ----------------------------------------------------------------------- #

    def __call__(self, x: BatchedImages) -> BatchedImages:
        return super().__call__(x)
