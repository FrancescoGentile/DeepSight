# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

import enum
import os
import pathlib
from typing import Annotated, Any, Literal, LiteralString

import torch

type Number = bool | float | int
"""Union of all types that can be used to represent a number."""

type PathLike = str | os.PathLike[str] | pathlib.Path
"""Union of all types that can be used to represent a path."""

type DeviceLike = str | torch.device
"""Union of all types that can be used to represent a device."""

type EnumLike[T: enum.Enum] = T | str
"""Union of all types that can be used to represent an enum."""

type Tensor[S: LiteralString, DT] = Annotated[torch.Tensor, S, DT]
"""A tensor with an annotated shape and dtype."""

type SparseTensor[S: LiteralString, DT] = Annotated[torch.Tensor, S, DT]
"""A sparse tensor with an annotated shape and dtype."""

type Loss = Tensor[Literal[""], float]
"""A loss is a scalar float tensor."""

type Losses = dict[str, Loss]
"""A mapping from loss names to losses."""

type Configs = dict[str, Any]
"""A dictionary of configuration values."""

type StateDict = dict[str, Any]
"""A dictionary of state values."""
