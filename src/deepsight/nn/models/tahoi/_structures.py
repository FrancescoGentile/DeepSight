##
##
##

from dataclasses import dataclass
from typing import Annotated

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class HCStep:
    similarity_matrix: Annotated[Tensor, "N N", float]
    mask: Annotated[Tensor, "N N", bool]
    coboundary_matrix: Annotated[Tensor, "M N", int] | None


@dataclass(frozen=True, slots=True)
class LayerOutput:
    steps: list[HCStep]
    num_nodes: list[int]
    num_edges: list[int]
    num_binary_edges: list[int]
    boundary_matrix: Annotated[Tensor, "N H", bool, torch.sparse_coo]
    interaction_logits: Annotated[Tensor, "H C", float]
    binary_interactions: Annotated[Tensor, "E 3", int]
    binary_interaction_logits: Annotated[Tensor, "E", float]


Output = list[LayerOutput]
