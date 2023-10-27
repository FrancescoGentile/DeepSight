##
##
##

from dataclasses import dataclass
from typing import Annotated

from torch import Tensor


@dataclass(frozen=True, slots=True)
class Output:
    num_nodes: list[int]
    num_edges: list[int]
    interactions: Annotated[Tensor, "E 2", int]
    interaction_logits: Annotated[Tensor, "E C", float]
