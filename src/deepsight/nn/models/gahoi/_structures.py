##
##
##

from dataclasses import dataclass
from typing import Annotated

from torch import Tensor


@dataclass(frozen=True, slots=True)
class LayerOutput:
    num_nodes: list[int]
    num_edges: list[int]
    indices: Annotated[Tensor, "E 2", int]
    label_logits: Annotated[Tensor, "E V", float]
    suppress_logits: Annotated[Tensor, "E 1", float]
    # for each interaction, we mask out the not valid verbs
    # not valid verbs are set to True in the mask
    interaction_mask: Annotated[Tensor, "E V", bool] | None


Output = list[LayerOutput]
