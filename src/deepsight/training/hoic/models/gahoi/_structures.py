##
##
##

from dataclasses import dataclass
from typing import Literal, Self

from deepsight.typing import Detachable, Tensor


@dataclass(frozen=True, slots=True)
class LayerOutput(Detachable):
    label_logits: Tensor[Literal["E V"], float]
    suppress_logits: Tensor[Literal["E 1"], float]

    def detach(self) -> Self:
        return self.__class__(self.label_logits.detach(), self.suppress_logits.detach())


@dataclass(frozen=True, slots=True)
class Output(Detachable):
    num_nodes: list[int]
    num_edges: list[int]
    indices: Tensor[Literal["E 2"], int]
    # for each interaction, we mask out the not valid verbs
    # not valid verbs are set to True in the mask
    interaction_mask: Tensor[Literal["E V"], bool] | None
    layers: list[LayerOutput]

    def detach(self) -> Self:
        return self.__class__(
            self.num_nodes,
            self.num_edges,
            self.indices.detach(),
            self.interaction_mask.detach()
            if self.interaction_mask is not None
            else None,
            [layer.detach() for layer in self.layers],
        )
