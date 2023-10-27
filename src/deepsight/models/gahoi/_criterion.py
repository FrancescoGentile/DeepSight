##
##
##

from collections.abc import Iterable
from typing import Annotated

import torch
from torch import Tensor
from torchvision import ops

from deepsight.models import Criterion as _Criterion
from deepsight.structures import Batch
from deepsight.tasks.hoic import Annotations
from deepsight.typing import Configurable, JSONPrimitive

from ._structures import Output


class Criterion(_Criterion[Output, Annotations], Configurable):
    def __init__(self, focal_alpha: float = 0.25, focal_gamma: float = 2.0) -> None:
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @property
    def losses(self) -> Iterable[str]:
        return ["focal_loss"]

    def compute(
        self, output: Output, annotations: Batch[Annotations]
    ) -> dict[str, Tensor]:
        pred_interactions, pred_labels = output.interactions, output.interaction_logits
        gt_interactions, gt_labels = _batch(annotations, output.num_nodes)

        matched = pred_interactions.unsqueeze(1) == gt_interactions  # (E, E', 2)
        matched = matched.all(dim=2)  # (E, E')
        matched_pred, matched_target = torch.nonzero(matched, as_tuple=True)

        target = torch.zeros_like(pred_labels)
        target[matched_pred] = gt_labels[matched_target]

        loss = ops.sigmoid_focal_loss(
            pred_labels,
            target,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="mean",
        )
        return {"focal_loss": loss}

    def get_config(self) -> JSONPrimitive:
        return {"focal_alpha": self.focal_alpha, "focal_gamma": self.focal_gamma}


# --------------------------------------------------------------------------- #
# Private helper functions
# --------------------------------------------------------------------------- #


def _batch(
    annotations: Batch[Annotations], num_nodes: list[int]
) -> tuple[Annotated[Tensor, "E 2", int], Annotated[Tensor, "E C", float]]:
    node_offset = 0
    interactions = []
    for n_nodes, annotation in zip(num_nodes, annotations, strict=True):
        interactions.append(annotation.interactions + node_offset)
        node_offset += n_nodes

    interactions = torch.cat(interactions)
    interaction_labels = torch.cat([s.interaction_labels for s in annotations])

    return interactions, interaction_labels
