##
##
##

from collections.abc import Iterable
from typing import Annotated

import torch
from torch import Tensor
from torchvision.ops import sigmoid_focal_loss

from deepsight.nn.models import Criterion as _Criterion
from deepsight.structures import Batch
from deepsight.tasks.hoic import Annotations
from deepsight.typing import Configurable, JSONPrimitive

from ._structures import Output


class Criterion(_Criterion[Output, Annotations], Configurable):
    def __init__(self, focal_alpha: float = 0.25, focal_gamma: float = 2.0) -> None:
        super().__init__()

        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @property
    def losses(self) -> Iterable[str]:
        return ["focal_loss"]

    def compute(
        self, output: Output, annotations: Batch[Annotations]
    ) -> dict[str, Annotated[Tensor, "", float]]:
        pred_indices, gt_indices = _batch_indices(output, annotations)

        matched = pred_indices.unsqueeze_(1) == gt_indices  # (I, I', 2)
        matched = matched.all(dim=2)  # (I, I')
        matched_pred, matched_gt = matched.nonzero(as_tuple=True)  # (N,) (N,)

        pred_logits = torch.cat(output.ho_logits)
        interaction_labels = torch.cat([ann.interaction_labels for ann in annotations])
        gt_labels = torch.zeros_like(pred_logits)
        gt_labels[matched_pred] = interaction_labels[matched_gt].to(gt_labels.dtype)

        focal_loss = sigmoid_focal_loss(
            pred_logits, gt_labels, self.focal_alpha, self.focal_gamma, reduction="mean"
        )

        return {"focal_loss": focal_loss}

    def get_config(self) -> JSONPrimitive:
        return {
            "focal_alpha": self.focal_alpha,
            "focal_gamma": self.focal_gamma,
        }


# --------------------------------------------------------------------------- #
# Private helper functions
# --------------------------------------------------------------------------- #


def _batch_indices(
    output: Output, annotations: Batch[Annotations]
) -> tuple[Annotated[Tensor, "I 2", int], Annotated[Tensor, "I' 2", int]]:
    pred_indices = []
    gt_indices = []

    entity_offset = 0
    for idx in range(len(annotations)):
        pred = output.ho_indices[idx]
        gt = annotations[idx].interaction_indices

        if entity_offset > 0:
            pred = pred + entity_offset
            gt = gt + entity_offset

        pred_indices.append(pred)
        gt_indices.append(gt)

        entity_offset += output.num_entities[idx]

    return torch.cat(pred_indices), torch.cat(gt_indices)
