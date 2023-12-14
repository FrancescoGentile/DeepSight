##
##
##

from collections.abc import Iterable
from typing import Annotated

import torch
import torch.nn.functional as F  # noqa
from torch import Tensor

from deepsight.training import Batch, LossInfo
from deepsight.training import Criterion as _Criterion
from deepsight.training.hoic import Annotations
from deepsight.typing import Configs, Configurable, Losses

from ._structures import Output


class Criterion(_Criterion[Output, Annotations], Configurable):
    def __init__(
        self,
        layer_indices: Iterable[int] | int,
        suppression_alpha: float = 0.25,
        suppression_gamma: float = 2.0,
        classification_alpha: float = 0.25,
        classification_gamma: float = 2.0,
        suppression_weight: float = 1.0,
        classification_weight: float = 1.0,
    ) -> None:
        self.layer_indices = (
            (layer_indices,) if isinstance(layer_indices, int) else tuple(layer_indices)
        )

        self.suppression_alpha = suppression_alpha
        self.suppression_gamma = suppression_gamma
        self.classification_alpha = classification_alpha
        self.classification_gamma = classification_gamma
        self.suppression_weight = suppression_weight
        self.classification_weight = classification_weight

    # ----------------------------------------------------------------------- #
    # Public methods
    # ----------------------------------------------------------------------- #

    def get_configs(self, recursive: bool) -> Configs:
        return {
            "suppression_alpha": self.suppression_alpha,
            "suppression_gamma": self.suppression_gamma,
            "classification_alpha": self.classification_alpha,
            "classification_gamma": self.classification_gamma,
            "suppression_weight": self.suppression_weight,
            "classification_weight": self.classification_weight,
        }

    def get_losses_info(self) -> tuple[LossInfo, ...]:
        losses = []
        for layer_index in self.layer_indices:
            losses.append(
                LossInfo(
                    name=f"suppression_loss_{layer_index}",
                    weight=self.suppression_weight,
                )
            )
            losses.append(
                LossInfo(
                    name=f"classification_loss_{layer_index}",
                    weight=self.classification_weight,
                )
            )

        return tuple(losses)

    def compute(self, output: Output, annotations: Batch[Annotations]) -> Losses:
        gt_interactions, gt_labels = _batch(annotations, output.num_nodes)
        matched = output.indices.unsqueeze(1) == gt_interactions  # (E, E', 2)
        matched = matched.all(dim=2)  # (E, E')
        matched_pred, matched_target = torch.nonzero(matched, as_tuple=True)

        losses = {}
        for idx, layer_output in enumerate(output.layers):
            # Compute the suppression loss
            suppression_target = torch.zeros_like(layer_output.suppress_logits)
            suppression_target[matched_pred] = 1.0
            suppression_loss = focal_loss(
                layer_output.suppress_logits,
                suppression_target,
                alpha=self.suppression_alpha,
                gamma=self.suppression_gamma,
                # weight=torch.tensor([20.0], device=output.label_logits.device),
            )
            suppression_loss = suppression_loss.mean() * self.suppression_weight

            # Compute the classification loss
            # we compute the classification loss only for the predicted interactions
            # that are matched with the ground-truth interactions
            cls_loss = focal_loss(
                layer_output.label_logits[matched_pred],
                gt_labels[matched_target],
                alpha=self.classification_alpha,
                gamma=self.classification_gamma,
            )
            if output.interaction_mask is not None:
                cls_loss = cls_loss.masked_select(
                    ~output.interaction_mask[matched_pred]
                )
            cls_loss = cls_loss.mean() * self.classification_weight

            losses[f"suppression_loss_{idx}"] = suppression_loss
            losses[f"classification_loss_{idx}"] = cls_loss

        return losses


# --------------------------------------------------------------------------- #
# Private helper functions
# --------------------------------------------------------------------------- #


def _batch(
    annotations: Batch[Annotations], num_nodes: list[int]
) -> tuple[Annotated[Tensor, "E 2", int], Annotated[Tensor, "E C", float]]:
    node_offset = 0
    interactions = []
    for n_nodes, annotation in zip(num_nodes, annotations, strict=True):
        if node_offset > 0:
            interactions.append(annotation.interaction_indices + node_offset)
        else:
            interactions.append(annotation.interaction_indices)
        node_offset += n_nodes

    interactions = torch.cat(interactions)
    interaction_labels = torch.cat([s.interaction_labels for s in annotations])

    return interactions, interaction_labels


def focal_loss(
    predictions: Annotated[Tensor, "N C", float],
    targets: Annotated[Tensor, "N C", float],
    alpha: float = 0.25,
    gamma: float = 2.0,
    weight: Annotated[Tensor, "C", float] | None = None,
) -> Annotated[Tensor, "N C", float]:
    p = torch.sigmoid(predictions)
    p_t = p * targets + (1 - p) * (1 - targets)
    ce_loss = F.binary_cross_entropy_with_logits(
        predictions, targets, reduction="none", weight=weight
    )
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss
