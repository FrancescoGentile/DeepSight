##
##
##

from collections.abc import Iterable
from typing import Annotated

import torch
import torch.nn.functional as F  # noqa
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from deepsight import utils
from deepsight.models import Criterion as _Criterion
from deepsight.models import LossInfo
from deepsight.tasks.meic import Annotations
from deepsight.typing import Configs, Configurable, Losses
from deepsight.utils import Batch

from ._structures import GTClusters, HCStep, Output


class Criterion(_Criterion[Output, Annotations], Configurable):
    def __init__(
        self,
        layer_indices: int | Iterable[int],
        jaccard_weight: float = 1.0,
        similarity_weight: float = 1.0,
        focal_loss_alpha: float = 0.25,
        focal_loss_gamma: float = 2.0,
        similarity_thresholds: float | Iterable[float] = 0.5,
        focal_loss_weight: float = 1.0,
        bce_loss_weight: float = 1.0,
        hc_loss_weight: float = 1.0,
    ) -> None:
        """Initialize a criterion.

        Args:
            layer_indices: The indices of the layers whose outputs should be used to
                compute the losses. The name of the losses will be postfixed
                with the index of the layer they are computed from.
            jaccard_weight: The weight applied to the jaccard index cost. Note that
                the cost is previously negated, so a positive weight should be used.
            similarity_weight: The weight applied to the class similarity cost. Note
                that the cost is previously negated, so a positive weight should be
                used.
            focal_loss_alpha: The alpha parameter of the focal loss. If negative, no
                focal loss is applied.
            focal_loss_gamma: The gamma parameter of the focal loss.
            similarity_thresholds: The similarity thresholds used to compute the
                hierarchical clustering loss. If a single value is provided, it is
                used for all layers. Otherwise, a value must be provided for each
                layer.
            focal_loss_weight: The weight applied to the focal loss.
            bce_loss_weight: The weight applied to the binary cross entropy loss.
            hc_loss_weight: The weight applied to the hierarchical clustering loss.
        """
        self.layer_indices = utils.to_tuple(layer_indices)

        # Weights to compute the cost of associating each predicted multi-entity
        self.jaccard_weight = jaccard_weight
        self.similarity_weight = similarity_weight

        # Focal loss parameters
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma

        # Hierarchical clustering parameters
        if isinstance(similarity_thresholds, Iterable):
            self.similarity_thresholds = tuple(similarity_thresholds)
            if len(self.similarity_thresholds) != len(self.layer_indices):
                raise ValueError(
                    "The number of similarity thresholds must be equal to the number "
                    "of layers."
                )
        else:
            self.similarity_thresholds = (similarity_thresholds,) * len(
                self.layer_indices
            )

        # Weights for each loss
        self.focal_loss_weight = focal_loss_weight
        self.bce_loss_weight = bce_loss_weight
        self.hc_loss_weight = hc_loss_weight

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_losses_info(self) -> tuple[LossInfo, ...]:
        losses = []
        for layer_idx in self.layer_indices:
            losses.extend([
                LossInfo(f"focal_loss_{layer_idx}", self.focal_loss_weight),
                LossInfo(f"bce_loss_{layer_idx}", self.bce_loss_weight),
                LossInfo(f"hc_loss_{layer_idx}", self.hc_loss_weight),
            ])

        return tuple(losses)

    def get_configs(self, recursive: bool) -> Configs:
        return {
            "layer_indices": self.layer_indices,
            "jaccard_weight": self.jaccard_weight,
            "similarity_weight": self.similarity_weight,
            "focal_loss_alpha": self.focal_loss_alpha,
            "focal_loss_gamma": self.focal_loss_gamma,
            "similarity_thresholds": self.similarity_thresholds,
            "focal_loss_weight": self.focal_loss_weight,
            "bce_loss_weight": self.bce_loss_weight,
            "hc_loss_weight": self.hc_loss_weight,
        }

    def compute(self, output: Output, annotations: Batch[Annotations]) -> Losses:
        losses = {}
        gt_num_hedges = [ann.interactions.size(1) for ann in annotations]
        batched_labels = torch.cat([ann.interaction_labels for ann in annotations])
        batched_binary = []
        node_offset, hedge_offset = 0, 0
        for ann in annotations:
            first_entity = ann.binary_interactions[0] + node_offset
            second_entity = ann.binary_interactions[1] + node_offset
            cluster = ann.binary_interactions[2] + hedge_offset

            binary = torch.stack([first_entity, second_entity, cluster], dim=0)
            batched_binary.append(binary)

            node_offset += ann.interactions.size(0)
            hedge_offset += ann.interactions.size(1)

        batched_binary = torch.cat(batched_binary, dim=1)

        if output.gt_clusters is not None:
            gt_clusters = output.gt_clusters
        else:
            gt_clusters = GTClusters.from_annotations(annotations)

        for layer_idx in self.layer_indices:
            with torch.no_grad():
                batched_indices = self._compute_assignment(
                    pred_bbm=output.layers[layer_idx].boundary_matrix.float(),
                    gt_bbm=gt_clusters.boundary_matrix,
                    pred_labels=output.layers[layer_idx].interaction_logits,
                    gt_labels=batched_labels,
                    pred_num_hedges=output.layers[layer_idx].num_hedges,
                    gt_num_hedges=gt_num_hedges,
                )

                bce_indices = self._compute_bce_indices(
                    batched_indices,
                    output.layers[layer_idx].binary_interactions,
                    batched_binary,
                )

            losses[f"focal_loss_{layer_idx}"] = self._compute_focal_loss(
                batched_indices,
                output.layers[layer_idx].interaction_logits,
                batched_labels,
            )

            losses[f"bce_loss_{layer_idx}"] = self._compute_bce_loss(
                bce_indices, output.layers[layer_idx].binary_interaction_logits
            )

            losses[f"hc_loss_{layer_idx}"] = self._compute_hc_loss(
                gt_clusters,
                output.layers[layer_idx].steps,
                self.similarity_thresholds[layer_idx],
            )

        return losses

    # ----------------------------------------------------------------------- #
    # Private methods
    # ----------------------------------------------------------------------- #

    def _compute_assignment(
        self,
        pred_bbm: Annotated[Tensor, "N H", int, torch.sparse_coo],
        gt_bbm: Annotated[Tensor, "N H'", float, torch.sparse_coo],
        pred_labels: Annotated[Tensor, "H C", float],
        gt_labels: Annotated[Tensor, "H' C", float],
        pred_num_hedges: list[int],
        gt_num_hedges: list[int],
    ) -> tuple[Annotated[Tensor, "I", int], Annotated[Tensor, "I", int]]:
        # Compute jaccard index
        intersection = torch.mm(pred_bbm.T, gt_bbm).to_dense()  # (H, H')
        pred_num_nodes = torch.sparse.sum(pred_bbm, dim=0).to_dense()  # (H,)
        gt_num_nodes = torch.sparse.sum(gt_bbm, dim=0).to_dense()  # (H',)
        union = pred_num_nodes.unsqueeze(1) + gt_num_nodes.unsqueeze(0) - intersection
        jaccard = intersection / union  # (H, H')

        # Compute class similarity
        similarity = torch.mm(pred_labels, gt_labels.T)  # (H, H')

        # Compute cost
        cost = (
            self.jaccard_weight * jaccard.neg_()
            + self.similarity_weight * similarity.neg_()
        )

        pred_indices, gt_indices = [], []
        pred_offset, gt_offset = 0, 0
        for pred_nhedges, gt_nhedges in zip(
            pred_num_hedges, gt_num_hedges, strict=True
        ):
            pred_limit = pred_offset + pred_nhedges
            gt_limit = gt_offset + gt_nhedges
            sample_cost = cost[pred_offset:pred_limit, gt_offset:gt_limit].cpu()
            i, j = linear_sum_assignment(sample_cost)
            i = torch.as_tensor(i, device=pred_bbm.device) + pred_offset
            j = torch.as_tensor(j, device=pred_bbm.device) + gt_offset
            pred_indices.append(i)
            gt_indices.append(j)

            pred_offset = pred_limit
            gt_offset = gt_limit

        return torch.cat(pred_indices), torch.cat(gt_indices)

    def _compute_focal_loss(
        self,
        indices: tuple[Annotated[Tensor, "I", int], Annotated[Tensor, "I", int]],
        logits: Annotated[Tensor, "H C", float],
        gt_labels: Annotated[Tensor, "H' C", float],
    ) -> Annotated[Tensor, "", float]:
        alpha = self.focal_loss_alpha
        gamma = self.focal_loss_gamma

        targets = torch.zeros_like(logits)
        targets[indices[0]].copy_(gt_labels[indices[1]])

        # Compute focal loss
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = ((1 - pt) ** gamma) * ce_loss

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean() * self.focal_loss_weight

    def _compute_bce_indices(
        self,
        indices: tuple[Annotated[Tensor, "I", int], Annotated[Tensor, "I", int]],
        pred_binary: Annotated[Tensor, "3 E", int],
        gt_binary: Annotated[Tensor, "3 E'", int],
    ) -> tuple[Annotated[Tensor, "I'", int], Annotated[Tensor, "I'", int]]:
        pred_cluster_indices = pred_binary[2]  # (E,)
        matched = pred_cluster_indices.unsqueeze(1) == indices[0]  # (E, I)
        matched_indices = matched.nonzero(as_tuple=True)  # (M,)

        pred_gt_cluster_indices = indices[1][matched_indices[1]].unsqueeze_(0)  # (1, M)
        pred_entity_indices = pred_binary[:2, matched_indices[0]]  # (2, M)
        pred_binary = torch.cat([pred_entity_indices, pred_gt_cluster_indices], dim=0)

        pred_binary = pred_binary.unsqueeze_(2)  # (3, M, 1)
        gt_binary = gt_binary.unsqueeze(1)  # (3, 1, E')
        matched2 = pred_binary == gt_binary  # (3, M, E')
        matched2 = matched2.all(dim=0)  # (M, E')
        matched2_indices = matched2.nonzero(as_tuple=True)  # (M',)

        return matched_indices[0][matched2_indices[0]], matched2_indices[1]

    def _compute_bce_loss(
        self, indices: tuple[Tensor, Tensor], logits: Annotated[Tensor, "E", float]
    ) -> Annotated[Tensor, "", float]:
        targets = torch.zeros_like(logits)
        targets[indices[0]] = 1.0

        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
        return loss * self.bce_loss_weight

    def _compute_hc_loss(
        self,
        gt_clusters: GTClusters,
        steps: list[HCStep],
        similarity_threshold: float,
    ) -> Annotated[Tensor, "", float]:
        total_loss = 0.0
        for step in steps:
            if step.target_similarity_matrix is not None:
                target = step.target_similarity_matrix
            else:
                target = gt_clusters.compute_target_similarity_matrix(
                    step.coboundary_matrix
                )

            # If two cluster must be grouped together, their similarity must be
            # greater than the similarity threshold, i.e. threshold - similarity < 0,
            # otherwise, the similarity must be less than the similarity threshold,
            # i.e. similarity - threshold < 0.
            loss = similarity_threshold - step.similarity_matrix
            loss = loss * target
            loss = torch.maximum(loss, torch.zeros_like(loss))
            loss = loss.masked_select(~step.mask).mean()
            total_loss = total_loss + loss

        return total_loss * self.hc_loss_weight  # type: ignore


# --------------------------------------------------------------------------- #
# Private functions
# --------------------------------------------------------------------------- #

# We have for each sample a matrix (N, H) where N is the number of nodes and H
# is the number of multi-entity interactions. We also have a matrix (H, C) that
# associate each multi-entity interaction to the class it belongs to.
# We can batch both matrix and obtain a matrix (BN, BH) where BN is the total
# number of nodes and BH is the total number of multi-entity interactions,
# and a matrix (BH, C).
# At this point we can compute the cost of associating each predicted multi-entity
# interactions to the ground truth multi-entity interactions. To do this, we need
# to compute the cost of associating each predicted multi-entity interaction to
# each ground truth multi-entity interaction. Two different cost functions are used:
# (1) the class similarity and (2) the jaccard index (number of common nodes over
# the number of nodes in the union of the two multi-entity interactions). The total
# cost is a weighted sum of the two costs. Then, hungarian-matching is used to find
# the optimal assignment.
# Once the optimal assignment is found, we can compute the focal loss.
