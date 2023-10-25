##
##
##

from collections.abc import Iterable
from typing import Annotated

import torch
import torch.nn.functional as F  # noqa
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from deepsight.models import Criterion as _Criterion
from deepsight.structures import Batch
from deepsight.tasks.meic import Annotations

from ._structures import HCStep, Output


class Criterion(_Criterion[Output, Annotations]):
    def __init__(
        self,
        layers: int | Iterable[int],
        jaccard_weight: float = 1.0,
        similarity_weight: float = 1.0,
        focal_loss_alpha: float = 0.25,
        focal_loss_gamma: float = 2.0,
        focal_loss_weight: float = 1.0,
        bce_loss_weight: float = 1.0,
        hc_loss_weight: float = 1.0,
    ) -> None:
        """Initialize a criterion.

        Args:
            layers: The indices of the layers whose outputs should be used to
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
            focal_loss_weight: The weight applied to the focal loss.
            bce_loss_weight: The weight applied to the binary cross entropy loss.
            hc_loss_weight: The weight applied to the hierarchical clustering loss.
        """
        self.layers = tuple(layers) if isinstance(layers, Iterable) else (layers,)

        # Weights to compute the cost of associating each predicted multi-entity
        self.jaccard_weight = jaccard_weight
        self.similarity_weight = similarity_weight

        # Focal loss parameters
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma

        # Weights for each loss
        self.focal_loss_weight = focal_loss_weight
        self.bce_loss_weight = bce_loss_weight
        self.hc_loss_weight = hc_loss_weight

    @property
    def losses(self) -> Iterable[str]:
        losses = []
        for layer in self.layers:
            losses.extend(
                [
                    f"focal_loss_{layer}",
                    f"bce_loss_{layer}",
                    f"hc_loss_{layer}",
                ]
            )
        return losses

    def compute(
        self,
        output: Output,
        annotations: Batch[Annotations],
    ) -> dict[str, Tensor]:
        losses = {}
        gt_num_edges = [ann.interactions.shape[1] for ann in annotations]
        batched_bm, batched_labels, batched_binary = _batch(annotations)
        batched_adj = torch.mm(batched_bm, batched_bm.T)  # (N, N)
        batched_adj.clamp_(max=1)

        for layer in self.layers:
            indices = self._compute_assignment(
                pred_bbm=output[layer].boundary_matrix,
                gt_bbm=batched_bm,
                pred_labels=output[layer].interaction_logits,
                gt_labels=batched_labels,
                pred_num_edges=output[layer].num_edges,
                gt_num_edges=gt_num_edges,
            )
            batched_indices = self._batch_indices(
                indices, output[layer].num_edges, gt_num_edges
            )
            bce_indices = self._compute_bce_indices(
                batched_indices, output[layer].binary_interactions, batched_binary
            )

            losses[f"focal_loss_{layer}"] = self._compute_focal_loss(
                batched_indices, output[layer].interaction_logits, batched_labels
            )
            losses[f"bce_loss_{layer}"] = self._compute_bce_loss(
                bce_indices, output[layer].binary_interaction_logits, batched_binary
            )
            losses[f"hc_loss_{layer}"] = self._compute_hc_loss(
                batched_adj, batched_bm, output[layer].steps
            )

        return losses

    # ----------------------------------------------------------------------- #
    # Private methods
    # ----------------------------------------------------------------------- #

    @torch.no_grad()
    def _compute_assignment(
        self,
        pred_bbm: Annotated[Tensor, "N H", int, torch.sparse_coo],
        gt_bbm: Annotated[Tensor, "N H'", int, torch.sparse_coo],
        pred_labels: Annotated[Tensor, "H C", float],
        gt_labels: Annotated[Tensor, "H' C", float],
        pred_num_edges: list[int],
        gt_num_edges: list[int],
    ) -> list[tuple[Tensor, Tensor]]:
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

        sample_costs = []
        pred_offset, gt_offset = 0, 0
        for pred_nedges, gt_nedges in zip(pred_num_edges, gt_num_edges, strict=True):
            pred_limit = pred_offset + pred_nedges
            gt_limit = gt_offset + gt_nedges
            sample_cost = cost[pred_offset:pred_limit, gt_offset:gt_limit]
            sample_costs.append(sample_cost.cpu())

            pred_offset = pred_limit
            gt_offset = gt_limit

        indices = [linear_sum_assignment(cost) for cost in sample_costs]
        return [(torch.as_tensor(i), torch.as_tensor(j)) for i, j in indices]

    def _batch_indices(
        self,
        indices: list[tuple[Tensor, Tensor]],
        pred_num_edges: list[int],
        gt_num_edges: list[int],
    ) -> tuple[Tensor, Tensor]:
        pred_offset, gt_offset = 0, 0
        pred_indices, gt_indices = [], []

        for idx, (i, j) in enumerate(indices):
            pred_indices.append(i.add_(pred_offset))
            gt_indices.append(j.add_(gt_offset))

            pred_offset += pred_num_edges[idx]
            gt_offset += gt_num_edges[idx]

        return torch.cat(pred_indices), torch.cat(gt_indices)

    def _compute_focal_loss(
        self,
        indices: tuple[Tensor, Tensor],
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
        indices: tuple[Tensor, Tensor],
        pred_binary: Annotated[Tensor, "E 3", int],
        gt_binary: Annotated[Tensor, "E' 3", int],
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def _compute_bce_loss(
        self,
        indices: tuple[Tensor, Tensor],
        logits: Annotated[Tensor, "E", float],
        gt_labels: Annotated[Tensor, "E'", float],
    ) -> Annotated[Tensor, "", float]:
        targets = torch.zeros_like(logits)
        targets[indices[0]].copy_(gt_labels[indices[1]])

        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
        return loss * self.bce_loss_weight

    def _compute_hc_loss(
        self,
        adj: Annotated[Tensor, "N N", int, torch.sparse_coo],
        boundary_matrix: Annotated[Tensor, "N H", int, torch.sparse_coo],
        steps: list[HCStep],
    ) -> Annotated[Tensor, "", float]:
        total_loss = 0.0
        nodes_per_edge = torch.sparse.sum(boundary_matrix, dim=0).to_dense()  # (H,)
        for idx, step in enumerate(steps):
            if idx == 0:
                target = adj.to_dense()
            else:
                cbm = steps[idx - 1].coboundary_matrix  # (M, N)
                assert cbm is not None
                target = cbm.mm(adj).mm(cbm.T)  # (M, M)
                target.clamp_(max=1).to_dense()

                shared_nodes = cbm.mm(boundary_matrix).to_dense()  # (M, H)
                has_all_nodes = shared_nodes == nodes_per_edge.unsqueeze(0)  # (M, H)
                has_all_nodes = has_all_nodes.any(dim=1, keepdim=True)  # (M, 1)
                has_all_nodes = has_all_nodes.expand_as(target)  # (M, M)
                # If an edge already has all the nodes, it should not be merged with
                # any other edge.
                target.masked_fill_(has_all_nodes, 0)

            target[target == 0] = -1

            loss = torch.sigmoid(step.similarity_matrix * target)
            loss = -loss.masked_select(~step.mask).mean()
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


def _batch(
    annotations: Batch[Annotations],
) -> tuple[
    Annotated[Tensor, "N H", int, torch.sparse_coo],
    Annotated[Tensor, "H C", float],
    Annotated[Tensor, "E 3", int],
]:
    total_nodes = sum(ann.interactions.shape[0] for ann in annotations)
    total_edges = sum(ann.interactions.shape[1] for ann in annotations)

    bm = torch.zeros(
        (total_nodes, total_edges),
        dtype=torch.bool,
        device=annotations.device,
    )

    node_offset = 0
    edge_offset = 0
    for ann in annotations:
        node_limit = node_offset + ann.interactions.shape[0]
        edge_limit = edge_offset + ann.interactions.shape[1]
        bm[node_offset:node_limit, edge_offset:edge_limit].copy_(ann.interactions)

    bm = bm.to_sparse().int()

    labels = torch.cat([ann.interaction_labels for ann in annotations], dim=0)
    labels = labels.float()

    binary = torch.cat([ann.binary_interactions for ann in annotations], dim=0)

    return bm, labels, binary
