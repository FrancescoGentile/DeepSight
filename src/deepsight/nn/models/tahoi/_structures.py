##
##
##

from dataclasses import dataclass
from typing import Annotated

import torch
from torch import Tensor
from typing_extensions import Self

from deepsight.structures import Batch
from deepsight.tasks.meic import Annotations


@dataclass(frozen=True, slots=True)
class GTClusters:
    """The ground truth clusters."""

    boundary_matrix: Annotated[Tensor, "N H", float, torch.sparse_coo]
    num_nodes_per_hedge: Annotated[Tensor, "H", int]
    adjacency_matrix: Annotated[Tensor, "N N", float, torch.sparse_coo]
    dense_adjacency_matrix: Annotated[Tensor, "N N", float]

    @classmethod
    def from_annotations(cls, annotations: Batch[Annotations]) -> Self:
        num_nodes = sum(annot.interactions.size(0) for annot in annotations)
        num_hedges = sum(annot.interactions.size(1) for annot in annotations)

        boundary_matrix = torch.zeros(
            (num_nodes, num_hedges),
            dtype=torch.float,
            device=annotations.device,
        )

        node_offset = 0
        hedge_offset = 0
        for annot in annotations:
            node_limit = node_offset + annot.interactions.size(0)
            hedge_limit = hedge_offset + annot.interactions.size(1)

            boundary_matrix[
                node_offset:node_limit, hedge_offset:hedge_limit
            ] = annot.interactions

            node_offset = node_limit
            hedge_offset = hedge_limit

        num_nodes_per_hedge = boundary_matrix.sum(dim=0).int()
        boundary_matrix = boundary_matrix.to_sparse()

        adjacency_matrix = boundary_matrix.mm(boundary_matrix.T)
        # clamp_max_ is not supported for sparse tensors
        adj_values = adjacency_matrix.values()
        adj_values.clamp_max_(1.0)
        adjacency_matrix = torch.sparse_coo_tensor(
            adjacency_matrix.indices(),
            adj_values,
            size=adjacency_matrix.size(),
            is_coalesced=adjacency_matrix.is_coalesced(),
        )

        dense_adjacency_matrix = adjacency_matrix.to_dense()

        return cls(
            boundary_matrix,
            num_nodes_per_hedge,
            adjacency_matrix,
            dense_adjacency_matrix,
        )

    def compute_target_similarity_matrix(
        self,
        coboundary_matrix: Annotated[Tensor, "H' N", float] | None,
    ) -> Annotated[Tensor, "N N", float]:
        if coboundary_matrix is None:
            target = self.dense_adjacency_matrix
        else:
            num_nodes_per_hedge = coboundary_matrix.sum(dim=1).int()  # (H',)

            shared_nodes = coboundary_matrix.mm(self.boundary_matrix)  # (H', H)
            # [i, j] = True if the i-th predicted hyperedge is a subset of the j-th
            # ground truth hyperedge (i.e. all the nodes of the i-th hyperedge are also
            # in the j-th hyperedge).
            is_subset = shared_nodes == num_nodes_per_hedge.unsqueeze_(1)
            is_subset = is_subset.float()  # matrix product is not supported for bool

            # If two predicted hyperedges are both subsets of the same ground truth
            # hyperedge, they should be merged together.
            target = torch.mm(is_subset, is_subset.T)  # (H', H')
            target.clamp_max_(1.0)

            # [i, j] = True if the i-th predicted hyperedge has all the nodes of the
            # j-th ground truth hyperedge.
            has_all_nodes = shared_nodes == self.num_nodes_per_hedge  # (H', H)
            has_all_nodes = has_all_nodes.any(dim=1, keepdim=True)  # (H', 1)
            has_all_nodes = has_all_nodes.expand_as(target)  # (H', H')

            # If a predicted hyperedge has all the nodes of a ground truth hyperedge,
            # it should not be merged with any other hyperedge.
            # For example, suppose we have the following ground truth hyperedges:
            # {1, 2, 3} and {2, 3}; and the following predicted hyperedges:
            # {1, 2}, {1, 3} and {2, 3}. While it would be correct to merge
            # {2, 3} with {1, 2} and {1, 3}, since they are all subsets of the
            # first ground truth hyperedge, by merging {2, 3} we would no longer
            # be able to obtain the second ground truth hyperedge. Thus, if a
            # predicted hyperedge is equal to a ground truth hyperedge, it should
            # not be merged with any other hyperedge. Only proper subsets of the
            # ground truth hyperedge should be merged.
            target.masked_fill_(has_all_nodes, 0)

        target[target == 0] = -1.0

        return target


@dataclass(frozen=True, slots=True)
class HCStep:
    """The output of a single hierarchical clustering step.

    !!! note

        By step we mean a single iteration of the hierarchical clustering
        algorithm. Multiple iterations may be required before the algorithm
        converges.

    Attributes:
        similarity_matrix: The similarity matrix of the current step. This is
            obtained by computing the cosine similarity between the clusters
            obtained in the previous step. In the first step, this is the
            similarity matrix between the nodes.
        target_similarity_matrix: If ground truth annotations are provided and
            teacher forcing is enabled, this is the similarity matrix derived
            from the ground truth annotations. Otherwise, this is `None`.
        mask: The mask to apply to the similarity matrix when computing the
            next step and the loss.
        coboundary_matrix: The coboundary matrix derived from the similarity
            matrix.
    """

    similarity_matrix: Annotated[Tensor, "N N", float]
    target_similarity_matrix: Annotated[Tensor, "N N", float] | None
    mask: Annotated[Tensor, "N N", bool]
    coboundary_matrix: Annotated[Tensor, "M N", float] | None


@dataclass(frozen=True, slots=True)
class LayerOutput:
    steps: list[HCStep]
    num_nodes: list[int]
    num_hedges: list[int]
    num_binary_interactions: list[int]
    boundary_matrix: Annotated[Tensor, "N H", bool, torch.sparse_coo]
    interaction_logits: Annotated[Tensor, "H C", float]
    binary_interactions: Annotated[Tensor, "3 E", int]
    binary_interaction_logits: Annotated[Tensor, "E", float]


@dataclass(frozen=True, slots=True)
class Output:
    layers: list[LayerOutput]
    gt_clusters: GTClusters | None
