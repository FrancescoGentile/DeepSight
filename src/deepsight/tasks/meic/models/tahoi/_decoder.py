##
##
##

import random
from collections.abc import Iterable
from typing import Literal

import torch
import torch.nn.functional as F  # noqa
from torch import nn

from deepsight.ops.geometric import (
    add_remaining_self_loops,
    coalesce,
    scatter_softmax,
    scatter_sum,
)
from deepsight.structures.geometric import BatchMode, CombinatorialComplex
from deepsight.structures.vision import (
    BatchedBoundingBoxes,
    BatchedImages,
    BatchedSequences,
)
from deepsight.typing import SparseTensor, Tensor

from ._structures import GTClusters, HCStep


class IntraRankAttention(nn.Module):
    """Attention layer between cells of the same rank."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        num_heads: int = 1,
        bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = embed_dim

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads "
                f"({num_heads})."
            )

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = embed_dim // num_heads

        self.source_hidden_proj = nn.Linear(
            embed_dim, hidden_dim * num_heads, bias=bias
        )
        self.target_hidden_proj = nn.Linear(
            embed_dim, hidden_dim * num_heads, bias=bias
        )
        self.bridge_hidden_proj = nn.Linear(
            embed_dim, hidden_dim * num_heads, bias=bias
        )
        self.gelu = nn.GELU()
        self.hidden_attn_proj = nn.Parameter(
            torch.randn(self.num_heads, self.hidden_dim)
        )
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.message_proj = nn.Linear(2 * embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_dropout = nn.Dropout(proj_dropout)

    def forward(
        self,
        cell_features: Tensor[Literal["N D"], float],
        bridge_cell_features: Tensor[Literal["M D"], float],
        inter_neighborhood: SparseTensor[Literal["N M"], bool | int],
    ) -> Tensor[Literal["N D"], float]:
        # For each pair of cells, we need to compute the average of the features of the
        # bridge cells to which they are both bound. For example, given two nodes we
        # need to compute the average of the features of the hyperedges to which both
        # nodes belong.
        inter_neighborhood = inter_neighborhood.to_dense()
        # (N, N, M), [i, j, k] = 1 if cells i and j are both bound to bridge cell k
        shared_bridges = (
            inter_neighborhood.unsqueeze(1)
            .expand(-1, inter_neighborhood.shape[0], -1)
            .logical_and(inter_neighborhood)
        )
        shared_bridges_indices = shared_bridges.nonzero(as_tuple=False)  # (K, 3)
        shared_bridges_indices = shared_bridges_indices.T  # (3, K)

        # indices includes all unique pairs of cells that have at least one bridge cell
        # in common
        indices, bridge_cells = coalesce(
            shared_bridges_indices[:2],
            bridge_cell_features[shared_bridges_indices[2]],
            reduce="mean",
            is_sorted=True,
        )

        # If there are cells that are isolated, i.e. they are not bound to any bridge
        # cell (e.g. nodes that are not bound to any hyperedge), then no update will
        # be performed for them. To avoid this, we add a fake bridge cell (with all
        # zero features) for each isolated cell. In this way, each isolated cell will
        # be updated using only its own features.
        indices, bridge_cells = add_remaining_self_loops(
            indices,
            bridge_cells,
            size=cell_features.shape[0],
            fill_value=0.0,  # type: ignore
        )

        source_cells = cell_features[indices[0]]
        target_cells = cell_features[indices[1]]

        # From here on, we can proceed as usual with the attention mechanism.
        source_hidden = self.source_hidden_proj(source_cells)
        target_hidden = self.target_hidden_proj(target_cells)
        bridge_hidden = self.bridge_hidden_proj(bridge_cells)

        hidden = source_hidden + target_hidden + bridge_hidden
        hidden = self.gelu(hidden)
        hidden = hidden.view(-1, self.num_heads, self.hidden_dim)
        attn_logits = (hidden * self.hidden_attn_proj).sum(dim=-1)

        attn_scores = scatter_softmax(attn_logits, indices[1], dim=0)
        attn_scores = self.attn_dropout(attn_scores)

        messages = torch.cat([source_cells, bridge_cells], dim=-1)
        messages = self.message_proj(messages)
        messages = messages.view(-1, self.num_heads, self.head_dim)
        messages = messages * attn_scores.unsqueeze(-1)

        grouped_messages = scatter_sum(messages, indices[1], dim=0)
        grouped_messages = grouped_messages.view(-1, self.num_heads * self.head_dim)

        out = self.out_proj(grouped_messages)
        out = self.out_dropout(out)

        return out

    def __call__(
        self,
        cell_features: Tensor[Literal["N D"], float],
        bridge_cell_features: Tensor[Literal["M D"], float],
        inter_neighborhood: SparseTensor[Literal["N M"], bool | int],
    ) -> Tensor[Literal["N D"], float]:
        return super().__call__(cell_features, bridge_cell_features, inter_neighborhood)


class InterRankAttention(nn.Module):
    """Attention layer between cells of different ranks."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        num_heads: int = 1,
        bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = embed_dim

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads "
                f"({num_heads})."
            )

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        self.source_hidden_proj = nn.Linear(
            embed_dim, hidden_dim * num_heads, bias=bias
        )
        self.target_hidden_proj = nn.Linear(
            embed_dim, hidden_dim * num_heads, bias=bias
        )
        self.gelu = nn.GELU()
        self.hidden_attn_proj = nn.Parameter(
            torch.randn(self.num_heads, self.hidden_dim)
        )
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.message_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_dropout = nn.Dropout(proj_dropout)

    def forward(
        self,
        source_features: Tensor[Literal["N D"], float],
        target_features: Tensor[Literal["M D"], float],
        inter_neighborhood: SparseTensor[Literal["M N"], bool | int],
    ) -> Tensor[Literal["M D"], float]:
        source_cells = source_features[inter_neighborhood.indices()[1]]
        target_cells = target_features[inter_neighborhood.indices()[0]]

        source_hidden = self.source_hidden_proj(source_cells)
        target_hidden = self.target_hidden_proj(target_cells)

        hidden = source_hidden + target_hidden
        hidden = self.gelu(hidden)
        hidden = hidden.view(-1, self.num_heads, self.hidden_dim)
        attn_logits = (hidden * self.hidden_attn_proj).sum(dim=-1)

        attn_scores = scatter_softmax(
            attn_logits, inter_neighborhood.indices()[0], dim=0
        )
        attn_scores = self.attn_dropout(attn_scores)

        messages = self.message_proj(source_cells)
        messages = messages.view(-1, self.num_heads, self.head_dim)
        messages = messages * attn_scores.unsqueeze(-1)

        # Here we need to specify the output size in case a target cell is not connected
        # to any source cell. If we don't do this, the output will have a length equal
        # to the number of target cells that are connected to at least one source cell.
        # Of course, the messages for the target cells that are not connected to any
        # source cell will be all zeros.
        grouped_messages = scatter_sum(
            messages,
            inter_neighborhood.indices()[0],
            dim=0,
            dim_output_size=target_features.shape[0],
        )
        grouped_messages = grouped_messages.view(-1, self.num_heads * self.head_dim)

        out = self.out_proj(grouped_messages)
        out = self.out_dropout(out)

        return out

    def __call__(
        self,
        source_features: Tensor[Literal["N D"], float],
        target_features: Tensor[Literal["M D"], float],
        inter_neighborhood: SparseTensor[Literal["M N"], bool | int],
    ) -> Tensor[Literal["M D"], float]:
        return super().__call__(source_features, target_features, inter_neighborhood)


class HyperGraphAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        num_heads: int = 1,
        bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.node_to_edge_attn = InterRankAttention(
            embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            bias=bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )

        self.edge_to_node_attn = InterRankAttention(
            embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            bias=bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )

        self.node_to_node_attn = IntraRankAttention(
            embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            bias=bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )

        self.edge_to_edge_attn = IntraRankAttention(
            embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            bias=bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )

    def forward(self, ccc: CombinatorialComplex) -> CombinatorialComplex:
        bm = ccc.boundary_matrix(1)  # (N, E)
        cbm = ccc.coboundary_matrix(1).coalesce()  # (E, N)
        nodes = ccc.cell_features(0)  # (N, D)
        edges = ccc.cell_features(1)  # (E, D)

        node_to_edge = self.node_to_edge_attn(nodes, edges, cbm)
        node_to_node = self.node_to_node_attn(nodes, edges, bm)

        edge_to_node = self.edge_to_node_attn(edges, nodes, bm)
        edge_to_edge = self.edge_to_edge_attn(edges, nodes, cbm)

        nodes = node_to_node + edge_to_node
        edges = edge_to_edge + node_to_edge

        return ccc.replace((nodes, 0), (edges, 1))

    def __call__(self, ccc: CombinatorialComplex) -> CombinatorialComplex:
        return super().__call__(ccc)


class HyperGraphStructureLearning(nn.Module):
    """Structure learning layer for hypergraphs.

    Given a current hypergraph, this layer modifies its structure by creating
    new hyperedges and/or removing existing ones.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        num_heads: int = 1,
        bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        similarity_threshold: float = 0.5,
        teacher_forcing: float = 1.0,
    ) -> None:
        super().__init__()

        if not -1 <= similarity_threshold <= 1:
            raise ValueError(
                f"similarity_threshold ({similarity_threshold}) must be in the "
                f"range [-1, 1]."
            )

        if not 0 <= teacher_forcing <= 1:
            raise ValueError(
                f"teacher_forcing ({teacher_forcing}) must be in the range [0, 1]."
            )

        self.similarity_threshold = similarity_threshold
        self.teacher_forcing = teacher_forcing

        self.node_layernorm = nn.LayerNorm(embed_dim)
        self.edge_layernorm = nn.LayerNorm(embed_dim)

        self.node_to_node_attn = IntraRankAttention(
            embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            bias=bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )

        self.edge_to_node_attn = InterRankAttention(
            embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            bias=bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )
        self.node_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_edge_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        ccc: CombinatorialComplex,
        gt_clusters: GTClusters | None,
    ) -> tuple[CombinatorialComplex, list[HCStep]]:
        # nodes_norm = self.node_layernorm(ccc.cell_features(0))
        # edges_norm = self.edge_layernorm(ccc.cell_features(1))
        # just for testing
        nodes_norm = ccc.cell_features(0)
        edges_norm = ccc.cell_features(1)

        node_to_node = self.node_to_node_attn(
            cell_features=nodes_norm,
            bridge_cell_features=edges_norm,
            inter_neighborhood=ccc.boundary_matrix(1),
        )

        edge_to_node = self.edge_to_node_attn(
            source_features=edges_norm,
            target_features=nodes_norm,
            inter_neighborhood=ccc.boundary_matrix(1),
        )

        nodes = node_to_node + edge_to_node  # (N, D)
        nodes = self.node_proj(nodes)  # (N, D)

        num_nodes = tuple(ccc.num_cells(0, BatchMode.SEQUENCE))
        coboundary_matrix, hc_output = self._create_coboundary_matrix(
            nodes, num_nodes, gt_clusters
        )

        # The features of the new hyperedges are the average of the features of the
        # nodes that belong to them. As node features, we do not use the output of the
        # structure learning layer, but the original features of the nodes.
        nodes = ccc.cell_features(0)  # (N, D)
        edge_degree = coboundary_matrix.sum(-1, keepdim=True)  # (M, 1)
        edges = torch.mm(coboundary_matrix, nodes)  # (M, D)
        edges = edges / edge_degree
        edges = self.out_edge_proj(edges)

        coboundary_matrix = coboundary_matrix.bool()
        edge_indices = (
            torch.arange(coboundary_matrix.shape[0], device=coboundary_matrix.device)
            .unsqueeze_(-1)
            .repeat((1, coboundary_matrix.shape[1]))
        )  # (M, N)
        edge_indices[~coboundary_matrix] = 0
        max_edge_indices, _ = edge_indices.max(0)  # (N,)

        num_edges: list[int] = []
        bm_sizes: list[int] = []
        node_offset = 0
        edge_offset = 0
        for idx, nnodes in enumerate(num_nodes):
            node_limit = node_offset + nnodes
            nedges = int(max_edge_indices[node_offset:node_limit].max().item()) + 1
            if idx > 0:
                nedges -= num_edges[-1]
            num_edges.append(nedges)
            edge_limit = edge_offset + nedges
            bm_size = int(
                coboundary_matrix[edge_offset:edge_limit, node_offset:node_limit]
                .count_nonzero()
                .item()
            )
            bm_sizes.append(bm_size)

            node_offset = node_limit
            edge_offset = edge_limit

        new_ccc = CombinatorialComplex(
            (nodes, edges),
            (coboundary_matrix.T,),
            [num_nodes, tuple(num_edges)],
            [tuple(bm_sizes)],
        )

        return new_ccc, hc_output

    def _create_coboundary_matrix(
        self,
        nodes: Tensor[Literal["N D"], float],
        num_cells: Iterable[int],
        gt_clusters: GTClusters | None,
    ) -> tuple[Tensor[Literal["M N"], bool], list[HCStep]]:
        hc_output = []
        nodes = F.normalize(nodes, dim=-1)

        # The mask is used to avoid merging clusters that cannot be merged. For
        # example, clusters that belong to different samples (when using batch mode)
        # cannot be merged. A value of True in the mask means that the corresponding
        # pair of clusters cannot be merged.
        mask = nodes.new_ones((nodes.shape[0], nodes.shape[0]), dtype=torch.bool)
        node_offset = 0
        for num_nodes in num_cells:
            node_limit = node_offset + num_nodes
            mask[node_offset:node_limit, node_offset:node_limit] = False
            node_offset = node_limit
        mask_float = mask.float()

        # We set the diagonal and the lower triangular part of the similarity matrix to
        # False to avoid merging a cluster with itself or merging two clusters more than
        # once.
        mask_indices = torch.tril_indices(*mask.shape)
        mask[mask_indices[0], mask_indices[1]] = True

        pred_similarity = nodes.mm(nodes.T)
        if random.random() < self.teacher_forcing and gt_clusters is not None:
            tgt_similarity = gt_clusters.compute_target_similarity_matrix(None)
            similarity = tgt_similarity
        else:
            tgt_similarity = None
            similarity = pred_similarity

        merge = similarity > self.similarity_threshold
        merge = merge.masked_fill_(mask, False)
        merge_indices = merge.nonzero(as_tuple=False)

        coboundary_matrix = torch.zeros(
            (len(merge_indices), nodes.shape[0]),
            dtype=torch.float,
            device=nodes.device,
        )
        cluster_indices = torch.arange(len(merge_indices), device=nodes.device)
        coboundary_matrix[cluster_indices, merge_indices[:, 0]] = 1.0
        coboundary_matrix[cluster_indices, merge_indices[:, 1]] = 1.0

        if len(merge_indices) == 0:
            step = HCStep(pred_similarity, tgt_similarity, mask, None)
            hc_output.append(step)
            return coboundary_matrix, hc_output

        step = HCStep(pred_similarity, tgt_similarity, mask, coboundary_matrix)
        hc_output.append(step)

        while True:
            clusters = coboundary_matrix.mm(nodes)  # (M, D)
            clusters_dim = coboundary_matrix.sum(-1, keepdim=True)  # (M, 1)
            clusters = clusters / clusters_dim

            clusters_mask = (
                coboundary_matrix.mm(mask_float).mm(coboundary_matrix.T).bool()
            )  # (M, M)
            mask_indices = torch.tril_indices(*clusters_mask.shape)
            clusters_mask[mask_indices[0], mask_indices[1]] = True

            pred_similarity = clusters.mm(clusters.T)
            if random.random() < self.teacher_forcing and gt_clusters is not None:
                tgt_similarity = gt_clusters.compute_target_similarity_matrix(
                    coboundary_matrix
                )
                similarity = tgt_similarity
            else:
                tgt_similarity = None
                similarity = pred_similarity

            merge = similarity > self.similarity_threshold
            merge = merge.masked_fill_(clusters_mask, False)
            merge_indices = merge.nonzero(as_tuple=False)

            if len(merge_indices) == 0:
                step = HCStep(pred_similarity, tgt_similarity, clusters_mask, None)
                hc_output.append(step)
                return coboundary_matrix, hc_output

            cbm = torch.zeros(
                (len(merge_indices), coboundary_matrix.shape[0]),
                dtype=torch.float,
                device=nodes.device,
            )  # (M', M)
            cluster_indices = torch.arange(len(merge_indices), device=nodes.device)
            cbm[cluster_indices, merge_indices[:, 0]] = 1.0
            cbm[cluster_indices, merge_indices[:, 1]] = 1.0

            cbm = cbm.mm(coboundary_matrix)  # (M', N)
            cbm.clamp_(max=1.0)

            # If there are some clusters from the previous iteration that have not been
            # merged into any other cluster, we need to keep them in the coboundary
            # matrix since they could be merged in the next iterations or they could
            # represent a complete hyperedge.
            not_merged = torch.ones(
                coboundary_matrix.shape[0],
                dtype=torch.bool,
                device=coboundary_matrix.device,
            )
            not_merged[merge_indices.flatten()] = False
            not_merged_indices = not_merged.nonzero(as_tuple=True)[0]
            cbm = torch.cat([cbm, coboundary_matrix[not_merged_indices]], dim=0)

            # Now we need to remove identical clusters (i.e. rows) from the coboundary
            # matrix. For example, if we have clusters A = {1, 2}, B = {2, 3} and
            # C = {1, 3}, and we merge A and B, B and C, and A and C, then we will end
            # up with the same cluster {1, 2, 3} three times. In the next iteration,
            # each pair of these clusters will be merged again, and so on infinitely.
            coboundary_matrix: torch.Tensor = cbm.unique(sorted=True, dim=0)

            # When we added to the new cbm the clusters that have not been merged, we
            # mixed the order of the clusters, i.e. all the clusters for the first
            # sample must come first, then all the clusters for the second sample, and
            # so on. By concatenating the new cbm with the old one, we are breaking
            # this order. In torch.unique, the rows are merged in ascending order, so
            # the clusters associated to the last sample will become the first ones
            # (this is because all the first elements of their rows are zeros since they
            # only include nodes from the last sample which correspond to the last
            # columns in the coboundary matrix). To fix this, we can simply reverse the
            # order of the rows.
            coboundary_matrix = coboundary_matrix.flip(0)

            step = HCStep(
                pred_similarity, tgt_similarity, clusters_mask, coboundary_matrix
            )
            hc_output.append(step)

    def __call__(
        self,
        ccc: CombinatorialComplex,
        gt_clusters: GTClusters | None,
    ) -> tuple[CombinatorialComplex, list[HCStep]]:
        return super().__call__(ccc, gt_clusters)


class CrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        cpb_hidden_dim: int,
        bias: bool = True,
        num_heads: int = 1,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads "
                f"({num_heads})."
            )

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.norm_factor = self.head_dim**-0.5
        self.attn_dropout = attn_dropout

        self.nq_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.eq_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=bias)

        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, cpb_hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(cpb_hidden_dim, num_heads, bias=False),
        )

        self.nout_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.eout_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_dropout = nn.Dropout(proj_dropout)

    def forward(
        self,
        ccc: CombinatorialComplex,
        images: BatchedSequences,
        relative_distances: Tensor[Literal["B N P 2"], float],
    ) -> CombinatorialComplex:
        B, N, P, _ = relative_distances.shape  # noqa: N806

        node_cpb: torch.Tensor = self.cpb_mlp(relative_distances)  # (B, N, P, H)
        node_cpb = node_cpb.permute(0, 3, 1, 2)  # (B, H, N, P)
        cbm = ccc.coboundary_matrix(1, BatchMode.STACK)  # (B, E, N)
        cbm = cbm.to_dense().unsqueeze(1)  # (B, 1, E, N)
        edge_cpb = torch.matmul(cbm.to(node_cpb.dtype), node_cpb)  # (B, H, E, P)

        nq = self.nq_proj(ccc.cell_features(0, BatchMode.STACK))
        nq = nq.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        eq = self.eq_proj(ccc.cell_features(1, BatchMode.STACK))
        eq = eq.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        kv = self.kv_proj(images.data)  # (B, P, 2D)
        kv = kv.view(B, P, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, B, H, P, Dh)
        k, v = kv.unbind(0)  # (B, H, P, Dh)

        mask = images.mask[:, None, None]  # (B, 1, 1, P)
        node_attn_mask = node_cpb.masked_fill_(mask, -torch.inf)
        edge_attn_mask = edge_cpb.masked_fill_(mask, -torch.inf)

        node_out = F.scaled_dot_product_attention(
            nq, k, v, node_attn_mask, self.attn_dropout if self.training else 0.0
        )  # (B, H, N, Dh)

        edge_out = F.scaled_dot_product_attention(
            eq, k, v, edge_attn_mask, self.attn_dropout if self.training else 0.0
        )  # (B, H, E, Dh)

        node_out = node_out.transpose(1, 2)  # (B, N, H, Dh)
        node_out = node_out.flatten(2)  # (B, N, D)
        node_out = self.nout_proj(node_out)
        node_out = self.out_dropout(node_out)

        edge_out = edge_out.transpose(1, 2)  # (B, E, H, Dh)
        edge_out = edge_out.flatten(2)
        edge_out = self.eout_proj(edge_out)
        edge_out = self.out_dropout(edge_out)

        return ccc.replace((node_out, 0), (edge_out, 1))

    def __call__(
        self,
        ccc: CombinatorialComplex,
        images: BatchedSequences,
        relative_distances: Tensor[Literal["B N P 2"], float],
    ) -> CombinatorialComplex:
        return super().__call__(ccc, images, relative_distances)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        cpb_hidden_dim: int,
        num_heads: int = 1,
        bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        similarity_threshold: float = 0.0,
    ) -> None:
        super().__init__()

        self.structure_learning = HyperGraphStructureLearning(
            embed_dim,
            num_heads=num_heads,
            bias=bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            similarity_threshold=similarity_threshold,
        )

        self.hgat_node_layernorm = nn.LayerNorm(embed_dim)
        self.hgat_edge_layernorm = nn.LayerNorm(embed_dim)
        self.hgat = HyperGraphAttention(
            embed_dim,
            num_heads=num_heads,
            bias=bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )

        self.ca_node_layernorm = nn.LayerNorm(embed_dim)
        self.ca_edge_layernorm = nn.LayerNorm(embed_dim)
        self.cross_attn = CrossAttention(
            embed_dim,
            cpb_hidden_dim=cpb_hidden_dim,
            bias=bias,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )

        self.ffn_node_layernorm = nn.LayerNorm(embed_dim)
        self.ffn_edge_layernorm = nn.LayerNorm(embed_dim)
        self.node_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4, bias=bias),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim, bias=bias),
            nn.Dropout(proj_dropout),
        )
        self.edge_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4, bias=bias),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim, bias=bias),
            nn.Dropout(proj_dropout),
        )

    def forward(
        self,
        ccc: CombinatorialComplex,
        images: BatchedSequences,
        relative_distances: Tensor[Literal["B N P 2"], float],
        gt_clusters: GTClusters | None,
    ) -> tuple[CombinatorialComplex, list[HCStep]]:
        # Structure learning
        ccc, hc_output = self.structure_learning(ccc, gt_clusters)
        nodes, edges = ccc.cell_features(0), ccc.cell_features(1)

        # Hypergraph attention
        nodes_norm = self.hgat_node_layernorm(nodes)
        edges_norm = self.hgat_edge_layernorm(edges)
        ccc = ccc.replace((nodes_norm, 0), (edges_norm, 1))
        ccc = self.hgat(ccc)
        nodes = ccc.cell_features(0) + nodes
        edges = ccc.cell_features(1) + edges

        # Cross-attention
        nodes_norm = self.ca_node_layernorm(nodes)
        edges_norm = self.ca_edge_layernorm(edges)
        ccc = ccc.replace((nodes_norm, 0), (edges_norm, 1))
        ccc = self.cross_attn(ccc, images, relative_distances)
        nodes = ccc.cell_features(0) + nodes
        edges = ccc.cell_features(1) + edges

        # FFN
        nodes_norm = self.ffn_node_layernorm(nodes)
        edges_norm = self.ffn_edge_layernorm(edges)
        nodes = self.node_ffn(nodes_norm) + nodes
        edges = self.edge_ffn(edges_norm) + edges

        ccc = ccc.replace((nodes, 0), (edges, 1))
        return ccc, hc_output

    def __call__(
        self,
        ccc: CombinatorialComplex,
        images: BatchedSequences,
        relative_distances: Tensor[Literal["B N P 2"], float],
        gt_clusters: GTClusters | None,
    ) -> tuple[CombinatorialComplex, list[HCStep]]:
        return super().__call__(ccc, images, relative_distances, gt_clusters)


class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        cpb_hidden_dim: int,
        num_heads: int = 1,
        bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        similarity_thresholds: float | Iterable[float] = 0.0,
        num_layers: int = 1,
    ) -> None:
        super().__init__()

        if isinstance(similarity_thresholds, (int, float)):
            similarity_thresholds = [float(similarity_thresholds)] * num_layers
        else:
            similarity_thresholds = list(similarity_thresholds)
            if len(similarity_thresholds) != num_layers:
                raise ValueError(
                    f"similarity_thresholds must be a float or an iterable of "
                    f"floats with length equal to num_layers ({num_layers})."
                )

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    embed_dim,
                    cpb_hidden_dim,
                    num_heads=num_heads,
                    bias=bias,
                    attn_dropout=attn_dropout,
                    proj_dropout=proj_dropout,
                    similarity_threshold=threshold,
                )
                for threshold in similarity_thresholds
            ]
        )

    def forward(
        self,
        ccc: CombinatorialComplex,
        boxes: BatchedBoundingBoxes,
        images: BatchedImages,
        gt_clusters: GTClusters | None,
    ) -> list[tuple[CombinatorialComplex, list[HCStep]]]:
        relative_distances = _compute_relative_distances(boxes, images)
        patches = images.to_sequences()

        outputs = []
        for layer in self.layers:
            ccc, hc_output = layer(ccc, patches, relative_distances, gt_clusters)
            outputs.append((ccc, hc_output))

        return outputs

    def __call__(
        self,
        ccc: CombinatorialComplex,
        boxes: BatchedBoundingBoxes,
        images: BatchedImages,
        gt_clusters: GTClusters | None,
    ) -> list[tuple[CombinatorialComplex, list[HCStep]]]:
        return super().__call__(ccc, boxes, images, gt_clusters)


# --------------------------------------------------------------------------- #
# Private helper functions
# --------------------------------------------------------------------------- #


def _compute_relative_distances(
    boxes: BatchedBoundingBoxes,  # (B, Q, 4)
    images: BatchedImages,
) -> Tensor[Literal["B Q HW 2"], float]:
    not_mask = ~images.mask  # (B, H, W)
    x = not_mask.cumsum(2, dtype=torch.float) - 1  # (B, H, W)
    y = not_mask.cumsum(1, dtype=torch.float) - 1  # (B, H, W)

    patch_cx = torch.stack([x, y], dim=3)  # (B, H, W, 2)
    patch_cx = patch_cx.view(patch_cx.shape[0], 1, -1, 2)  # (B, 1, HW, 2)

    box_cx = boxes.denormalize().to_cxcywh().coordinates[..., :2]  # (B, Q, 2)
    box_cx = box_cx.unsqueeze(2)  # (B, Q, 1, 2)

    distances = patch_cx - box_cx  # (B, Q, HW, 2)
    distances = torch.sign(distances) * torch.log(1 + torch.abs(distances))

    return distances
