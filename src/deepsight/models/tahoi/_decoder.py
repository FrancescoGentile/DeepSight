##
##
##

from collections.abc import Iterable
from numbers import Number
from typing import Annotated

import torch
import torch.nn.functional as F  # noqa
from torch import Tensor, nn

from deepsight.structures import BatchedBoundingBoxes, BatchMode, CombinatorialComplex
from deepsight.utils.geometric import (
    add_remaining_self_loops,
    coalesce,
    scatter_softmax,
    scatter_sum,
)

from ._structures import HCStep


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
        cell_features: Annotated[Tensor, "N D", float],
        bridge_cell_features: Annotated[Tensor, "M D", float],
        inter_neighborhood: Annotated[Tensor, "N M", bool | int, torch.sparse_coo],
    ) -> Annotated[Tensor, "N D", float]:
        # For each pair of cells, we need to compute the average of the features of the
        # bridge cells to which they are both bound. For example, given two nodes we
        # need to compute the average of the features of the hyperedges to which both
        # nodes belong.
        inter_neighborhood = inter_neighborhood.to_dense()
        # (N, N, M), [i, j, k] = 1 if cells i and j are both bound to bridge cell k
        shared_bridges = inter_neighborhood.unsqueeze(1).logical_and(inter_neighborhood)
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
        cell_features: Annotated[Tensor, "N D", float],
        bridge_cell_features: Annotated[Tensor, "M D", float],
        inter_neighborhood: Annotated[Tensor, "N M", bool | int, torch.sparse_coo],
    ) -> Annotated[Tensor, "N D", float]:
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
        source_features: Annotated[Tensor, "N D", float],
        target_features: Annotated[Tensor, "M D", float],
        inter_neighborhood: Annotated[Tensor, "M N", Number, torch.sparse_coo],
    ) -> Annotated[Tensor, "M D", float]:
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
        source_features: Annotated[Tensor, "N D", float],
        target_features: Annotated[Tensor, "M D", float],
        inter_neighborhood: Annotated[Tensor, "M N", Number, torch.sparse_coo],
    ) -> Annotated[Tensor, "M D", float]:
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
        cbm = ccc.coboundary_matrix(1)  # (E, N)
        nodes = ccc.cell_features(0)  # (N, D)
        edges = ccc.cell_features(1)  # (E, D)

        node_to_edge = self.node_to_edge_attn(nodes, edges, cbm)
        node_to_node = self.node_to_node_attn(nodes, edges, bm)

        edge_to_node = self.edge_to_node_attn(edges, nodes, bm)
        edge_to_edge = self.edge_to_edge_attn(edges, nodes, cbm)

        nodes = node_to_edge + node_to_node
        edges = edge_to_node + edge_to_edge

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
        similarity_threshold: float = 0.0,
        use_edge_to_node_attn: bool = True,
    ) -> None:
        super().__init__()

        if not -1 <= similarity_threshold <= 1:
            raise ValueError(
                f"similarity_threshold ({similarity_threshold}) must be in the "
                f"range [-1, 1]."
            )

        self.node_to_node_attn = IntraRankAttention(
            embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            bias=bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )

        if use_edge_to_node_attn:
            self.edge_to_node_attn = InterRankAttention(
                embed_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                bias=bias,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
            )
        else:
            self.edge_to_node_attn = None

        self.similarity_threshold = similarity_threshold

    def forward(
        self, ccc: CombinatorialComplex
    ) -> tuple[CombinatorialComplex, list[HCStep]]:
        node_to_node = self.node_to_node_attn(
            cell_features=ccc.cell_features(0),
            bridge_cell_features=ccc.cell_features(1),
            inter_neighborhood=ccc.boundary_matrix(1),
        )

        if self.edge_to_node_attn is not None:
            edge_to_node = self.edge_to_node_attn(
                source_features=ccc.cell_features(1),
                target_features=ccc.cell_features(0),
                inter_neighborhood=ccc.coboundary_matrix(1),
            )
            nodes = node_to_node + edge_to_node  # (N, D)
        else:
            nodes = node_to_node  # (N, D)

        num_nodes = tuple(ccc.num_cells(0, BatchMode.SEQUENCE))
        coboundary_matrix, hc_output = self._create_coboundary_matrix(nodes, num_nodes)

        # The features of the new hyperedges are the average of the features of the
        # nodes that belong to them. As node features, we do not use the output of the
        # structure learning layer, but the original features of the nodes.
        nodes = ccc.cell_features(0)  # (N, D)
        edge_degree = coboundary_matrix.sum(-1, keepdim=True)  # (M, 1)
        edges = torch.mm(coboundary_matrix.to(nodes.dtype), nodes)  # (M, D)
        edges = edges / edge_degree

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
        self, nodes: Annotated[Tensor, "N D", float], num_cells: Iterable[int]
    ) -> tuple[Annotated[Tensor, "M N", bool], list[HCStep]]:
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
        mask_int = mask.int()

        similarity = nodes.mm(nodes.T)
        # We set the diagonal and the lower triangular part of the similarity matrix to
        # False to avoid merging a cluster with itself or merging two clusters more than
        # once.
        mask_indices = torch.tril_indices(*mask.shape)
        mask[mask_indices[0], mask_indices[1]] = True
        merge = (similarity > self.similarity_threshold).masked_fill_(mask, False)
        merge_indices = merge.nonzero(as_tuple=False)

        coboundary_matrix = torch.zeros(
            (len(merge_indices), nodes.shape[0]), dtype=torch.int, device=nodes.device
        )
        cluster_indices = torch.arange(len(merge_indices), device=nodes.device)
        coboundary_matrix[cluster_indices, merge_indices[:, 0]] = 1
        coboundary_matrix[cluster_indices, merge_indices[:, 1]] = 1

        if len(merge_indices) == 0:
            step = HCStep(similarity, mask, None)
            hc_output.append(step)
            return coboundary_matrix, hc_output

        step = HCStep(similarity, mask, coboundary_matrix)
        hc_output.append(step)

        while True:
            clusters = coboundary_matrix.to(nodes.dtype).mm(nodes)  # (M, D)
            clusters_dim = coboundary_matrix.sum(-1, keepdim=True)  # (M, 1)
            clusters = clusters / clusters_dim

            clusters_mask = (
                coboundary_matrix.mm(mask_int).mm(coboundary_matrix.T).bool()
            )  # (M, M)
            mask_indices = torch.tril_indices(*clusters_mask.shape)
            clusters_mask[mask_indices[0], mask_indices[1]] = True

            similarity = clusters.mm(clusters.T)
            merge = similarity > self.similarity_threshold
            merge = merge.masked_fill_(clusters_mask, False)
            merge_indices = merge.nonzero(as_tuple=False)

            if len(merge_indices) == 0:
                step = HCStep(similarity, clusters_mask, None)
                hc_output.append(step)
                return coboundary_matrix, hc_output

            cbm = torch.zeros(
                (len(merge_indices), coboundary_matrix.shape[0]),
                dtype=torch.int,
                device=nodes.device,
            )  # (M', M)
            cluster_indices = torch.arange(len(merge_indices), device=nodes.device)
            cbm[cluster_indices, merge_indices[:, 0]] = 1
            cbm[cluster_indices, merge_indices[:, 1]] = 1

            cbm = cbm.mm(coboundary_matrix)  # (M', N)
            cbm.clamp_(max=1)

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
            coboundary_matrix: Tensor = cbm.unique(sorted=True, dim=0)

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

            step = HCStep(similarity, clusters_mask, coboundary_matrix)
            hc_output.append(step)

    def __call__(
        self, ccc: CombinatorialComplex
    ) -> tuple[CombinatorialComplex, list[HCStep]]:
        return super().__call__(ccc)


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

        self.nq_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.eq_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.kv_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, cpb_hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(cpb_hidden_dim, num_heads, bias=False),
        )

        self.attn_dropout = nn.Dropout(attn_dropout)

        self.nout_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.eout_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_dropout = nn.Dropout(proj_dropout)

    def forward(
        self,
        ccc: CombinatorialComplex,
        images: Annotated[Tensor, "B L D"],
        relative_distances: Annotated[Tensor, "B N L 2"],
    ) -> CombinatorialComplex:
        B, N, L, _ = images.shape  # noqa

        node_cpb = self.cpb_mlp(relative_distances)  # (B, N, L, H)
        node_cpb = node_cpb.permute(0, 3, 1, 2)  # (B, H, N, L)
        cbm = ccc.coboundary_matrix(1, BatchMode.STACK)  # (B, E, N)
        edge_cpb = torch.bmm(cbm[:, None].to(node_cpb.dtype), node_cpb)  # (B, H, E, L)

        nq = self.nq_proj(ccc.cell_features(0, BatchMode.STACK))
        nq = nq.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        eq = self.eq_proj(ccc.cell_features(1, BatchMode.STACK))
        eq = eq.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        kv = self.kv_proj(images)  # (B, L, D)
        kv = kv.view(B, L, self.num_heads, 2 * self.head_dim)
        kv = kv.transpose(1, 2)  # (B, H, L, 2D)
        k, v = kv.chunk(2, dim=-1)  # (B, H, L, D)

        node_attn_logits = (nq @ k.transpose(-1, -2)) * self.norm_factor
        node_attn_logits = node_attn_logits + node_cpb
        node_attn_scores = node_attn_logits.softmax(dim=-1)
        node_attn_scores = self.attn_dropout(node_attn_scores)

        edge_attn_logits = (eq @ k.transpose(-1, -2)) * self.norm_factor
        edge_attn_logits = edge_attn_logits + edge_cpb
        edge_attn_scores = edge_attn_logits.softmax(dim=-1)
        edge_attn_scores = self.attn_dropout(edge_attn_scores)

        node_out = (node_attn_scores @ v).transpose(1, 2)  # (B, N, H, Dh)
        node_out = node_out.flatten(2)  # (B, N, D)
        node_out = self.nout_proj(node_out)
        node_out = self.out_dropout(node_out)

        edge_out = (edge_attn_scores @ v).transpose(1, 2)  # (B, E, H, Dh)
        edge_out = edge_out.flatten(2)
        edge_out = self.eout_proj(edge_out)
        edge_out = self.out_dropout(edge_out)

        return ccc.replace((node_out, 0), (edge_out, 1))

    def __call__(
        self,
        ccc: CombinatorialComplex,
        images: Annotated[Tensor, "B L D"],
        relative_distances: Annotated[Tensor, "B N L 2"],
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
        use_edge_to_node_attn: bool = True,
    ) -> None:
        super().__init__()

        self.node_layernorm1 = nn.LayerNorm(embed_dim)
        self.edge_layernorm1 = nn.LayerNorm(embed_dim)
        self.structure_learning = HyperGraphStructureLearning(
            embed_dim,
            num_heads=num_heads,
            bias=bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            similarity_threshold=similarity_threshold,
            use_edge_to_node_attn=use_edge_to_node_attn,
        )
        self.hgat = HyperGraphAttention(
            embed_dim,
            num_heads=num_heads,
            bias=bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )

        self.node_layernorm2 = nn.LayerNorm(embed_dim)
        self.edge_layernorm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = CrossAttention(
            embed_dim,
            cpb_hidden_dim=cpb_hidden_dim,
            bias=bias,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )

        self.node_layernorm3 = nn.LayerNorm(embed_dim)
        self.edge_layernorm3 = nn.LayerNorm(embed_dim)
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
        images: Annotated[Tensor, "B L D"],
        relative_distances: Annotated[Tensor, "B N L 2"],
    ) -> tuple[CombinatorialComplex, list[HCStep]]:
        # Structure learning and HGAT
        nodes, edges = ccc.cell_features(0), ccc.cell_features(1)
        nodes_norm = self.node_layernorm1(nodes)
        edges_norm = self.edge_layernorm1(edges)
        ccc = ccc.replace((nodes_norm, 0), (edges_norm, 1))
        ccc, hc_output = self.structure_learning(ccc)
        ccc = self.hgat(ccc)
        nodes = ccc.cell_features(0) + nodes
        edges = ccc.cell_features(1) + edges

        # Cross-attention
        nodes_norm = self.node_layernorm2(nodes)
        edges_norm = self.edge_layernorm2(edges)
        ccc = ccc.replace((nodes_norm, 0), (edges_norm, 1))
        ccc = self.cross_attn(ccc, images, relative_distances)
        nodes = ccc.cell_features(0) + nodes
        edges = ccc.cell_features(1) + edges

        # FFN
        nodes_norm = self.node_layernorm3(nodes)
        edges_norm = self.edge_layernorm3(edges)
        nodes = self.node_ffn(nodes_norm) + nodes
        edges = self.edge_ffn(edges_norm) + edges

        ccc = ccc.replace((nodes, 0), (edges, 1))
        return ccc, hc_output

    def __call__(
        self,
        ccc: CombinatorialComplex,
        images: Annotated[Tensor, "B L D"],
        relative_distances: Annotated[Tensor, "B N L 2"],
    ) -> tuple[CombinatorialComplex, list[HCStep]]:
        return super().__call__(ccc, images, relative_distances)


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
                    use_edge_to_node_attn=idx > 0,
                )
                for idx, threshold in enumerate(similarity_thresholds)
            ]
        )

    def forward(
        self,
        ccc: CombinatorialComplex,
        boxes: BatchedBoundingBoxes,
        images: Annotated[Tensor, "B C H W"],
    ) -> list[tuple[CombinatorialComplex, list[HCStep]]]:
        relative_distances = _compute_relative_distances(boxes, images)
        images = images.flatten(2).transpose(1, 2)  # (B, HW, C)

        outputs = []
        for layer in self.layers:
            ccc, hc_output = layer(ccc, images, relative_distances)
            outputs.append((ccc, hc_output))

        return outputs

    def __call__(
        self,
        ccc: CombinatorialComplex,
        boxes: BatchedBoundingBoxes,
        images: Annotated[Tensor, "B C H W"],
    ) -> list[tuple[CombinatorialComplex, list[HCStep]]]:
        return super().__call__(ccc, boxes, images)


# --------------------------------------------------------------------------- #
# Private helper functions
# --------------------------------------------------------------------------- #


def _compute_relative_distances(
    boxes: BatchedBoundingBoxes,  # (B, Q, 4)
    images: Annotated[Tensor, "B C H W"],
) -> Annotated[Tensor, "B Q HW 2"]:
    H, W = images.shape[-2:]  # noqa
    image_coords = torch.cartesian_prod(
        torch.arange(W, device=images.device), torch.arange(H, device=images.device)
    )  # (K, 2)
    image_coords = image_coords[None, None]  # (1, 1, HW, 2)

    box_coords = boxes.denormalize().to_cxcywh().coordinates[..., :2]  # (B, Q, 2)
    box_coords = box_coords[:, :, None]  # (B, Q, 1, 2)

    distances = image_coords - box_coords  # (B, Q, HW, 2)
    distances = torch.sign(distances) * torch.log(1 + torch.abs(distances))

    return distances
