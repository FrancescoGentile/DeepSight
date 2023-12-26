##
##
##

from typing import Literal

import torch
import torch.nn.functional as F  # noqa
import torch.sparse
from torch import nn

from deepsight.ops.geometric import scatter_softmax, scatter_sum
from deepsight.structures.geometric import BatchMode, Graph
from deepsight.structures.vision import (
    BatchedBoundingBoxes,
    BatchedImages,
    BatchedSequences,
)
from deepsight.typing import Tensor

from ._config import Config


class GraphAttention(nn.Module):
    """A Graph Attention Layer for graph-based DETR."""

    def __init__(self, config: Config) -> None:
        """Initialize a graph attention layer."""
        super().__init__()

        self.num_heads = config.num_heads
        self.head_dim = config.node_dim // config.num_heads
        self.hidden_dim = config.node_dim

        self.ni_proj = nn.Linear(config.node_dim, self.hidden_dim * self.num_heads)
        self.nj_proj = nn.Linear(config.node_dim, self.hidden_dim * self.num_heads)
        self.e_proj = nn.Linear(config.edge_dim, self.hidden_dim * self.num_heads)

        self.leaky_relu = nn.LeakyReLU(config.negative_slope)
        self.hidden_dropout = nn.Dropout(config.qkv_dropout)

        self.attn_proj = nn.Parameter(torch.randn(self.num_heads, self.hidden_dim))
        self.attn_dropout = nn.Dropout(config.attn_dropout)

        self.message_proj = nn.Linear(
            config.node_dim + config.edge_dim, config.node_dim
        )
        self.message_dropout = nn.Dropout(config.qkv_dropout)

        self.out_proj = nn.Linear(config.node_dim, config.node_dim)
        self.proj_dropout = nn.Dropout(config.proj_dropout)

    def forward(self, graphs: Graph) -> Graph:
        ni = graphs.node_features()[graphs.adjacency_matrix().indices()[0]]
        nj = graphs.node_features()[graphs.adjacency_matrix().indices()[1]]

        ni_hidden = self.ni_proj(ni)
        nj_hidden = self.nj_proj(nj)

        if graphs.edge_features() is None:
            raise ValueError("edge features must be provided.")
        e_hidden = self.e_proj(graphs.edge_features())
        hidden = ni_hidden + nj_hidden + e_hidden

        hidden = self.leaky_relu(hidden)
        hidden = hidden.view(-1, self.num_heads, self.hidden_dim)
        hidden = self.hidden_dropout(hidden)
        attn_logits = (hidden * self.attn_proj).sum(dim=-1)  # (E, H)

        attn_scores = scatter_softmax(
            attn_logits, graphs.adjacency_matrix().indices()[0], dim=0
        )
        attn_scores = self.attn_dropout(attn_scores)

        edge_features = graphs.edge_features()
        if edge_features is not None:
            messages = torch.cat((nj, edge_features), dim=-1)
        else:
            messages = nj

        messages = self.message_proj(messages)
        messages = messages.view(-1, self.num_heads, self.head_dim)
        messages = messages * attn_scores.unsqueeze(-1)

        messages = scatter_sum(
            messages,
            graphs.adjacency_matrix().indices()[0],
            dim=0,
            dim_output_size=graphs.num_nodes(),
        )
        messages = messages.view(-1, self.num_heads * self.head_dim)
        messages = self.message_dropout(messages)

        out = self.out_proj(messages)
        out = self.proj_dropout(out)

        return graphs.replace(node_features=out)

    def __call__(self, graphs: Graph) -> Graph:
        """Update the node features by performing graph attention.

        Args:
            graphs: The graphs to update.

        Returns:
            The updated graphs.
        """
        return super().__call__(graphs)


class CrossAttention(nn.Module):
    """Cross-attention layer for graph-based DETR."""

    def __init__(self, config: Config) -> None:
        """Initialize a cross-attention layer."""
        super().__init__()

        self.num_heads = config.num_heads
        self.head_dim = config.node_dim // config.num_heads
        self.attn_dropout = config.attn_dropout

        self.q_proj = nn.Linear(config.node_dim, config.node_dim)
        self.kv_proj = nn.Linear(config.node_dim, config.node_dim * 2)
        self.qkv_dropout = nn.Dropout(config.qkv_dropout)

        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, config.cpb_hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(config.cpb_hidden_dim, config.num_heads, bias=False),
        )

        self.out_proj = nn.Linear(config.node_dim, config.node_dim)
        self.proj_dropout = nn.Dropout(config.proj_dropout)

    def forward(
        self,
        graph: Graph,
        images: BatchedSequences,
        relative_distances: Tensor[Literal["B Q P 2"], float],
    ) -> Graph:
        B, Q, K, _ = relative_distances.shape  # noqa

        nodes = graph.node_features(BatchMode.STACK)  # (B, Q, D)
        q = self.q_proj(nodes)
        q = q.view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.qkv_dropout(q)

        kv = self.kv_proj(images.data)
        kv = kv.view(B, K, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, B, H, P, D)
        kv = self.qkv_dropout(kv)
        k, v = kv.unbind(dim=0)

        # Compute relative continuous position bias
        cpb = self.cpb_mlp(relative_distances)
        cpb = cpb.permute(0, 3, 1, 2)  # (B, H, Q, P)

        mask = images.mask[:, None, None]  # (B, 1, 1, P)
        att_mask = cpb.masked_fill_(mask, -torch.inf)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=att_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )

        out = out.reshape(B, Q, self.num_heads * self.head_dim)
        out = self.out_proj(out)
        out = self.proj_dropout(out)

        return graph.replace(node_features=out)

    def __call__(
        self,
        graph: Graph,
        images: BatchedSequences,
        relative_distances: Tensor[Literal["B Q P 2"], float],
    ) -> Graph:
        """Update the entities features by attending to the images.

        Args:
            graph: The graph containing the entities to update.
            images: The image features being attended to.
            relative_distances: The relative distances between the centers of the
                bounding boxes of the entities and the position of each patch in the
                images.

        Returns:
            The graph with the updated entities features.
        """
        return super().__call__(graph, images, relative_distances)


class DecoderLayer(nn.Module):
    def __init__(self, config: Config) -> None:
        """Initialize a decoder layer."""
        super().__init__()

        self.gat_layernorm = nn.LayerNorm(config.node_dim)
        self.gat = GraphAttention(config)

        self.ca_layernorm = nn.LayerNorm(config.node_dim)
        self.cross_attn = CrossAttention(config)

        self.ffn_layernorm = nn.LayerNorm(config.node_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.node_dim, config.node_dim * 4),
            nn.GELU(),
            nn.Dropout(config.ffn_dropout),
            nn.Linear(config.node_dim * 4, config.node_dim),
            nn.Dropout(config.ffn_dropout),
        )

    def forward(
        self,
        graph: Graph,
        images: BatchedSequences,
        relative_coords: Tensor[Literal["B Q P 2"], float],
    ) -> Graph:
        nodes = graph.node_features()
        nodes_norm = self.gat_layernorm(nodes)
        graph = graph.replace(node_features=nodes_norm)
        graph = self.gat(graph)
        nodes = nodes + graph.node_features()

        nodes_norm = self.ca_layernorm(nodes)
        graph = graph.replace(node_features=nodes_norm)
        graph = self.cross_attn(graph, images, relative_coords)
        nodes = nodes + graph.node_features()

        nodes_norm = self.ffn_layernorm(nodes)
        nodes = self.ffn(nodes_norm)
        nodes = nodes + nodes_norm

        return graph.replace(node_features=nodes)

    def __call__(
        self,
        graph: Graph,
        images: BatchedSequences,
        relative_coords: Tensor[Literal["B Q P 2"], float],
    ) -> Graph:
        """Update the node features by performing GAT, cross-attention, and FFN."""
        return super().__call__(graph, images, relative_coords)


class Decoder(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.num_decoder_layers)
        ])

    def forward(
        self,
        graphs: Graph,
        boxes: BatchedBoundingBoxes,
        images: BatchedImages,
    ) -> list[Graph]:
        relative_distances = _compute_relative_distances(boxes, images)
        patches = images.to_sequences()

        outputs = []
        for layer in self.layers:
            graphs = layer(graphs, patches, relative_distances)
            outputs.append(graphs)

        return outputs

    def __call__(
        self,
        graphs: Graph,
        boxes: BatchedBoundingBoxes,
        images: BatchedImages,
    ) -> list[Graph]:
        """Update the node features through the decoder layers."""
        return super().__call__(graphs, boxes, images)


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
