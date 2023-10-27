##
##
##

from typing import Annotated

import torch
import torch.sparse
from torch import Tensor, nn

from deepsight.structures import BatchedBoundingBoxes, BatchMode, Graph
from deepsight.utils.geometric import scatter_softmax, scatter_sum


class GraphAttention(nn.Module):
    """A Graph Attention Layer for graph-based DETR."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int | None = None,
        hidden_dim: int | None = None,
        share_weights: bool = False,
        bias: bool = True,
        num_heads: int = 8,
        negative_slope: float = 0.2,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        """Initialize a graph attention layer.

        Args:
            node_dim: The dimension of the node features.
            edge_dim: If edge features should be used to compute the attention
                scores and the messages, the dimension of the edge features.
            hidden_dim: The dimension of the hidden layer in the MLP used to compute
                the attention scores. If `None`, the hidden dimension is set to
                `node_dim`.
            share_weights: Whether to use the same weights for both the source and
                target nodes in the MLP used to compute the attention scores.
                Setting this to `True` makes the attention scores symmetric.
            bias: Whether to use a bias term in the linear layers.
            num_heads: The number of attention heads.
            negative_slope: The negative slope of the leaky ReLU activation.
            attn_dropout: The dropout probability applied to the attention scores.
            proj_dropout: The dropout probability applied to the output of the
                attention layer.
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = node_dim

        if node_dim % num_heads != 0:
            raise ValueError(
                f"node_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."  # noqa
            )

        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.hidden_dim = hidden_dim

        self.ni_proj = nn.Linear(node_dim, hidden_dim * num_heads, bias=bias)
        if share_weights:
            self.nj_proj = self.ni_proj
        else:
            self.nj_proj = nn.Linear(node_dim, hidden_dim * num_heads, bias=bias)

        if edge_dim is None:
            self.e_proj = None
        else:
            self.e_proj = nn.Linear(edge_dim, hidden_dim * num_heads, bias=bias)

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.attn_proj = nn.Parameter(torch.randn(self.num_heads, self.hidden_dim))
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.message_proj = nn.Linear(
            node_dim + (edge_dim if edge_dim is not None else 0), node_dim, bias=bias
        )

        self.out_proj = nn.Linear(node_dim, node_dim, bias=bias)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, graphs: Graph) -> Graph:
        ni = graphs.node_features()[graphs.adjacency_matrix().indices()[0]]
        nj = graphs.node_features()[graphs.adjacency_matrix().indices()[1]]

        ni_hidden = self.ni_proj(ni)
        nj_hidden = self.nj_proj(nj)

        if self.e_proj is not None:
            if graphs.edge_features() is None:
                raise ValueError("edge features must be provided.")
            e_hidden = self.e_proj(graphs.edge_features())
            hidden = ni_hidden + nj_hidden + e_hidden
        else:
            hidden = ni_hidden + nj_hidden

        hidden = self.leaky_relu(hidden)
        hidden = hidden.view(-1, self.num_heads, self.hidden_dim)
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

        messages = scatter_sum(messages, graphs.adjacency_matrix().indices()[0], dim=0)
        messages = messages.view(-1, self.num_heads * self.head_dim)

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

    def __init__(
        self,
        embed_dim: int,
        cpb_hidden_dim: int,
        bias: bool = True,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        """Initialize a cross-attention layer.

        Args:
            embed_dim: The dimension of the inputs used to compute the queries,
                keys, and values. This is also the dimension of the outputs.
            cpb_hidden_dim: The dimension of the hidden layer used to compute the
                continuous position bias.
            bias: Whether to use a bias term in the linear layers.
            num_heads: The number of attention heads.
            attn_dropout: The dropout probability applied to the attention scores.
            proj_dropout: The dropout probability applied to the output of the
                attention layer.
        """
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.norm_factor = self.head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2, bias=bias)
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, cpb_hidden_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(cpb_hidden_dim, num_heads, bias=bias),
        )

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(
        self,
        graph: Graph,
        images: Annotated[Tensor, "B K D"],
        relative_distances: Annotated[Tensor, "B Q K 2"],
    ) -> Graph:
        B, Q, K, _ = relative_distances.shape  # noqa

        nodes = graph.node_features(BatchMode.STACK)  # (B, Q, D)

        q = self.q_proj(nodes)
        q = q.view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)

        kv = self.kv_proj(images)
        kv = kv.view(B, K, self.num_heads, 2 * self.head_dim)
        kv = kv.transpose(1, 2)  # (B, H, K, 2D)
        k, v = kv.chunk(2, dim=-1)  # (B, H, K, D)

        # compute attention scores
        attn_logits = torch.matmul(q, k.transpose(-2, -1))  # (B, H, Q, K)
        attn_logits = attn_logits * self.norm_factor

        # Compute relative continuous position bias
        cpb = self.cpb_mlp(relative_distances)
        cpb = cpb.permute(0, 3, 1, 2)  # (B, H, Q, K)

        attn_logits = attn_logits + cpb
        attn_scores = torch.softmax(attn_logits, dim=-1)
        attn_scores = self.attn_dropout(attn_scores)

        # compute output
        out = torch.matmul(attn_scores, v)  # (B, H, Q, D)
        out = out.transpose(1, 2).contiguous()  # (B, Q, H, D)
        out = out.view(B, Q, self.num_heads * self.head_dim)
        out = self.out_proj(out)
        out = self.proj_dropout(out)

        return graph.replace(node_features=out)

    def __call__(
        self,
        graph: Graph,
        images: Annotated[Tensor, "B K D"],
        relative_distances: Annotated[Tensor, "B Q K 2"],
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
    def __init__(
        self,
        node_dim: int,
        edge_dim: int | None = None,
        cpb_hidden_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        """Initialize a decoder layer."""
        super().__init__()

        self.layernorm1 = nn.LayerNorm(node_dim)
        self.gat = GraphAttention(
            node_dim,
            edge_dim,
            hidden_dim=node_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )

        self.layernorm2 = nn.LayerNorm(node_dim)
        self.cross_attn = CrossAttention(
            node_dim,
            cpb_hidden_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )

        self.layernorm3 = nn.LayerNorm(node_dim)
        self.ffn = nn.Sequential(
            nn.Linear(node_dim, node_dim * 4),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(node_dim * 4, node_dim),
            nn.Dropout(proj_dropout),
        )

    def forward(
        self,
        graph: Graph,
        images: Annotated[Tensor, "B K D", float],
        relative_coords: Annotated[Tensor, "B Q K 2", float],
    ) -> Graph:
        nodes = graph.node_features()
        nodes_norm = self.layernorm1(nodes)
        graph = graph.replace(node_features=nodes_norm)
        graph = self.gat(graph)
        nodes = nodes + graph.node_features()

        nodes_norm = self.layernorm2(nodes)
        graph = graph.replace(node_features=nodes_norm)
        graph = self.cross_attn(graph, images, relative_coords)
        nodes = nodes + graph.node_features()

        nodes_norm = self.layernorm3(nodes)
        nodes = self.ffn(nodes_norm)
        nodes = nodes + nodes_norm

        return graph.replace(node_features=nodes)

    def __call__(
        self,
        graph: Graph,
        images: Annotated[Tensor, "B K D", float],
        relative_coords: Annotated[Tensor, "B Q K 2", float],
    ) -> Graph:
        """Update the node features by performing GAT, cross-attention, and FFN."""
        return super().__call__(graph, images, relative_coords)


class Decoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int | None = None,
        cpb_hidden_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        num_layers: int = 6,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    node_dim,
                    edge_dim,
                    cpb_hidden_dim,
                    num_heads,
                    attn_dropout,
                    proj_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        graphs: Graph,
        boxes: BatchedBoundingBoxes,
        images: Annotated[Tensor, "B C H W"],
    ) -> Graph:
        relative_distances = _compute_relative_distances(boxes, images)
        flattened_images = images.flatten(start_dim=2).transpose(1, 2)  # (B, K, D)

        for layer in self.layers:
            graphs = layer(graphs, flattened_images, relative_distances)

        return graphs

    def __call__(
        self,
        graphs: Graph,
        boxes: BatchedBoundingBoxes,
        images: Annotated[Tensor, "B C H W"],
    ) -> Graph:
        """Update the node features through the decoder layers."""
        return super().__call__(graphs, boxes, images)


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
