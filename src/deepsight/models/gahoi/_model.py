##
##
##

from dataclasses import dataclass
from typing import Annotated

import torch
import torch.nn.functional as F  # noqa
from torch import Tensor, nn
from torchvision.ops import RoIAlign

from deepsight.models import DeepSightModel
from deepsight.structures import (
    Batch,
    BatchedBoundingBoxes,
    BatchMode,
    BoundingBoxes,
    Graph,
)
from deepsight.tasks.hoic import Annotations, Predictions, Sample
from deepsight.typing import Configurable, JSONPrimitive

from ._decoder import Decoder
from ._encoder import ViTEncoder
from ._structures import Output


@dataclass(frozen=True)
class Config:
    """Configuration for the GAHOI model."""

    human_class_id: int
    num_interaction_classes: int
    allow_human_human: bool
    encoder: str = "vit_base_patch16_224"
    image_size: tuple[int, int] = (224, 224)
    num_decoder_layers: int = 6
    node_dim: int = 256
    edge_dim: int = 256
    cpb_hidden_dim: int = 256
    num_heads: int = 8
    attn_dropout: float = 0.1
    proj_dropout: float = 0.1


class GAHOI(DeepSightModel[Sample, Output, Annotations, Predictions], Configurable):
    """The Graph-based Approach to Human-Object Interaction (GAHOI) model."""

    # ----------------------------------------------------------------------- #
    # Constructor
    # ----------------------------------------------------------------------- #

    def __init__(self, config: Config) -> None:
        super().__init__()

        self._config = config
        self.human_class_id = config.human_class_id
        self.allow_human_human = config.allow_human_human

        # Parameters
        self.encoder = ViTEncoder(config.encoder, config.image_size, config.node_dim)
        self.roi_align = RoIAlign(1, 1.0, -1, True)
        self.edge_proj = nn.Linear(36, config.edge_dim)
        self.decoder = Decoder(
            node_dim=config.node_dim,
            edge_dim=config.edge_dim,
            cpb_hidden_dim=config.cpb_hidden_dim,
            num_layers=config.num_decoder_layers,
            num_heads=config.num_heads,
            attn_dropout=config.attn_dropout,
            proj_dropout=config.proj_dropout,
        )

        classifier_dim = 2 * config.node_dim + config.edge_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_dim),
            nn.Linear(
                in_features=classifier_dim,
                out_features=config.num_interaction_classes,
                bias=False,
            ),
        )

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def forward(
        self, samples: Batch[Sample], annotations: Batch[Annotations] | None
    ) -> Output:
        # Encoder
        images = self.encoder((sample.image for sample in samples))
        H, W = images.shape[-2:]  # noqa
        boxes = [s.entities.resize((H, W)).denormalize().to_xyxy() for s in samples]

        coords = [box.coordinates for box in boxes]
        node_features = self.roi_align(images, coords)
        node_features = node_features.flatten(1)  # (N, C)
        node_features = node_features.split_with_sizes([len(box) for box in boxes])

        # Decoder
        boxes = BatchedBoundingBoxes.batch(boxes)
        graphs = Graph.batch(
            (
                self._create_interaction_graph(sample, features)
                for sample, features in zip(samples, node_features, strict=True)
            )
        )
        edge_features = self.edge_proj(graphs.edge_features())
        graphs = graphs.replace(edge_features=edge_features)
        graphs = self.decoder(graphs, boxes, images)

        # Classifier
        edge_features = graphs.edge_features()
        assert edge_features is not None

        edge_indices = graphs.adjacency_matrix().indices()
        # keep only the edges where the first node is a human
        entity_labels = torch.cat([sample.entity_labels for sample in samples])
        keep = entity_labels[edge_indices[0]] == self.human_class_id
        edge_indices = edge_indices[:, keep]

        first_node = graphs.node_features()[edge_indices[0]]
        second_node = graphs.node_features()[edge_indices[1]]
        edge_features = edge_features[keep]

        interaction_features = torch.cat(
            [first_node, second_node, edge_features], dim=1
        )
        interaction_logits = self.classifier(interaction_features)

        # since we removed some edges, we need to update the number of edges
        keep = keep.split_with_sizes(list(graphs.num_edges(BatchMode.SEQUENCE)))
        num_edges = [int(torch.sum(k).item()) for k in keep]

        return Output(
            num_nodes=list(graphs.num_nodes(BatchMode.SEQUENCE)),
            num_edges=num_edges,
            interactions=edge_indices.transpose(0, 1),
            interaction_logits=interaction_logits,
        )

    def postprocess(self, output: Output) -> Batch[Predictions]:
        predictions = []
        node_offset = 0

        interactions = output.interactions.split_with_sizes(output.num_edges)
        interaction_logits = F.sigmoid(output.interaction_logits)
        interaction_labels = interaction_logits.split_with_sizes(output.num_edges)

        for idx, num_nodes in enumerate(output.num_nodes):
            predictions.append(
                Predictions(interactions[idx] - node_offset, interaction_labels[idx])
            )
            node_offset += num_nodes

        return Batch(predictions)

    def get_config(self) -> JSONPrimitive:
        return self._config.__dict__.copy()

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _create_interaction_graph(
        self, sample: Sample, node_features: Annotated[Tensor, "N D", float]
    ) -> Graph:
        human_labels = sample.entity_labels == self.human_class_id
        human_indices = torch.nonzero(human_labels, as_tuple=True)[0]
        object_indices = torch.nonzero(human_labels.logical_not_(), as_tuple=True)[0]

        human_object_edges = torch.cartesian_prod(human_indices, object_indices)
        object_human_edges = human_object_edges.flip(1)

        edges = [human_object_edges, object_human_edges]

        if self.allow_human_human:
            human_human_edges = torch.cartesian_prod(human_indices, human_indices)
            # remove self-loops
            keep = human_human_edges[:, 0] != human_human_edges[:, 1]
            human_human_edges = human_human_edges[keep]

            edges.append(human_human_edges)

        edges = torch.cat(edges, dim=0).transpose(0, 1)  # (2, E)
        adj_matrix = torch.sparse_coo_tensor(
            indices=edges,
            values=torch.ones(edges.shape[1], device=node_features.device),
            size=(node_features.shape[0], node_features.shape[0]),
            is_coalesced=True,
        )
        edge_features = self._get_edge_features(edges, sample.entities)  # (E, D)

        return Graph(adj_matrix, node_features, edge_features)

    def _get_edge_features(
        self, edges: Annotated[Tensor, "2 E", int], boxes: BoundingBoxes
    ) -> Annotated[Tensor, "E D", float]:
        eps = torch.finfo(boxes.coordinates.dtype).eps
        edge_features = []

        boxes = boxes.normalize().to_cxcywh()
        boxes1 = boxes[edges[0]]
        boxes2 = boxes[edges[1]]

        edge_features.extend([boxes1.coordinates[:, i] for i in range(4)])
        edge_features.extend([boxes2.coordinates[:, i] for i in range(4)])
        edge_features.append(boxes1.aspect_ratio())
        edge_features.append(boxes2.aspect_ratio())

        area1 = boxes1.area()
        area2 = boxes2.area()
        edge_features.append(area1)
        edge_features.append(area2)
        edge_features.append(area1 / (area2 + eps))

        edge_features.append(boxes1.iou(boxes2))

        dx = boxes1.coordinates[:, 0] - boxes2.coordinates[:, 0]
        dx = dx / (boxes1.coordinates[:, 2] + eps)

        dy = boxes1.coordinates[:, 1] - boxes2.coordinates[:, 1]
        dy = dy / (boxes1.coordinates[:, 3] + eps)

        edge_features.extend([F.relu(dx), F.relu(-dx), F.relu(dy), F.relu(-dy)])

        edge_features = torch.stack(edge_features, dim=1)
        edge_features = torch.cat(
            [edge_features, torch.log(edge_features + eps)], dim=1
        )

        return edge_features
