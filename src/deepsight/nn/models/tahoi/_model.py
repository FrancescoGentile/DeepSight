##
##
##

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated

import torch
import torch.nn.functional as F  # noqa
from torch import Tensor, nn
from torchvision.ops import RoIAlign

from deepsight.nn.models import DeepSightModel
from deepsight.structures import (
    Batch,
    BatchedBoundingBoxes,
    BatchMode,
    BoundingBoxes,
    CombinatorialComplex,
)
from deepsight.tasks.meic import Annotations, Predictions, Sample
from deepsight.typing import Configurable, JSONPrimitive

from ._decoder import Decoder
from ._encoder import ViTEncoder
from ._structures import LayerOutput, Output


@dataclass(frozen=True)
class Config:
    """The configuration of the TAHOI model."""

    human_class_id: int
    num_interaction_classes: int
    encoder: str = "vit_base_patch16_224"
    image_size: tuple[int, int] = (224, 224)
    embed_dim: int = 256
    cpb_hidden_dim: int = 256
    num_heads: int = 8
    attn_dropout: float = 0.1
    proj_dropout: float = 0.1
    similarity_thresholds: float | Iterable[float] = 0.4
    num_decoder_layers: int = 6


class TAHOI(DeepSightModel[Sample, Output, Annotations, Predictions], Configurable):
    """The Topological Approach to Human Object Interaction (TAHOI) model."""

    # ----------------------------------------------------------------------- #
    # Constructor
    # ----------------------------------------------------------------------- #

    def __init__(self, config: Config) -> None:
        super().__init__()

        self._config = config
        self.human_class_id = config.human_class_id

        self.encoder = ViTEncoder(config.encoder, config.image_size, config.embed_dim)
        self.roi_align = RoIAlign(
            output_size=1, spatial_scale=1.0, sampling_ratio=-1, aligned=True
        )
        self.edge_proj = nn.Linear(36, config.embed_dim)

        self.decoder = Decoder(
            embed_dim=config.embed_dim,
            cpb_hidden_dim=config.cpb_hidden_dim,
            num_heads=config.num_heads,
            attn_dropout=config.attn_dropout,
            proj_dropout=config.proj_dropout,
            similarity_thresholds=config.similarity_thresholds,
            num_layers=config.num_decoder_layers,
        )

        self.mei_classifier = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.num_interaction_classes, bias=False),
        )

        self.binary_classifier = nn.Sequential(
            nn.LayerNorm(3 * config.embed_dim),
            nn.Linear(3 * config.embed_dim, 1, bias=False),
        )

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def forward(
        self, samples: Batch[Sample], annotations: Batch[Annotations] | None
    ) -> Output:
        images = self.encoder((sample.image for sample in samples))
        H, W = images.shape[-2:]  # noqa
        boxes = [
            sample.entities.resize((H, W)).denormalize().to_xyxy() for sample in samples
        ]

        coords = [box.coordinates for box in boxes]
        node_features = self.roi_align(images, coords)
        node_features = node_features.flatten(1)  # (N, C)
        node_features = node_features.split_with_sizes([len(box) for box in boxes])

        interaction_graphs = [
            self._create_interaction_graph(sample, features)
            for sample, features in zip(samples, node_features, strict=True)
        ]
        ccc = CombinatorialComplex.batch(interaction_graphs)
        edges = self.edge_proj(ccc.cell_features(1))
        ccc = ccc.replace((edges, 1))

        bboxes = BatchedBoundingBoxes.batch(boxes)

        layer_outputs = self.decoder(ccc, bboxes, images)

        entity_labels = torch.cat([sample.entity_labels for sample in samples])
        output = []

        for ccc, hc_output in layer_outputs:
            bm = ccc.boundary_matrix(1)  # (N, H)
            interaction_logits = self.mei_classifier(ccc.cell_features(1))

            binary_interactions = bm.unsqueeze(1).logical_and(bm)  # (N, N, H)
            binary_interactions = binary_interactions.nonzero(as_tuple=False)

            # remove binary interactions where the first entity is not human
            mask = entity_labels[binary_interactions[:, 0]] == self.human_class_id
            binary_interactions = binary_interactions[mask]

            # remove self interactions
            mask = binary_interactions[:, 0] != binary_interactions[:, 1]
            binary_interactions = binary_interactions[mask]

            nodes1 = ccc.cell_features(0)[binary_interactions[:, 0]]
            nodes2 = ccc.cell_features(0)[binary_interactions[:, 1]]
            edge = ccc.cell_features(1)[binary_interactions[:, 2]]
            binary = torch.cat([nodes1, nodes2, edge], dim=1)  # (E, 3C)
            binary_logits = self.binary_classifier(binary).squeeze(1)  # (E,)

            output.append(
                LayerOutput(
                    steps=hc_output,
                    num_nodes=list(ccc.num_cells(0, BatchMode.SEQUENCE)),
                    num_edges=list(ccc.num_cells(1, BatchMode.SEQUENCE)),
                    boundary_matrix=bm,
                    interaction_logits=interaction_logits,
                    binary_interactions=binary_interactions,
                    binary_interaction_logits=binary_logits,
                )
            )

        return output

    def postprocess(self, output: Output) -> Batch[Predictions]: ...

    def get_config(self) -> JSONPrimitive:
        return self._config.__dict__.copy()

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _create_interaction_graph(
        self, sample: Sample, node_features: Annotated[Tensor, "N D", float]
    ) -> CombinatorialComplex:
        human_indices = sample.entity_labels == self.human_class_id
        human_indices = torch.nonzero(human_indices, as_tuple=True)[0]
        target_indices = torch.arange(
            len(sample.entity_labels), device=human_indices.device
        )

        human_target_edges = torch.cartesian_prod(human_indices, target_indices)
        target_human_edges = human_target_edges.flip(1)
        edges = torch.cat([human_target_edges, target_human_edges], dim=0)
        edges = edges.transpose(0, 1)  # (2, E)

        # remove self-loops
        keep = edges[0] != edges[1]
        edges = edges[:, keep]

        edge_features = self._get_edge_features(edges, sample.entities)

        # create boundary matrix
        boundary_matrix = torch.zeros(
            (len(sample.entities), edges.shape[1]),
            dtype=torch.bool,
            device=edges.device,
        )
        edge_indices = torch.arange(edges.shape[1], device=edges.device)
        boundary_matrix[edges[0], edge_indices] = True
        boundary_matrix[edges[1], edge_indices] = True

        return CombinatorialComplex([node_features, edge_features], [boundary_matrix])

    def _get_edge_features(
        self, edges: Annotated[Tensor, "2 E", int], boxes: BoundingBoxes
    ) -> Annotated[Tensor, "E 36", float]:
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

        edge_features.append(boxes1.iou(boxes2))
        edge_features.append(area1 / area2)

        dx = boxes1.coordinates[:, 0] - boxes2.coordinates[:, 0]
        dx = dx / boxes1.coordinates[:, 2]

        dy = boxes1.coordinates[:, 1] - boxes2.coordinates[:, 1]
        dy = dy / boxes1.coordinates[:, 3]

        edge_features.extend([F.relu(dx), F.relu(-dx), F.relu(dy), F.relu(-dy)])

        edge_features = torch.stack(edge_features, dim=1)
        eps = torch.finfo(edge_features.dtype).eps
        edge_features = torch.cat(
            [edge_features, torch.log(edge_features + eps)], dim=1
        )

        return edge_features
