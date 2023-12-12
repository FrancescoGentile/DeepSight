##
##
##

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from deepsight.core import Batch
from deepsight.core import Model as _Model
from deepsight.layers.vision import ViTEncoder
from deepsight.ops.geometric import coalesce
from deepsight.ops.vision import RoIAlign
from deepsight.structures.geometric import BatchMode, CombinatorialComplex
from deepsight.structures.vision import (
    BatchedBoundingBoxes,
    BatchedImages,
    BoundingBoxes,
)
from deepsight.tasks.meic import Annotations, Predictions, Sample
from deepsight.typing import Configs, Configurable, Tensor

from ._decoder import Decoder
from ._structures import GTClusters, HCStep, LayerOutput, Output


@dataclass(frozen=True)
class Config:
    """The configuration of the TAHOI model."""

    human_class_id: int
    num_interaction_classes: int
    encoder_variant: ViTEncoder.Variant = ViTEncoder.Variant.BASE
    encoder_patch_size: Literal[16, 32] = 16
    encoder_image_size: Literal[224, 384] = 384
    embed_dim: int = 256
    cpb_hidden_dim: int = 256
    num_heads: int = 8
    attn_dropout: float = 0.1
    proj_dropout: float = 0.1
    similarity_thresholds: float | Iterable[float] = 0.5
    num_decoder_layers: int = 6


class Model(_Model[Sample, Output, Annotations, Predictions], Configurable):
    """The Topological Approach to Human Object Interaction (TAHOI) model."""

    # ----------------------------------------------------------------------- #
    # Constructor
    # ----------------------------------------------------------------------- #

    def __init__(self, config: Config) -> None:
        super().__init__()

        self._config = config
        self.human_class_id = config.human_class_id

        self.encoder = ViTEncoder.build(
            config.encoder_variant, config.encoder_patch_size, config.encoder_image_size
        )
        if self.encoder.output_channels != config.embed_dim:
            self.proj = nn.Conv2d(self.encoder.output_channels, config.embed_dim, 1)
        else:
            self.proj = nn.Identity()

        self.roi_align = RoIAlign(output_size=1, sampling_ratio=-1, aligned=True)

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

    def get_configs(self, recursive: bool) -> Configs:
        return self._config.__dict__.copy()

    def forward(
        self,
        samples: Batch[Sample],
        annotations: Batch[Annotations] | None,
    ) -> Output:
        images = BatchedImages.batch([sample.image.data for sample in samples])
        images = self.encoder(images)
        features = self.proj(images.data)

        images = images.replace(data=features)

        ccc, entity_boxes = self._create_decoder_input(samples, images)

        if annotations is not None:
            gt_clusters = GTClusters.from_annotations(annotations)
        else:
            gt_clusters = None

        layer_outputs = self.decoder(ccc, entity_boxes, images, gt_clusters)

        output = self._create_output(
            layer_outputs,
            [sample.entity_labels for sample in samples],
            gt_clusters,
        )

        return output

    def postprocess(self, output: Output) -> Batch[Predictions]:
        num_nodes = output.layers[-1].num_nodes
        num_hedges = output.layers[-1].num_hedges
        num_binary_edges = output.layers[-1].num_binary_interactions
        logits = output.layers[-1].interaction_logits
        binary_logits = output.layers[-1].binary_interaction_logits
        bbm = output.layers[-1].boundary_matrix.to_dense()
        binary_interactions = output.layers[-1].binary_interactions

        labels = torch.sigmoid(logits)
        binary_labels = labels[binary_interactions[2]]
        binary_labels = torch.sigmoid(binary_logits).unsqueeze(1) * binary_labels

        labels = labels.split_with_sizes(num_hedges)
        binary_labels = binary_labels.split_with_sizes(num_binary_edges)
        binary_interactions = binary_interactions.split_with_sizes(
            num_binary_edges, dim=1
        )

        predictions: list[Predictions] = []
        node_offset, hedge_offset = 0, 0
        for idx in range(len(num_nodes)):
            node_limit = node_offset + num_nodes[idx]
            edge_limit = hedge_offset + num_hedges[idx]

            bm = bbm[node_offset:node_limit, hedge_offset:edge_limit]

            bindices = binary_interactions[idx] - node_offset

            if bindices.shape[1] > 0:
                bindices, blabels = coalesce(bindices, binary_labels[idx], reduce="max")
            else:
                blabels = binary_labels[idx]

            node_offset = node_limit
            hedge_offset = edge_limit

            predictions.append(Predictions(bm, labels[idx], bindices, blabels))

        return Batch(predictions)

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _create_decoder_input(
        self,
        samples: Batch[Sample],
        images: BatchedImages,
    ) -> tuple[CombinatorialComplex, BatchedBoundingBoxes]:
        entity_boxes = [
            sample.entity_boxes.resize(size).denormalize().to_xyxy()
            for sample, size in zip(samples, images.image_sizes, strict=True)
        ]

        node_features = self.roi_align(images.data, entity_boxes)
        node_features = node_features.flatten(1)  # (N, C)
        node_features = node_features.split_with_sizes(
            [len(box) for box in entity_boxes]
        )

        boundary_matrices: list[torch.Tensor] = []
        union_boxes: list[BoundingBoxes] = []
        for sample, boxes in zip(samples, entity_boxes, strict=True):
            bm, ub = self._create_interactions(sample.entity_labels, boxes)
            boundary_matrices.append(bm)
            union_boxes.append(ub.denormalize().to_xyxy())

        edge_features = self.roi_align(images.data, union_boxes)
        edge_features = edge_features.flatten(1)  # (E, C)
        edge_features = edge_features.split_with_sizes(
            [len(box) for box in union_boxes]
        )

        graphs = [
            CombinatorialComplex([nf, ef], [bm])
            for nf, ef, bm in zip(
                node_features, edge_features, boundary_matrices, strict=True
            )
        ]

        ccc = CombinatorialComplex.batch(graphs)
        bboxes = BatchedBoundingBoxes.batch(entity_boxes)

        return ccc, bboxes

    def _create_interactions(
        self,
        entity_labels: Tensor[Literal["N"], int],
        entity_boxes: BoundingBoxes,
    ) -> tuple[Tensor[Literal["N M"], bool], BoundingBoxes]:
        # create all unique pairs of entities
        entity_indices = torch.arange(len(entity_labels), device=entity_labels.device)
        edges = torch.combinations(entity_indices, r=2, with_replacement=False)
        edges = edges.transpose(0, 1)  # (2, E)

        # remove edges where no entity is human
        keep = entity_labels[edges[0]] == self.human_class_id
        keep = keep.logical_or_(entity_labels[edges[1]] == self.human_class_id)
        edges = edges[:, keep]

        # create boundary matrix
        boundary_matrix = torch.zeros(
            (len(entity_labels), edges.shape[1]),
            dtype=torch.bool,
            device=edges.device,
        )
        edge_indices = torch.arange(edges.shape[1], device=edges.device)
        boundary_matrix[edges[0], edge_indices] = True
        boundary_matrix[edges[1], edge_indices] = True

        union_boxes = entity_boxes[edges[0]].union(entity_boxes[edges[1]])

        return boundary_matrix, union_boxes

    def _create_output(
        self,
        layer_outputs: list[tuple[CombinatorialComplex, list[HCStep]]],
        entity_labels: list[Tensor[Literal["N"], int]],
        gt_clusters: GTClusters | None,
    ) -> Output:
        layers = []

        for ccc, hc_output in layer_outputs:
            num_nodes = list(ccc.num_cells(0, BatchMode.SEQUENCE))
            num_hedges = list(ccc.num_cells(1, BatchMode.SEQUENCE))

            bm = ccc.boundary_matrix(1)  # (N, H)
            interaction_logits = self.mei_classifier(ccc.cell_features(1))

            dense_bm = bm.to_dense()
            # [i, j, k] = 1 if nodes i and j are connected by hyperedge k
            binary_interactions = (
                dense_bm.unsqueeze(1)
                .expand(-1, dense_bm.shape[0], -1)
                .logical_and(dense_bm)
            )  # (N, N, H)

            binary_interactions_list = []
            offset = 0
            for idx in range(len(num_nodes)):
                limit = offset + num_nodes[idx]
                sample_binary = binary_interactions[offset:limit, offset:limit]
                sample_binary = sample_binary.nonzero(as_tuple=False)

                # remove binary interactions where the first entity is not human
                keep = entity_labels[idx][sample_binary[:, 0]] == self.human_class_id
                sample_binary = sample_binary[keep]

                # remove self interactions
                keep = sample_binary[:, 0] != sample_binary[:, 1]
                sample_binary = sample_binary[keep]

                sample_binary[:, :2] += offset
                binary_interactions_list.append(sample_binary)

                offset = limit

            num_binary_interactions = [len(edges) for edges in binary_interactions_list]
            binary_interactions = torch.cat(binary_interactions_list, dim=0)
            binary_interactions.transpose_(0, 1)  # (3, E)

            nodes1 = ccc.cell_features(0)[binary_interactions[0]]
            nodes2 = ccc.cell_features(0)[binary_interactions[1]]
            edge = ccc.cell_features(1)[binary_interactions[2]]
            binary = torch.cat([nodes1, nodes2, edge], dim=1)  # (E, 3C)
            binary_logits = self.binary_classifier(binary).squeeze(1)  # (E,)

            layers.append(
                LayerOutput(
                    steps=hc_output,
                    num_nodes=num_nodes,
                    num_hedges=num_hedges,
                    num_binary_interactions=num_binary_interactions,
                    boundary_matrix=bm,
                    interaction_logits=interaction_logits,
                    binary_interactions=binary_interactions,
                    binary_interaction_logits=binary_logits,
                )
            )

        return Output(layers, gt_clusters)
