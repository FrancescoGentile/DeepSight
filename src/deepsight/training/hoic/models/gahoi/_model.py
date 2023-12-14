##
##
##

from dataclasses import replace
from typing import Literal

import torch
import torch.nn.functional as F  # noqa
from torch import nn

from deepsight.nn.vision import vit
from deepsight.ops.vision import RoIAlign
from deepsight.structures.geometric import BatchMode, Graph
from deepsight.structures.vision import (
    BatchedBoundingBoxes,
    BatchedImages,
    BoundingBoxes,
)
from deepsight.training import Batch
from deepsight.training import Model as _Model
from deepsight.training.hoic import Annotations, Predictions, Sample
from deepsight.typing import Configs, Configurable, Tensor

from ._config import Configs as ModelConfigs
from ._decoder import Decoder
from ._structures import LayerOutput, Output
from ._utils import get_interaction_mask


class Model(_Model[Sample, Output, Annotations, Predictions], Configurable):
    """The Graph-based Approach to Human-Object Interaction (GAHOI) model."""

    # ----------------------------------------------------------------------- #
    # Constructor
    # ----------------------------------------------------------------------- #

    def __init__(
        self,
        configs: ModelConfigs,
        obj_to_interactions: list[list[int]] | None = None,
    ) -> None:
        """Initialize the GAHOI model.

        Args:
            configs: The configurations for the model.
            obj_to_interactions: The lists of interactions for each object class.
                The number of lists must be equal to the number of entity classes.
                Each list contains the indices of the interactions in which the
                corresponding entity can be an object. For example, if
                `obj_to_interactions[0] = [0, 1]`, then the first object class
                can participate in the first and second interactions. If provided,
                this will be used to filter out interactions that an object class
                cannot participate in (i.e., the probability of the non-valid
                interactions will be set to zero). If not provided, all interactions
                will be considered valid for all object classes.
        """
        super().__init__()

        self._config = configs
        self.human_class_id = configs.human_class_id
        self.allow_human_human = configs.allow_human_human

        # Parameters
        vit_configs = vit.Configs.from_variant(
            configs.encoder_variant,
            patch_size=configs.encoder_patch_size,
            image_size=configs.encoder_image_size,
        )
        vit_configs = replace(
            vit_configs,
            qkv_dropout=configs.qkv_dropout,
            attn_dropout=configs.attn_dropout,
            proj_dropout=configs.proj_dropout,
            ffn_dropout=configs.ffn_dropout,
        )
        self.encoder = vit.Encoder(vit_configs)

        if self.encoder.output_channels != configs.node_dim:
            self.proj = nn.Conv2d(self.encoder.output_channels, configs.node_dim, 1)
        else:
            self.proj = nn.Identity()

        self.roi_align = RoIAlign(output_size=1, sampling_ratio=-1, aligned=True)
        self.edge_proj = nn.Linear(36, configs.edge_dim)
        self.decoder = Decoder(configs)

        classifier_dim = 2 * configs.node_dim + configs.edge_dim
        self.suppress_classifier = nn.Sequential(
            nn.LayerNorm(classifier_dim),
            nn.Linear(
                in_features=classifier_dim,
                out_features=1,
                bias=False,
            ),
        )

        self.interaction_classifier = nn.Sequential(
            nn.LayerNorm(classifier_dim),
            nn.Linear(
                in_features=classifier_dim,
                out_features=configs.num_interaction_classes,
                bias=False,
            ),
        )

        # Buffers
        self.register_buffer(
            "oi_mask",
            get_interaction_mask(
                obj_to_interactions,
                configs.num_entity_classes,
                configs.num_interaction_classes,
            ),
        )
        self.oi_mask: torch.Tensor | None  # for type checking

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
        # Encoder
        images = BatchedImages.batch([sample.image.data for sample in samples])
        images = self.encoder(images)
        features = self.proj(images.data)
        images = images.replace(data=features)

        entity_boxes = [
            sample.entity_boxes.resize(size).denormalize().to_xyxy()
            for sample, size in zip(samples, images.image_sizes, strict=True)
        ]
        node_features = self.roi_align(images, entity_boxes)
        node_features = node_features.flatten(1)  # (N, C)
        node_features = node_features.split_with_sizes([
            len(box) for box in entity_boxes
        ])

        # Decoder
        boxes = BatchedBoundingBoxes.batch(entity_boxes)
        batched_graphs = Graph.batch([
            self._create_interaction_graph(sample, features)
            for sample, features in zip(samples, node_features, strict=True)
        ])
        edge_features = self.edge_proj(batched_graphs.edge_features())
        batched_graphs = batched_graphs.replace(edge_features=edge_features)
        decoder_output = self.decoder(batched_graphs, boxes, images)

        entity_labels = torch.cat([sample.entity_labels for sample in samples])
        return self._create_output(decoder_output, entity_labels)

    def postprocess(self, output: Output) -> Batch[Predictions]:
        last_layer = output.layers[-1]

        predictions = []
        node_offset = 0

        indices = output.indices.split_with_sizes(output.num_edges)

        suppress = torch.sigmoid(last_layer.suppress_logits)
        labels = torch.sigmoid(last_layer.label_logits) * suppress
        if output.interaction_mask is not None:
            # set the probability of the non-valid interactions to zero
            labels.masked_fill_(output.interaction_mask, 0.0)

        labels = labels.split_with_sizes(output.num_edges)

        for idx, num_nodes in enumerate(output.num_nodes):
            predictions.append(Predictions(indices[idx] - node_offset, labels[idx]))
            node_offset += num_nodes

        return Batch(predictions)

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _create_interaction_graph(
        self,
        sample: Sample,
        node_features: Tensor[Literal["N D"], float],
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
        edge_features = self._get_edge_features(edges, sample.entity_boxes)  # (E, D)

        return Graph(adj_matrix, node_features, edge_features)

    def _get_edge_features(
        self, edges: Tensor[Literal["2 E"], int], boxes: BoundingBoxes
    ) -> Tensor[Literal["E D"], float]:
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

    def _create_output(
        self,
        decoder_output: list[Graph],
        entity_labels: Tensor[Literal["N C"], int],
    ) -> Output:
        graph = decoder_output[-1]
        edge_indices = graph.adjacency_matrix().indices()
        keep = entity_labels[edge_indices[0]] == self.human_class_id
        edge_indices = edge_indices[:, keep]
        edge_features = graph.edge_features()[keep]  # type: ignore

        num_nodes = list(graph.num_nodes(BatchMode.SEQUENCE))
        num_edges = [
            int(torch.sum(k).item())
            for k in keep.split_with_sizes(list(graph.num_edges(BatchMode.SEQUENCE)))
        ]

        # filter out interactions that an object class cannot participate in
        if self.oi_mask is not None:
            obj_classes = entity_labels[edge_indices[1]]
            mask = self.oi_mask[obj_classes]
        else:
            mask = None

        outputs = []
        for graph in decoder_output:
            first_node = graph.node_features()[edge_indices[0]]
            second_node = graph.node_features()[edge_indices[1]]
            interaction_features = torch.cat(
                [first_node, second_node, edge_features], dim=1
            )
            label_logits = self.interaction_classifier(interaction_features)
            suppress_logits = self.suppress_classifier(interaction_features)

            outputs.append(
                LayerOutput(label_logits=label_logits, suppress_logits=suppress_logits)
            )

        return Output(
            num_nodes=num_nodes,
            num_edges=num_edges,
            indices=edge_indices.transpose(0, 1),
            interaction_mask=mask,
            layers=outputs,
        )
