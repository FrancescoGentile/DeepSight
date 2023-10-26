##
##
##

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torchvision.ops import RoIAlign

from deepsight.models import DeepSightModel
from deepsight.structures import Batch, BatchedBoundingBoxes, BatchedSequences
from deepsight.tasks.hoic import Annotations, Predictions, Sample
from deepsight.typing import Configurable, JSONPrimitive

from ._encodings import (
    Sinusoidal2DPositionEncodings,
    compute_pairwise_spatial_encodings,
    compute_sinusoidal_position_encodings,
)
from ._fusion import MultiModalFusion
from ._structures import Output
from ._transformer import TranformerEncoder, TransformerDecoder
from ._vit_encoder import ViTEncoder


@dataclass(frozen=True)
class Config:
    """Configuration for PVIC model."""

    human_class_id: int
    num_interaction_classes: int
    allow_human_human: bool
    entity_embed_dim: int = 384
    image_embed_dim: int = 256
    image_encoder: str = "vit_base_patch16_224"
    image_size: tuple[int, int] = (224, 224)
    num_heads: int = 8
    dropout: float = 0.1
    temperature: float = 20
    entity_encoder_num_layers: int = 2
    ho_decoder_num_layers: int = 2


class PVIC(DeepSightModel[Sample, Output, Annotations, Predictions], Configurable):
    """The Predicate Visual Context (PVIC) model.

    This is a modified version of the model from the paper "Exploring Predicate Visual
    Context for Detecting Human-Object Interactions" by Zhang et al. (2023).
    """

    def __init__(self, config: Config) -> None:
        super().__init__()

        if config.image_embed_dim % 2 != 0:
            raise ValueError("image_embed_dim must be divisible by 2.")

        self._config = config

        self.human_class_id = config.human_class_id
        self.allow_human_human = config.allow_human_human
        self.image_embed_dim = config.image_embed_dim
        self.temperature = config.temperature

        self.vit_encoder = ViTEncoder(
            config.image_encoder, config.image_size, config.image_embed_dim
        )
        self.roi_align = RoIAlign(1, 1.0, -1, True)

        self.entity_encoder = TranformerEncoder(
            config.image_embed_dim,
            config.num_heads,
            config.dropout,
            config.entity_encoder_num_layers,
        )

        self.ref_anchor_head = nn.Sequential(
            nn.Linear(config.image_embed_dim, config.image_embed_dim),
            nn.ReLU(),
            nn.Linear(config.image_embed_dim, 2),
        )

        self.spatial_encoding_head = nn.Sequential(
            nn.Linear(36, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, config.entity_embed_dim),
            nn.ReLU(),
        )
        self.mmf = MultiModalFusion(
            2 * config.image_embed_dim, config.entity_embed_dim, config.entity_embed_dim
        )

        self.image_pos_encodings = Sinusoidal2DPositionEncodings(
            config.image_embed_dim // 2, temperature=config.temperature
        )
        self.ho_decoder = TransformerDecoder(
            config.entity_embed_dim,
            config.image_embed_dim,
            config.num_heads,
            config.dropout,
            config.ho_decoder_num_layers,
        )

        self.classifier = nn.Linear(
            config.entity_embed_dim, config.num_interaction_classes
        )

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_config(self) -> JSONPrimitive:
        return self._config.__dict__.copy()

    def forward(
        self, samples: Batch[Sample], annotations: Batch[Annotations] | None
    ) -> Output:
        images = self.vit_encoder((sample.image for sample in samples))
        H, W = images.shape[-2:]  # noqa
        boxes = [s.entities.resize((H, W)).denormalize().to_xyxy() for s in samples]
        coords = [box.coordinates for box in boxes]
        entity_embeddings = self.roi_align(images, coords)
        entity_embeddings = entity_embeddings.flatten(1)
        entity_embeddings = entity_embeddings.split_with_sizes([len(b) for b in boxes])

        entity_embeddings = BatchedSequences.batch(entity_embeddings)  # (B, N, C)
        boxes = BatchedBoundingBoxes.batch(boxes)
        box_pe, cx_pe = self._create_entity_positional_encodings(
            entity_embeddings, boxes
        )
        entity_embeddings = self.entity_encoder(entity_embeddings, box_pe)  # (B, N, C)

        ho_indices = []
        human_boxes = []
        object_boxes = []
        ho_embeddings = []
        ho_box_pe = []
        ho_cx_pe = []

        for idx, sample in enumerate(samples):
            pairs = self._create_matches(sample)
            ho_indices.append(pairs)

            human_boxes.append(sample.entities[pairs[:, 0]])
            object_boxes.append(sample.entities[pairs[:, 1]])

            human_embeds = entity_embeddings.data[idx, pairs[:, 0]]
            object_embeds = entity_embeddings.data[idx, pairs[:, 1]]
            ho_embeddings.append(torch.cat([human_embeds, object_embeds], dim=1))

            human_box_pe = box_pe[idx, pairs[:, 0]]
            object_box_pe = box_pe[idx, pairs[:, 1]]
            ho_box_pe.append(torch.cat([human_box_pe, object_box_pe], dim=1))

            human_cx_pe = cx_pe[idx, pairs[:, 0]]
            object_cx_pe = cx_pe[idx, pairs[:, 1]]
            ho_cx_pe.append(torch.cat([human_cx_pe, object_cx_pe], dim=1))

        human_boxes = BatchedBoundingBoxes.batch(human_boxes)  # (B, I, 4)
        object_boxes = BatchedBoundingBoxes.batch(object_boxes)  # (B, I, 4)
        ho_embeddings = BatchedSequences.batch(ho_embeddings)  # (B, I, 2C)
        ho_box_pe = pad_sequence(ho_box_pe, batch_first=True)  # (B, I, 4C)
        ho_cx_pe = pad_sequence(ho_cx_pe, batch_first=True)  # (B, I, 2C)

        # (B, I, 36)
        pairwise_se = compute_pairwise_spatial_encodings(human_boxes, object_boxes)
        pairwise_se = self.spatial_encoding_head(pairwise_se)  # (B, I, D)
        fused_ho_embeddings = self.mmf(ho_embeddings.data, pairwise_se)  # (B, I, D)
        ho_embeddings = ho_embeddings.replace(fused_ho_embeddings)

        images_pos_encodings = self.image_pos_encodings(images)
        images = images.to_sequences()
        images_pos_encodings = images_pos_encodings.flatten(2).transpose(1, 2)

        ho_embeddings = self.ho_decoder(
            ho_embeddings, ho_box_pe, ho_cx_pe, images, images_pos_encodings
        )

        ho_logits = self.classifier(ho_embeddings.data)  # (B, I, V)

        logits = []
        for idx, sample in enumerate(samples):
            logits.append(ho_logits[idx, : len(sample.entities)])

        return Output(logits, ho_indices, [len(s.entities) for s in samples])

    def postprocess(self, output: Output) -> Batch[Predictions]:
        raise NotImplementedError

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _create_matches(self, sample: Sample) -> Annotated[Tensor, "I 2", int]:
        human_labels = sample.entity_labels == self.human_class_id
        human_indices = human_labels.nonzero(as_tuple=True)[0]
        object_indices = (human_labels.logical_not_()).nonzero(as_tuple=True)[0]

        pairs = torch.cartesian_prod(human_indices, object_indices)

        if self.allow_human_human:
            human_human = torch.cartesian_prod(human_indices, human_indices)
            # remove self-loops
            keep = human_human[:, 0] != human_human[:, 1]
            human_human = human_human[keep]

            pairs = torch.cat([pairs, human_human], dim=0)

        return pairs

    def _create_entity_positional_encodings(
        self, embeddings: BatchedSequences, boxes: BatchedBoundingBoxes
    ) -> tuple[Annotated[Tensor, "B N 2D", float], Annotated[Tensor, "B N D", float]]:
        pos_dim = self.image_embed_dim // 2
        cx_pe, wh_pe = compute_sinusoidal_position_encodings(
            boxes, pos_dim, self.temperature
        )

        box_pe = torch.cat([cx_pe, wh_pe], dim=-1)

        ref_wh_cond = self.ref_anchor_head(embeddings.data)
        eps = torch.finfo(boxes.coordinates.dtype).eps

        boxes = boxes.to_cxcywh()
        cx_pe[..., :pos_dim] *= ref_wh_cond[..., 0] / (boxes.coordinates[..., 2] + eps)
        cx_pe[..., pos_dim:] *= ref_wh_cond[..., 1] / (boxes.coordinates[..., 3] + eps)

        return box_pe, cx_pe
