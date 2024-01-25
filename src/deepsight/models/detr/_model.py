##
##
##

import torch
from torch import nn

from deepsight.data import Batch
from deepsight.data.object_detection import Annotation, Prediction, Sample
from deepsight.models import Model as _Model
from deepsight.modules import SinusoidalImagePositionEmbedding, attention
from deepsight.structures import BatchedImages, BoundingBoxes, BoundingBoxFormat

from ._config import Config
from ._decoder import Decoder
from ._encoder import Encoder
from ._output import Output


class Model(_Model[Sample, Output, Annotation, Prediction]):
    """The DETR model for object detection."""

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.backbone = config.backbone
        self.input_proj = nn.Conv2d(
            in_channels=self.backbone.get_stages_info()[-1].out_channels,
            out_channels=config.embedding_dim,
            kernel_size=1,
        )

        self.position_embedding = SinusoidalImagePositionEmbedding(
            embed_dim=config.embedding_dim,
            temperature=config.pos_temperature,
            normalize=config.pos_normalize,
            xy_order="yx",
        )

        self.encoder = Encoder(config)

        self.query_embed = nn.Embedding(config.num_queries, config.embedding_dim)
        self.decoder = Decoder(config)

        self.class_head = nn.Linear(config.embedding_dim, config.num_classes + 1)
        self.bbox_head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.ReLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.ReLU(),
            nn.Linear(config.embedding_dim, 4),
        )

        self.threshold = config.threshold

    def __call__(
        self,
        samples: Batch[Sample],
        annotations: Batch[Annotation] | None = None,
    ) -> Output:
        images = BatchedImages.batch([s.image.data for s in samples])
        features = self.backbone(images, return_stages=(-1,))[-1]
        features = features.new_with(data=self.input_proj(features.data))

        pos_embeds = self.position_embedding(features)

        flattened_features = features.to_sequences()
        flattened_pos_embeds = pos_embeds.to_sequences()

        if flattened_features.is_padded():
            mask = attention.Mask.from_key_padding_mask(flattened_features.padding_mask)
        else:
            mask = None

        features = self.encoder(
            flattened_features.data, flattened_pos_embeds.data, mask
        )

        query_embeds = self.query_embed.weight[None].expand(len(samples), -1, -1)
        queries = self.decoder(
            query=torch.zeros_like(query_embeds),
            memory=features,
            query_pos=query_embeds,
            memory_pos=flattened_pos_embeds.data,
            memory_mask=mask,
        )

        class_logits = self.class_head(queries)
        bbox_preds = self.bbox_head(queries).sigmoid()

        return Output(class_logits, bbox_preds, images.image_sizes)

    def postprocess(self, output: Output) -> Batch[Prediction]:
        # take the output of the last decoder layer
        class_probs = output.class_logits[-1].softmax(-1)  # (B, Q, C + 1)
        bbox_coords = output.box_coords[-1]  # (B, Q, 4)

        scores, labels = class_probs[..., :-1].max(-1)  # (B, Q)

        # filter out predictions with low confidence scores
        keep = scores > self.threshold

        predictions = []
        for idx in range(len(output.image_sizes)):
            pred_coords = bbox_coords[idx][keep[idx]]  # (Q', 4)
            pred_scores = scores[idx][keep[idx]]
            pred_labels = labels[idx][keep[idx]]

            boxes = BoundingBoxes(
                pred_coords,
                format=BoundingBoxFormat.CXCYWH,
                normalized=True,
                image_size=output.image_sizes[idx],
            )

            predictions.append(Prediction(boxes, pred_labels, pred_scores))

        return Batch(predictions)
