##
##
##

from typing import Annotated

import torch
import torch.nn.functional as F  # noqa
from torch import Tensor, nn

from deepsight.structures import BatchedBoundingBoxes, BatchedImages


def compute_pairwise_spatial_encodings(
    human_boxes: BatchedBoundingBoxes, object_boxes: BatchedBoundingBoxes
) -> Annotated[Tensor, "B N 36", float]:
    encodings = []  # list of tensors of shape (B, N)

    human_boxes = human_boxes.normalize().to_cxcywh()
    object_boxes = object_boxes.normalize().to_cxcywh()

    encodings.extend([human_boxes.coordinates[:, i] for i in range(4)])
    encodings.extend([object_boxes.coordinates[:, i] for i in range(4)])

    human_area = human_boxes.area()
    object_area = object_boxes.area()

    encodings.append(human_area)
    encodings.append(object_area)
    encodings.append(object_area / human_area)

    encodings.append(human_boxes.aspect_ratio())
    encodings.append(object_boxes.aspect_ratio())

    encodings.append(human_boxes.iou(object_boxes))

    dx = human_boxes.coordinates[:, 0] - object_boxes.coordinates[:, 0]
    dx = dx / human_boxes.coordinates[:, 2]

    dy = human_boxes.coordinates[:, 1] - object_boxes.coordinates[:, 1]
    dy = dy / human_boxes.coordinates[:, 3]

    encodings.extend([F.relu(dx), F.relu(-dx), F.relu(dy), F.relu(-dy)])

    encodings = torch.stack(encodings, dim=2)  # (B, N, 18)
    eps = torch.finfo(encodings.dtype).eps
    log_encodings = torch.log(encodings + eps)
    encodings = torch.cat([encodings, log_encodings], dim=2)  # (B, N, 36)

    return encodings


def compute_sinusoidal_position_encodings(
    boxes: BatchedBoundingBoxes, dimension: int, temperature: float
) -> tuple[Annotated[Tensor, "B N D", float], Annotated[Tensor, "B N D", float]]:
    """Compute sinusoidal position encodings for the given boxes.

    This method computes the sinusoidal positional encodings for the center and the size
    of the given boxes.

    Args:
        boxes: A batch of bounding boxes.
        dimension: The dimension of the positional encodings. The final positional
            encodings will have a shape of (B, N, 2 * dimension).
        temperature: The temperature of the positional encodings.

    Returns:
        A tuple containing the positional encodings for the center and the size of the
        given boxes.
    """
    boxes = boxes.normalize().to_cxcywh()
    positions = [boxes.coordinates[..., :2], boxes.coordinates[..., 2:]]
    encodings: list[Tensor] = []

    scale = 2 * torch.pi
    for pos in positions:
        dim_t = torch.arange(dimension, dtype=torch.float, device=pos.device)
        dim_t = temperature ** (2 * (dim_t // 2) / dimension)
        pos_x = (pos[..., 0] * scale).unsqueeze(-1) / dim_t  # (B, N, D)
        pos_y = (pos[..., 1] * scale).unsqueeze(-1) / dim_t  # (B, N, D)
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=3)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=3)
        pos = torch.cat([pos_x.flatten(2), pos_y.flatten(2)], dim=2)
        encodings.append(pos)

    return tuple(encodings)  # type: ignore


class Sinusoidal2DPositionEncodings(nn.Module):
    def __init__(
        self,
        dimension: int,
        temperature: float,
        normalize: bool = False,
        scale: float = 2 * torch.pi,
    ) -> None:
        super().__init__()

        self.dimension = dimension
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(self, images: BatchedImages) -> Annotated[Tensor, "B 2D H W", float]:
        not_mask = images.mask.logical_not()
        pos_y = not_mask.cumsum(dim=1, dtype=torch.float)
        pos_x = not_mask.cumsum(dim=2, dtype=torch.float)

        if self.normalize:
            eps = torch.finfo(pos_y.dtype).eps
            pos_y = pos_y / (pos_y[:, -1:, :] + eps) * self.scale
            pos_x = pos_x / (pos_x[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.dimension, dtype=torch.float, device=images.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.dimension)

        pos_x = pos_x.unsqueeze(3) / dim_t
        pos_y = pos_y.unsqueeze(3) / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4)
        pos = torch.cat([pos_x.flatten(3), pos_y.flatten(3)], dim=3)
        pos = pos.permute(0, 3, 1, 2)

        return pos

    def __call__(self, images: BatchedImages) -> Annotated[Tensor, "B 2D H W", float]:
        return super().__call__(images)
