##
##
##

from collections.abc import Iterable
from typing import Annotated

import torch
from torch import Tensor
from typing_extensions import Self

from ._bboxes import BoundingBoxes, BoundingBoxFormat


class BatchedBoundingBoxes:
    # ----------------------------------------------------------------------- #
    # Constructor and Factory Methods
    # ----------------------------------------------------------------------- #

    def __init__(
        self,
        coordinates: Annotated[Tensor, "B N 4", float],
        format: BoundingBoxFormat,  # noqa: A002
        normalized: bool,
        image_sizes: tuple[tuple[int, int], ...],
    ) -> None:
        self._coordinates = coordinates
        self._format = format
        self._normalized = normalized
        self._image_sizes = image_sizes

    @classmethod
    def batch(cls, boxes: Iterable[BoundingBoxes]) -> Self:
        """Batch a list of bounding boxes into a single tensor."""
        boxes = list(boxes)

        if len(boxes) == 0:
            raise ValueError("Cannot batch empty list of bounding boxes")

        boxes = [b.convert_like(boxes[0]) for b in boxes]

        max_num_boxes = max(len(b) for b in boxes)
        coordinates = boxes[0].coordinates.new_zeros((len(boxes), max_num_boxes, 4))
        mask = torch.zeros(
            (len(boxes), max_num_boxes),
            dtype=torch.bool,
            device=boxes[0].coordinates.device,
        )

        for i, b in enumerate(boxes):
            coordinates[i, : len(b)].copy_(b.coordinates)
            mask[i, len(b) :] = True

        return cls(
            coordinates=coordinates,
            format=boxes[0].format,
            normalized=boxes[0].normalized,
            image_sizes=tuple(b.image_size for b in boxes),
        )

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def coordinates(self) -> Annotated[Tensor, "B N 4", float]:
        return self._coordinates

    @property
    def format(self) -> BoundingBoxFormat:  # noqa: A002
        return self._format

    @property
    def normalized(self) -> bool:
        return self._normalized

    @property
    def image_sizes(self) -> tuple[tuple[int, int], ...]:
        return self._image_sizes

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def to_xyxy(self) -> Self:
        """Convert bounding boxes to the XYXY format.

        !!! note

            If the bounding boxes are already in the XYXY format, then `self` is
            returned. Otherwise, a new `BatchedBoundingBoxes` instance is
            returned.
        """
        match self.format:
            case BoundingBoxFormat.XYXY:
                return self
            case BoundingBoxFormat.XYWH:
                xy = self._coordinates[..., :2]
                wh = self._coordinates[..., 2:]
                coordinates = torch.cat([xy, xy + wh], dim=-1)
            case BoundingBoxFormat.CXCYWH:
                xy = self._coordinates[..., :2] - self._coordinates[..., 2:] / 2
                wh = self._coordinates[..., 2:]
                coordinates = torch.cat([xy, xy + wh], dim=-1)

        return self.__class__(
            coordinates=coordinates,
            format=BoundingBoxFormat.XYXY,
            normalized=self._normalized,
            image_sizes=self._image_sizes,
        )

    def to_xywh(self) -> Self:
        """Convert bounding boxes to the XYWH format.

        !!! note

            If the bounding boxes are already in the XYWH format, then `self` is
            returned. Otherwise, a new `BatchedBoundingBoxes` instance is
            returned.
        """
        match self.format:
            case BoundingBoxFormat.XYXY:
                xy = self._coordinates[..., :2]
                wh = self._coordinates[..., 2:] - xy
                coordinates = torch.cat([xy, wh], dim=-1)
            case BoundingBoxFormat.XYWH:
                return self
            case BoundingBoxFormat.CXCYWH:
                xy = self._coordinates[..., :2] - self._coordinates[..., 2:] / 2
                wh = self._coordinates[..., 2:]
                coordinates = torch.cat([xy, wh], dim=-1)

        return self.__class__(
            coordinates=coordinates,
            format=BoundingBoxFormat.XYWH,
            normalized=self._normalized,
            image_sizes=self._image_sizes,
        )

    def to_cxcywh(self) -> Self:
        """Convert bounding boxes to the CXCWH format.

        !!! note

            If the bounding boxes are already in the CXCWH format, then `self` is
            returned. Otherwise, a new `BatchedBoundingBoxes` instance is
            returned.
        """
        match self.format:
            case BoundingBoxFormat.XYXY:
                xy = self._coordinates[..., :2]
                wh = self._coordinates[..., 2:] - xy
                coordinates = torch.cat([xy + wh / 2, wh], dim=-1)
            case BoundingBoxFormat.XYWH:
                xy = self._coordinates[..., :2]
                wh = self._coordinates[..., 2:]
                coordinates = torch.cat([xy + wh / 2, wh], dim=-1)
            case BoundingBoxFormat.CXCYWH:
                return self

        return self.__class__(
            coordinates=coordinates,
            format=BoundingBoxFormat.CXCYWH,
            normalized=self._normalized,
            image_sizes=self._image_sizes,
        )

    def normalize(self) -> Self:
        """Normalize the coordinates of the bounding boxes.

        !!! note

            If the bounding boxes are already normalized, then `self` is returned.
            Otherwise, a new `BatchedBoundingBoxes` instance is returned.
        """
        if self._normalized:
            return self

        image_sizes = torch.tensor(self._image_sizes, device=self._coordinates.device)
        image_sizes = image_sizes[:, None]  # (B, 1, 2)
        norm_factor = image_sizes.repeat(1, 1, 2)  # (B, 1, 4)

        coordinates = self._coordinates / norm_factor

        return self.__class__(
            coordinates=coordinates,
            format=self._format,
            normalized=False,
            image_sizes=self._image_sizes,
        )

    def denormalize(self) -> Self:
        """Denormalize the coordinates of the bounding boxes.

        !!! note

            If the bounding boxes are already denormalized, then `self` is returned.
            Otherwise, a new `BatchedBoundingBoxes` instance is returned.
        """
        if not self._normalized:
            return self

        image_sizes = torch.tensor(self._image_sizes, device=self._coordinates.device)
        image_sizes = image_sizes[:, None]  # (B, 1, 2)
        norm_factor = image_sizes.repeat(1, 1, 2)  # (B, 1, 4)

        coordinates = self._coordinates * norm_factor

        return self.__class__(
            coordinates=coordinates,
            format=self._format,
            normalized=False,
            image_sizes=self._image_sizes,
        )

    # ----------------------------------------------------------------------- #
    # Private fields
    # ----------------------------------------------------------------------- #

    __slots__ = ("_coordinates", "_format", "_normalized", "_image_sizes")
