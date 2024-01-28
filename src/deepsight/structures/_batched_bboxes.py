# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Literal, Self

import torch

from deepsight.typing import Tensor

from ._bboxes import BoundingBoxes, BoundingBoxFormat


class BatchedBoundingBoxes:
    """Structure to store a batch of multiple bounding boxes per image."""

    # ----------------------------------------------------------------------- #
    # Constructor and Factory Methods
    # ----------------------------------------------------------------------- #

    def __init__(
        self,
        coordinates: Tensor[Literal["B N 4"], float],
        format: BoundingBoxFormat | str,  # noqa: A002
        normalized: bool,
        image_sizes: tuple[tuple[int, int], ...],
        num_boxes_per_image: tuple[int, ...],
    ) -> None:
        """Initialize the bounding boxes."""
        if coordinates.ndim != 3 or coordinates.shape[-1] != 4:
            raise ValueError(
                "The coordinates must have shape (B, N, 4), "
                f"but got {coordinates.shape}."
            )

        if len(coordinates) != len(image_sizes) != len(num_boxes_per_image):
            raise ValueError(
                "The number of bounding boxes, image sizes, and number of boxes "
                f"per image must be the same, but got {len(coordinates)}, "
                f"{len(image_sizes)}, and {len(num_boxes_per_image)}."
            )

        self._coordinates = coordinates.float()
        self._format = BoundingBoxFormat(format)
        self._normalized = normalized
        self._image_sizes = image_sizes
        self._num_boxes = num_boxes_per_image

    @classmethod
    def batch(cls, boxes: Sequence[BoundingBoxes]) -> Self:
        """Batch a list of bounding boxes into a single tensor."""
        if len(boxes) == 0:
            raise ValueError("Cannot batch empty list of bounding boxes")

        boxes = [b.convert_like(boxes[0]) for b in boxes]

        num_boxes = tuple(len(b) for b in boxes)
        max_num_boxes = max(num_boxes)
        coordinates = boxes[0].coordinates.new_zeros((len(boxes), max_num_boxes, 4))

        for i, b in enumerate(boxes):
            coordinates[i, : len(b)].copy_(b.coordinates)

        return cls(
            coordinates=coordinates,
            format=boxes[0].format,
            normalized=boxes[0].normalized,
            image_sizes=tuple(b.image_size for b in boxes),
            num_boxes_per_image=num_boxes,
        )

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def coordinates(self) -> Tensor[Literal["B N 4"], float]:
        """The coordinates of the bounding boxes."""
        return self._coordinates

    @property
    def format(self) -> BoundingBoxFormat:  # noqa: A002
        """The format of the bounding boxes."""
        return self._format

    @property
    def normalized(self) -> bool:
        """Whether the bounding boxes are normalized with respect to the image size."""
        return self._normalized

    @property
    def image_sizes(self) -> tuple[tuple[int, int], ...]:
        """The image sizes of the bounding boxes."""
        return self._image_sizes

    @property
    def num_boxes_per_image(self) -> tuple[int, ...]:
        """The number of bounding boxes per image."""
        return self._num_boxes

    @property
    def device(self) -> torch.device:
        """The device of the bounding boxes."""
        return self._coordinates.device

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
            num_boxes_per_image=self._num_boxes,
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
            num_boxes_per_image=self._num_boxes,
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
            num_boxes_per_image=self._num_boxes,
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
        # swap width and height
        image_sizes = image_sizes.flip(-1)  # (B, 2)
        image_sizes = image_sizes[:, None]  # (B, 1, 2)
        norm_factor = image_sizes.repeat(1, 1, 2)  # (B, 1, 4)

        coordinates = self._coordinates / norm_factor

        return self.__class__(
            coordinates=coordinates,
            format=self._format,
            normalized=False,
            image_sizes=self._image_sizes,
            num_boxes_per_image=self._num_boxes,
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
        # swap width and height
        image_sizes = image_sizes.flip(-1)  # (B, 2)
        image_sizes = image_sizes[:, None]  # (B, 1, 2)
        norm_factor = image_sizes.repeat(1, 1, 2)  # (B, 1, 4)

        coordinates = self._coordinates * norm_factor

        return self.__class__(
            coordinates=coordinates,
            format=self._format,
            normalized=False,
            image_sizes=self._image_sizes,
            num_boxes_per_image=self._num_boxes,
        )

    def convert(
        self,
        format: BoundingBoxFormat | None = None,  # noqa: A002
        normalized: bool | None = None,
    ) -> Self:
        """Convert the bounding box coordinates to the given format and normalization."""  # noqa
        boxes = self
        match format:
            case BoundingBoxFormat.XYXY:
                boxes = self.to_xyxy()
            case BoundingBoxFormat.XYWH:
                boxes = self.to_xywh()
            case BoundingBoxFormat.CXCYWH:
                boxes = self.to_cxcywh()
            case None:
                pass

        match normalized:
            case True:
                boxes = boxes.normalize()
            case False:
                boxes = boxes.denormalize()
            case None:
                pass

        return boxes

    def convert_like(self, other: Self) -> Self:
        """Convert the bounding box coordinates to the same format and normalization as
        the given bounding box object."""  # noqa
        return self.convert(other.format, other.normalized)

    def area(self) -> Tensor[Literal["B N"], float]:
        """Compute the area of the bounding boxes.

        !!! note

            If the bounding box coordinates are normalized with respect to the
            image size, then the area is also normalized. Otherwise, the area
            is in pixels.
        """
        match self.format:
            case BoundingBoxFormat.XYXY:
                wh = self._coordinates[..., 2:] - self._coordinates[..., :2]
                area = wh[..., 0] * wh[..., 1]
            case BoundingBoxFormat.XYWH:
                area = self._coordinates[..., 2] * self._coordinates[..., 3]
            case BoundingBoxFormat.CXCYWH:
                wh = self._coordinates[..., 2:]
                area = wh[..., 0] * wh[..., 1]

        return area

    def aspect_ratio(self) -> Tensor[Literal["B Ns"], float]:
        """Compute the aspect ratio of the bounding boxes.

        The aspect ratio is computed as the width divided by the height.

        !!! note

            To avoid division by zero, a small epsilon is added to the height
            before computing the aspect ratio.
        """
        match self.format:
            case BoundingBoxFormat.XYXY:
                wh = self._coordinates[..., 2:] - self._coordinates[..., :2]
                w, h = wh.unbind(dim=-1)
            case BoundingBoxFormat.XYWH:
                w, h = self._coordinates[..., 2], self._coordinates[..., 3]
            case BoundingBoxFormat.CXCYWH:
                w, h = self._coordinates[..., 2], self._coordinates[..., 3]

        eps = torch.finfo(w.dtype).eps
        return w / (h + eps)

    def union(self, other: Self) -> Self:
        """Compute the union of the bounding boxes."""
        boxes1 = self.to_xyxy()
        boxes2 = other.convert_like(boxes1)
        boxes1._check_compatibility(boxes2)

        x1y1 = torch.min(boxes1._coordinates[..., :2], boxes2._coordinates[..., :2])
        x2y2 = torch.max(boxes1._coordinates[..., 2:], boxes2._coordinates[..., 2:])

        return self.__class__(
            coordinates=torch.cat([x1y1, x2y2], dim=-1),
            format=BoundingBoxFormat.XYXY,
            normalized=boxes1._normalized,
            image_sizes=boxes1._image_sizes,
            num_boxes_per_image=boxes1._num_boxes,
        )

    def intersection(self, other: Self) -> Self:
        """Compute the intersection of the bounding boxes."""
        boxes1 = self.to_xyxy()
        boxes2 = other.convert_like(boxes1)
        boxes1._check_compatibility(boxes2)

        x1y1 = torch.max(boxes1._coordinates[..., :2], boxes2._coordinates[..., :2])
        x2y2 = torch.min(boxes1._coordinates[..., 2:], boxes2._coordinates[..., 2:])
        wh = torch.clamp(x2y2 - x1y1, min=0)

        return self.__class__(
            coordinates=torch.cat([x1y1, wh], dim=-1),
            format=BoundingBoxFormat.XYWH,
            normalized=boxes1._normalized,
            image_sizes=boxes1._image_sizes,
            num_boxes_per_image=boxes1._num_boxes,
        )

    def union_area(self, other: Self) -> Tensor[Literal["B N"], float]:
        """Compute the union area of the bounding boxes.

        !!! note

            This is not the same as the area of the union of the bounding boxes.
            The union area is the sum of the areas of the bounding boxes minus
            the intersection area. To compute the area of the union of the
            bounding boxes, use `self.union(other).area()` instead.

        Args:
            other: The other bounding box object.

        Returns:
            The union area of the bounding boxes.

        Raises:
            ValueError: If the bounding boxes do not have the same normalization.
        """
        if self._normalized != other._normalized:
            raise ValueError(
                "The bounding boxes must have the same normalization, "
                f"but got {self._normalized} and {other._normalized}."
            )

        area1 = self.area()
        area2 = other.area()
        intersection = self.intersection_area(other)

        return area1 + area2 - intersection

    def intersection_area(self, other: Self) -> Tensor[Literal["B N"], float]:
        """Compute the intersection area of the bounding boxes.

        Args:
            other: The other bounding box object.

        Returns:
            The intersection area of the bounding boxes.

        Raises:
            ValueError: If the bounding boxes do not have the same normalization.
        """
        if self._normalized != other._normalized:
            raise ValueError(
                "The bounding boxes must have the same normalization, "
                f"but got {self._normalized} and {other._normalized}."
            )

        intersection = self.intersection(other).area()

        return intersection

    def iou(self, other: Self) -> Tensor[Literal["B N"], float]:
        """Compute the intersection over union (IoU) of the bounding boxes."""
        boxes1 = self.normalize()
        boxes2 = other.normalize()

        intersection_area = boxes1.intersection_area(boxes2)
        union_area = boxes1.union_area(boxes2)

        eps = torch.finfo(intersection_area.dtype).eps

        return intersection_area / (union_area + eps)

    def unbatch(self) -> tuple[BoundingBoxes, ...]:
        """Unbatch the bounding boxes into a list of bounding boxes."""
        return tuple(self[i] for i in range(len(self)))

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __getitem__(self, index: int) -> BoundingBoxes:
        """Get the bounding boxes for the given index."""
        coordinates = self._coordinates[index][: self._num_boxes[index]]

        return BoundingBoxes(
            coordinates=coordinates,
            format=self._format,
            normalized=self._normalized,
            image_size=self._image_sizes[index],
        )

    def __len__(self) -> int:
        """Get the batch size."""
        return len(self._coordinates)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"shape=({len(self), self._coordinates.shape[1]}), "
            f"format={self._format}, "
            f"normalized={self._normalized}, "
        )

    def __repr__(self) -> str:
        return str(self)

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _check_compatibility(self, other: Self) -> None:
        """Check that the given bounding boxes are compatible."""
        if self._coordinates.shape[:2] != other._coordinates.shape[:2]:
            raise ValueError(
                "The number of bounding boxes must be the same for both objects, "
                f"but got {self._coordinates.shape[:2]} and "
                f"{other._coordinates.shape[:2]}."
            )

        if self._image_sizes != other._image_sizes:
            raise ValueError(
                "The image sizes must be the same for both objects, "
                f"but got {self._image_sizes} and {other._image_sizes}."
            )

    # ----------------------------------------------------------------------- #
    # Private fields
    # ----------------------------------------------------------------------- #

    __slots__ = ("_coordinates", "_format", "_normalized", "_image_sizes", "_num_boxes")
