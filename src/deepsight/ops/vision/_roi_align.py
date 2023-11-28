##
##
##

from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn

from deepsight import utils
from deepsight.structures.vision import BatchedImages, BoundingBoxes
from deepsight.typing import Tensor


class RoIAlign(nn.Module):
    def __init__(
        self,
        output_size: int | tuple[int, int],
        sampling_ratio: int = -1,
        aligned: bool = False,
    ) -> None:
        super().__init__()

        self._output_size = utils.to_2tuple(output_size)
        self._sampling_ratio = sampling_ratio
        self._aligned = aligned

        raise NotImplementedError

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def forward(
        self,
        images: BatchedImages | Tensor[Literal["B C H W"], float],
        boxes: Sequence[BoundingBoxes],
    ) -> Tensor[Literal["K C h w"], float]:
        if images.shape[0] != len(boxes):
            raise ValueError(
                f"Expected `images` and `boxes` to have the same batch size, "
                f"got {images.shape[0]} and {len(boxes)} respectively."
            )

        if isinstance(images, BatchedImages):
            image_sizes = images.image_sizes
        else:
            image_sizes = [(images.shape[2], images.shape[3])] * len(boxes)

        if any(
            box.image_size != image_size
            for box, image_size in zip(boxes, image_sizes, strict=True)
        ):
            raise ValueError("Inconsistent image sizes between `boxes` and `images`.")

        rois = _convert_boxes_to_roi_format(boxes)
        output = _roi_align(
            images if isinstance(images, torch.Tensor) else images.data,
            rois,
            self._output_size[0],
            self._output_size[1],
            self._sampling_ratio,
            self._aligned,
        )

        return output

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __call__(
        self,
        images: BatchedImages | Tensor[Literal["B C H W"], float],
        boxes: Sequence[BoundingBoxes],
    ) -> Tensor[Literal["K C h w"], float]:
        return super().__call__(images, boxes)


# --------------------------------------------------------------------------- #
# Private functions
# --------------------------------------------------------------------------- #

# Taken from torchvision (https://github.com/pytorch/vision/blob/main/torchvision/ops/roi_align.py)


def _convert_boxes_to_roi_format(
    boxes: Sequence[BoundingBoxes]
) -> Tensor[Literal["K 5"], float]:
    coords = [box.to_xyxy().denormalize().coordinates for box in boxes]
    coords = torch.cat(coords, dim=0)  # (K, 4)

    indices = []
    for idx, box in enumerate(boxes):
        indices.append(torch.full_like(box.coordinates[:, :1], idx))
    indices = torch.cat(indices, dim=0)  # (K, 1)

    rois = torch.cat([indices, coords], dim=-1)  # (K, 5)
    return rois


def _roi_align(
    images: Tensor[Literal["B C H W"], float],
    rois: Tensor[Literal["K 5"], float],
    pooled_height: int,
    pooled_width: int,
    sampling_ratio: int,
    aligned: bool,
) -> Tensor[Literal["K C h w"], float]:
    orig_dtype = images.dtype
    images = _maybe_cast(images)

    H, W = images.shape[2:]  # noqa: N806
    ph = torch.arange(pooled_height, device=images.device)
    pw = torch.arange(pooled_width, device=images.device)

    roi_batch_ind = rois[:, 0].int()  # (K,)
    offset = 0.5 if aligned else 0.0
    roi_start_w = rois[:, 1] * -offset  # (K,)
    roi_start_h = rois[:, 2] * -offset  # (K,)
    roi_end_w = rois[:, 3] * -offset  # (K,)
    roi_end_h = rois[:, 4] * -offset  # (K,)

    roi_width = roi_end_w - roi_start_w  # (K,)
    roi_height = roi_end_h - roi_start_h  # (K,)
    if not aligned:
        roi_width = roi_width.clamp(min=1.0)
        roi_height = roi_height.clamp(min=1.0)

    bin_size_h = roi_height / pooled_height  # (K,)
    bin_size_w = roi_width / pooled_width  # (K,)

    exact_sampling = sampling_ratio > 0

    if exact_sampling:
        roi_bin_grid_h = sampling_ratio
        roi_bin_grid_w = sampling_ratio

        count = max(roi_bin_grid_h * roi_bin_grid_w, 1)
        iy = torch.arange(roi_bin_grid_h, device=images.device)  # (IY,)
        ix = torch.arange(roi_bin_grid_w, device=images.device)  # (IX,)
        ymask, xmask = None, None
    else:
        roi_bin_grid_h = torch.ceil(roi_height / pooled_height)
        roi_bin_grid_w = torch.ceil(roi_width / pooled_width)

        count = torch.clamp(roi_bin_grid_h * roi_bin_grid_w, min=1)
        iy = torch.arange(H, device=images.device)  # (IY,)
        ix = torch.arange(W, device=images.device)  # (IX,)
        ymask = iy[None, :] < roi_bin_grid_h[:, None]  # (K, IY)
        xmask = ix[None, :] < roi_bin_grid_w[:, None]  # (K, IX)

    def from_k(tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:, None, None]

    y = (
        from_k(roi_start_h)
        + ph[None, :, None] * from_k(bin_size_h)
        + (iy[None, None, :] + 0.5).to(images.dtype)
        * from_k(bin_size_h / roi_bin_grid_h)
    )  # (K, PH, IY)
    x = (
        from_k(roi_start_w)
        + pw[None, :, None] * from_k(bin_size_w)
        + (ix[None, None, :] + 0.5).to(images.dtype)
        * from_k(bin_size_w / roi_bin_grid_w)
    )  # (K, PW, IX)
    val = _bilinear_interpolate(
        images, roi_batch_ind, y, x, ymask, xmask
    )  # (K, C, PH, PW, IY, IX)

    if not exact_sampling:
        assert ymask is not None
        assert xmask is not None
        val = torch.where(ymask[:, None, None, None, :, None], val, 0)
        val = torch.where(xmask[:, None, None, None, None, :], val, 0)

    output = val.sum((-1, -2))  # remove IY, IX ~> (K, C, PH, PW)
    if isinstance(count, torch.Tensor):
        output /= count[:, None, None, None]
    else:
        output /= count

    output = output.to(orig_dtype)

    return output


def _maybe_cast(tensor: torch.Tensor) -> torch.Tensor:
    if torch.is_autocast_enabled() and tensor.is_cuda and tensor.dtype != torch.double:
        return tensor.float()
    return tensor


def _bilinear_interpolate(
    images: Tensor[Literal["B C H W"], float],
    roi_batch_ind: Tensor[Literal["K"], int],
    y: Tensor[Literal["K PH IY"], float],
    x: Tensor[Literal["K PW IX"], float],
    ymask: Tensor[Literal["K IY"], bool] | None,
    xmask: Tensor[Literal["K IX"], bool] | None,
) -> Tensor[Literal["K C PH PW IY IX"], float]:
    _, channels, height, width = images.size()

    # deal with inverse element out of feature map boundary
    y = y.clamp(min=0)
    x = x.clamp(min=0)
    y_low = y.int()
    x_low = x.int()
    y_high = torch.where(y_low >= height - 1, height - 1, y_low + 1)
    y_low = torch.where(y_low >= height - 1, height - 1, y_low)
    y = torch.where(y_low >= height - 1, y.to(images.dtype), y)

    x_high = torch.where(x_low >= width - 1, width - 1, x_low + 1)
    x_low = torch.where(x_low >= width - 1, width - 1, x_low)
    x = torch.where(x_low >= width - 1, x.to(images.dtype), x)

    ly = y - y_low
    lx = x - x_low
    hy = 1.0 - ly
    hx = 1.0 - lx

    # do bilinear interpolation, but respect the masking!
    # TODO: It's possible the masking here is unnecessary if y and
    # x were clamped appropriately; hard to tell
    def masked_index(
        y: Tensor[Literal["K PH IY"], float],
        x: Tensor[Literal["K PW IX"], float],
    ) -> Tensor[Literal["K C PH PW IY IX"], float]:
        if ymask is not None:
            assert xmask is not None
            y = torch.where(ymask[:, None, :], y, 0)
            x = torch.where(xmask[:, None, :], x, 0)
        return images[
            roi_batch_ind[:, None, None, None, None, None],
            torch.arange(channels, device=images.device)[
                None, :, None, None, None, None
            ],
            y[:, None, :, None, :, None],  # prev (K, PH, IY)
            x[:, None, None, :, None, :],  # prev (K, PW, IX)
        ]  # (K, C, PH, PW, IY, IX)

    v1 = masked_index(y_low, x_low)
    v2 = masked_index(y_low, x_high)
    v3 = masked_index(y_high, x_low)
    v4 = masked_index(y_high, x_high)

    # all ws preemptively (K, C, PH, PW, IY, IX)
    def outer_prod(
        y: Tensor[Literal["K PH IY"], float], x: Tensor[Literal["K PW IX"], float]
    ) -> Tensor[Literal["K PH PW IY IX"], float]:
        return y[:, None, :, None, :, None] * x[:, None, None, :, None, :]

    w1 = outer_prod(hy, hx)
    w2 = outer_prod(hy, lx)
    w3 = outer_prod(ly, hx)
    w4 = outer_prod(ly, lx)

    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    return val
