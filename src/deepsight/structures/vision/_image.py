##
##
##

from __future__ import annotations

import enum
from collections.abc import Sequence
from typing import Literal, Self

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image as PILImage

from deepsight.typing import Moveable, Number, PathLike, Tensor, str_enum
from deepsight.utils import InterpolationMode


class Image(Moveable):
    """A wrapper around a tensor representing an image."""

    @str_enum
    class Mode(enum.Enum):
        """The image mode."""

        GRAYSCALE = "grayscale"
        RGB = "rgb"

        @classmethod
        def from_pil_mode(cls, mode: str) -> Self:
            """Convert a PIL mode to an image mode.

            Args:
                mode: The PIL mode.

            Returns:
                The image mode.
            """
            match mode:
                case "L":
                    return cls.GRAYSCALE
                case "RGB":
                    return cls.RGB
                case _:
                    raise ValueError(f"Unsupported PIL mode: {mode}")

        def to_pil_mode(self) -> str:
            """Convert the image mode to a PIL mode.

            Returns:
                The PIL mode.
            """
            match self:
                case self.GRAYSCALE:
                    return "L"
                case self.RGB:
                    return "RGB"

        def num_channels(self) -> int:
            """The number of channels."""
            match self:
                case self.GRAYSCALE:
                    return 1
                case self.RGB:
                    return 3

    # ----------------------------------------------------------------------- #
    # Constructor and Factory Methods
    # ----------------------------------------------------------------------- #

    def __init__(
        self,
        data: Tensor[Literal["* H W"], Number],
        mode: Mode | str,
    ) -> None:
        if data.ndim != 3:
            raise ValueError(
                f"Expected the image to be 3-dimensional. Got {data.ndim}-dimensional "
                f"image."
            )

        mode = self.Mode(mode)
        if data.shape[0] != mode.num_channels():
            raise ValueError(
                f"Expected the image to have {mode.num_channels()} channels. Got "
                f"{data.shape[0]} channels."
            )

        self._data = data
        self._mode = mode

    @classmethod
    def open(cls, path: PathLike, mode: Mode | str | None = None) -> Self:
        """Opens an image from a file.

        Args:
            path: The path to the image file.
            mode: The image mode to optionally convert the image to. If `None`, the
                image is not converted.

        Returns:
            The image.
        """
        pil_image = PILImage.open(str(path))
        if mode is not None:
            mode = cls.Mode(mode)
            pil_image = pil_image.convert(mode=mode.to_pil_mode())
        else:
            mode = cls.Mode.from_pil_mode(pil_image.mode)

        data = torch.from_numpy(np.array(pil_image))
        match mode:
            case cls.Mode.GRAYSCALE:
                if data.ndim == 2:
                    data = data.unsqueeze_(0)
            case cls.Mode.RGB:
                data = data.permute(2, 0, 1)

        return cls(data, mode)

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def data(self) -> Tensor[Literal["* H W"], Number]:
        """The underlying tensor."""
        return self._data

    @property
    def mode(self) -> Mode:
        """The image mode."""
        return self._mode

    @property
    def size(self) -> tuple[int, int]:
        """The size of the image as (height, width)."""
        return self.height, self.width

    @property
    def height(self) -> int:
        """The height of the image."""
        return self._data.shape[1]

    @property
    def width(self) -> int:
        """The width of the image."""
        return self._data.shape[2]

    @property
    def dtype(self) -> torch.dtype:
        return self._data.dtype

    @property
    def device(self) -> torch.device:
        return self._data.device

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> Self:
        if self.device == torch.device(device):
            return self

        return self.__class__(
            self._data.to(device, non_blocking=non_blocking),
            self._mode,
        )

    # ----------------------------------------------------------------------- #
    # Geometric Transformations
    # ----------------------------------------------------------------------- #

    def resize(
        self,
        size: tuple[int, int],
        interpolation_mode: InterpolationMode | str,
        antialias: bool = True,
    ) -> Self:
        """Resize the image.

        !!! note
            If the size is the same as the current size, `self` is returned.

        Args:
            size: The new size of the image as (height, width).
            interpolation_mode: The interpolation mode.
            antialias: Whether to use antialiasing.

        Returns:
            The resized image.

        Raises:
            ValueError: If `size` is not positive.
        """
        if size == self.size:
            return self
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError(
                f"Expected `size` to be positive. Got {size[0]}x{size[1]}."
            )

        data = F.interpolate(
            self._data.unsqueeze(0),
            size=size,
            mode=str(interpolation_mode),
            antialias=antialias,
        )
        data.squeeze_(0)

        return self.__class__(data, self._mode)

    def horizontal_flip(self) -> Self:
        """Flip the image horizontally."""
        data = self._data.flip(-1)
        return self.__class__(data, self._mode)

    # ----------------------------------------------------------------------- #
    # Color Transformations
    # ----------------------------------------------------------------------- #

    def to_grayscale(self) -> Self:
        """Convert the image to grayscale.

        !!! note

            If the image is already grayscale or binary, `self` is returned.
        """
        match self._mode:
            case self.Mode.GRAYSCALE:
                return self
            case self.Mode.RGB:
                data = _rgb_to_grayscale(
                    self._data, num_output_channels=1, preserve_dtype=True
                )
                return self.__class__(data, self.Mode.GRAYSCALE)

    def to_rgb(self) -> Self:
        """Convert the image to RGB."""
        match self._mode:
            case self.Mode.GRAYSCALE:
                data = self._data.expand(3, -1, -1)
                return self.__class__(data, self.Mode.RGB)
            case self.Mode.RGB:
                return self

    def to_mode(self, mode: Mode | str) -> Self:
        """Convert the image to a different mode.

        Args:
            mode: The image mode.

        Returns:
            The converted image.
        """
        mode = self.Mode(mode)
        match mode:
            case self.Mode.GRAYSCALE:
                return self.to_grayscale()
            case self.Mode.RGB:
                return self.to_rgb()

    def adjust_brightness(self, brightness_factor: float) -> Self:
        """Adjust the brightness of the image.

        Args:
            brightness_factor: The brightness factor.

        Returns:
            The adjusted image.

        Raises:
            ValueError: If `brightness_factor` is negative.
        """
        if brightness_factor < 0:
            raise ValueError(
                f"Expected `brightness_factor` to be non-negative. Got "
                f"{brightness_factor}."
            )

        bound = _max_dtype_value(self.dtype)
        data = self._data.mul(brightness_factor).clamp_(0, bound)
        data = data.to(dtype=self.dtype)

        return self.__class__(data, self._mode)

    def adjust_contrast(self, contrast_factor: float) -> Self:
        """Adjust the contrast of the image.

        Args:
            contrast_factor: The contrast factor.

        Returns:
            The adjusted image.

        Raises:
            ValueError: If `contrast_factor` is negative.
        """
        if contrast_factor < 0:
            raise ValueError(
                f"Expected `contrast_factor` to be non-negative. Got "
                f"{contrast_factor}."
            )

        match self._mode:
            case self.Mode.GRAYSCALE:
                grayscale_data = (
                    self._data
                    if self._data.is_floating_point()
                    else self._data.to(torch.float32)
                )
            case self.Mode.RGB:
                grayscale_data = _rgb_to_grayscale(
                    self._data, num_output_channels=1, preserve_dtype=False
                )
                if not self.dtype.is_floating_point:
                    grayscale_data = grayscale_data.floor_()

        mean = torch.mean(grayscale_data, dim=(-3, -2, -1), keepdim=True)
        data = _blend(self._data, mean, contrast_factor)

        return self.__class__(data, self._mode)

    def adjust_saturation(self, saturation_factor: float) -> Self:
        """Adjust the saturation of the image.

        Args:
            saturation_factor: The saturation factor.

        Returns:
            The adjusted image.

        Raises:
            ValueError: If `saturation_factor` is negative.
        """
        if saturation_factor < 0:
            raise ValueError(
                f"Expected `saturation_factor` to be non-negative. Got "
                f"{saturation_factor}."
            )

        match self.mode:
            case self.Mode.GRAYSCALE:
                return self
            case self.Mode.RGB:
                grayscale_data = _rgb_to_grayscale(
                    self._data,
                    num_output_channels=1,
                    preserve_dtype=False,
                )
                if not grayscale_data.is_floating_point():
                    grayscale_data = grayscale_data.floor_()

                data = _blend(self._data, grayscale_data, saturation_factor)

                return self.__class__(data, self._mode)

    def adjust_hue(self, hue_factor: float) -> Self:
        """Adjust the hue of the image.

        Args:
            hue_factor: The hue factor.

        Returns:
            The adjusted image.

        Raises:
            ValueError: If `hue_factor` is not in the range [-0.5, 0.5].
        """
        if not -0.5 <= hue_factor <= 0.5:
            raise ValueError(
                f"Expected `hue_factor` to be in the range [-0.5, 0.5]. Got "
                f"{hue_factor}."
            )

        match self.mode:
            case self.Mode.GRAYSCALE:
                return self
            case self.Mode.RGB:
                if self._data.numel() == 0:
                    return self

                image = self.to_dtype(torch.float32, scale=True)
                hsv_data = _rgb_to_hsv(image.data)
                h, s, v = hsv_data.unbind(dim=0)
                h.add_(hue_factor).remainder_(1.0)
                hsv_data = torch.stack((h, s, v), dim=0)
                data = _hsv_to_rgb(hsv_data)
                data = _to_dtype(data, image.dtype, scale=True)

                return self.__class__(data, self._mode)

    # ----------------------------------------------------------------------- #
    # Miscellaneous Transformations
    # ----------------------------------------------------------------------- #

    def standardize(self, mean: Sequence[float], std: Sequence[float]) -> Self:
        """Standardize the image.

        Args:
            mean: The mean values for each channel.
            std: The standard deviation values for each channel.

        Returns:
            The standardized image.
        """
        if not self._data.is_floating_point():
            raise NotImplementedError(
                f"Currently, standardization is only supported for floating-point "
                f"images. Got {self.dtype}."
            )

        if len(mean) != self._data.shape[0]:
            raise ValueError(
                f"The number of values in `mean` must match the number of channels in "
                f"the image. Got {len(mean)} values for {self._data.shape[0]} channels."
            )

        if len(std) != self._data.shape[0]:
            raise ValueError(
                f"The number of values in `std` must match the number of channels in "
                f"the image. Got {len(std)} values for {self._data.shape[0]} channels."
            )

        mean_tensor = torch.as_tensor(mean, dtype=self.dtype, device=self.device)
        std_tensor = torch.as_tensor(std, dtype=self.dtype, device=self.device)
        mean_tensor = mean_tensor.view(-1, 1, 1)
        std_tensor = std_tensor.view(-1, 1, 1)

        data = self.data.clone()
        data.sub_(mean_tensor).div_(std_tensor)

        return self.__class__(data, self._mode)

    def to_dtype(self, dtype: torch.dtype, scale: bool) -> Self:
        """Convert the image to a different dtype."""
        data = _to_dtype(self._data, dtype, scale)
        return self.__class__(data, self._mode)

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"

    def __str__(self) -> str:
        return repr(self)

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    # ----------------------------------------------------------------------- #
    # Private fields
    # ----------------------------------------------------------------------- #

    __slots__ = ("_data",)


# --------------------------------------------------------------------------- #
# Private Functions
# --------------------------------------------------------------------------- #


def _max_dtype_value(dtype: torch.dtype) -> int:
    """The maximum value of a dtype.

    Args:
        dtype: The dtype.

    Returns:
        The maximum value.
    """
    match dtype:
        case torch.uint8:
            return 255
        case torch.int8:
            return 127
        case torch.int16:
            return 32767
        case torch.int32:
            return 2147483647
        case torch.int64:
            return 9223372036854775807
        case _:
            if dtype.is_floating_point:
                return 1
            else:
                raise ValueError(f"Unsupported dtype: {dtype}.")


def _num_value_bits(dtype: torch.dtype) -> int:
    match dtype:
        case torch.uint8:
            return 8
        case torch.int8:
            return 7
        case torch.int16:
            return 15
        case torch.int32:
            return 31
        case torch.int64:
            return 63
        case _:
            raise TypeError(
                "Number of value bits is only defined for integer dtypes, "
                f"but got {dtype}."
            )


def _rgb_to_grayscale(
    data: Tensor[Literal["3 H W"], Number],
    num_output_channels: int = 1,
    preserve_dtype: bool = False,
) -> Tensor[Literal["(1|3) H W"], Number]:
    if data.shape[0] != 3:
        raise ValueError(
            f"Expected the image to have 3 channels. Got {data.shape[0]} channels."
        )

    orig_dtype = data.dtype
    r, g, b = data.unbind(0)
    data = r.mul(0.2989).add_(g, alpha=0.587).add_(b, alpha=0.114)
    data = data.unsqueeze(0)

    if preserve_dtype:
        data = data.to(dtype=orig_dtype)
    if num_output_channels == 3:
        data = data.expand(3, -1, -1)

    return data


# https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/functional/_color.py#L272
def _rgb_to_hsv(
    data: Tensor[Literal["3 H W"], Number]
) -> Tensor[Literal["3 H W"], Number]:
    r, g, _ = data.unbind(dim=-3)

    # Implementation is based on
    # https://github.com/python-pillow/Pillow/blob/4174d4267616897df3746d315d5a2d0f82c656ee/src/libImaging/Convert.c#L330
    minc, maxc = torch.aminmax(data, dim=-3)

    # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
    # from happening in the results, because
    #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
    #   + H channel has division by `(maxc - minc)`.
    #
    # Instead of overwriting NaN afterwards, we just prevent it from occurring so
    # we don't need to deal with it in case we save the NaN in a buffer in
    # backprop, if it is ever supported, but it doesn't hurt to do so.
    eqc = maxc == minc

    channels_range = maxc - minc
    # Since `eqc => channels_range = 0`, replacing denominator with 1 when `eqc` is fine
    ones = torch.ones_like(maxc)
    s = channels_range / torch.where(eqc, ones, maxc)
    # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
    # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
    # would not matter what values `rc`, `gc`, and `bc` have here, and thus
    # replacing denominator with 1 when `eqc` is fine.
    channels_range_divisor = torch.where(eqc, ones, channels_range).unsqueeze_(dim=-3)
    rc, gc, bc = ((maxc.unsqueeze(dim=-3) - data) / channels_range_divisor).unbind(
        dim=-3
    )

    mask_maxc_neq_r = maxc != r
    mask_maxc_eq_g = maxc == g

    hg = rc.add(2.0).sub_(bc).mul_(mask_maxc_eq_g & mask_maxc_neq_r)
    hr = bc.sub_(gc).mul_(~mask_maxc_neq_r)
    hb = (
        gc.add_(4.0)
        .sub_(rc)
        .mul_(mask_maxc_neq_r.logical_and_(mask_maxc_eq_g.logical_not_()))
    )

    h = hr.add_(hg).add_(hb)
    h = h.mul_(1.0 / 6.0).add_(1.0).fmod_(1.0)
    return torch.stack((h, s, maxc), dim=-3)


def _hsv_to_rgb(
    data: Tensor[Literal["3 H W"], Number]
) -> Tensor[Literal["3 H W"], Number]:
    h, s, v = data.unbind(dim=-3)
    h6 = h.mul(6)
    i = torch.floor(h6)
    f = h6.sub_(i)
    i = i.to(dtype=torch.int32)

    sxf = s * f
    one_minus_s = 1.0 - s
    q = (1.0 - sxf).mul_(v).clamp_(0.0, 1.0)
    t = sxf.add_(one_minus_s).mul_(v).clamp_(0.0, 1.0)
    p = one_minus_s.mul_(v).clamp_(0.0, 1.0)
    i.remainder_(6)

    vpqt = torch.stack((v, p, q, t), dim=-3)

    # vpqt -> rgb mapping based on i
    select = torch.tensor(
        [[0, 2, 1, 1, 3, 0], [3, 0, 0, 2, 1, 1], [1, 1, 3, 0, 0, 2]], dtype=torch.long
    )
    select = select.to(device=data.device, non_blocking=True)

    select = select[:, i]
    if select.ndim > 3:
        # if input.shape is (B, ..., C, H, W) then
        # select.shape is (C, B, ...,  H, W)
        # thus we move C axis to get (B, ..., C, H, W)
        select = select.moveaxis(0, -3)

    return vpqt.gather(-3, select)


def _blend(
    data1: Tensor[Literal["* H W"], Number],
    data2: Tensor[Literal["* H W"], Number],
    ratio: float,
) -> Tensor[Literal["* H W"], Number]:
    """Blend two images.

    Args:
        data1: The first image.
        data2: The second image.
        ratio: The ratio of the first image.

    Returns:
        The blended image.
    """
    orig_dtype = data1.dtype
    bound = _max_dtype_value(data1.dtype)
    output = data1.mul(ratio).add_(data2, alpha=(1 - ratio)).clamp_(0, bound)
    output = output.to(dtype=orig_dtype)

    return output


def _to_dtype(
    data: Tensor[Literal["* H W"], Number],
    dtype: torch.dtype,
    scale: bool,
) -> Tensor[Literal["* H W"], Number]:
    if data.dtype == dtype:
        return data
    elif not scale:
        return data.to(dtype=dtype)

    if data.is_floating_point():
        if dtype.is_floating_point:
            return data.to(dtype=dtype)

        if (data.dtype == torch.float32 and dtype in (torch.int32, torch.int64)) or (
            data.dtype == torch.float64 and dtype == torch.int64
        ):
            raise RuntimeError(
                f"The conversion from {data.dtype} to {dtype} cannot be performed "
                "without loss of precision."
            )

        eps = 1e-3
        max_value = float(_max_dtype_value(dtype))
        return data.mul(max_value + 1.0 - eps).to(dtype)
    else:
        if dtype.is_floating_point:
            return data.to(dtype).mul_(1.0 / _max_dtype_value(data.dtype))

        num_value_bits_input = _num_value_bits(data.dtype)
        num_value_bits_output = _num_value_bits(dtype)

        if num_value_bits_input > num_value_bits_output:
            return data.bitwise_right_shift(
                num_value_bits_input - num_value_bits_output
            ).to(dtype)
        else:
            return data.to(dtype).bitwise_left_shift_(
                num_value_bits_output - num_value_bits_input
            )
