##
##
##

from typing import Literal

import torch

from deepsight.typing import Number, Tensor


def max_dtype_value(dtype: torch.dtype) -> int:
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


def num_value_bits(dtype: torch.dtype) -> int:
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


def rgb_to_grayscale(
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
def rgb_to_hsv(
    data: Tensor[Literal["3 H W"], Number],
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


def hsv_to_rgb(
    data: Tensor[Literal["3 H W"], Number],
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


def blend(
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
    bound = max_dtype_value(data1.dtype)
    output = data1.mul(ratio).add_(data2, alpha=(1 - ratio)).clamp_(0, bound)
    output = output.to(dtype=orig_dtype)

    return output


def to_dtype(
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
        max_value = float(max_dtype_value(dtype))
        return data.mul(max_value + 1.0 - eps).to(dtype)
    else:
        if dtype.is_floating_point:
            return data.to(dtype).mul_(1.0 / max_dtype_value(data.dtype))

        num_value_bits_input = num_value_bits(data.dtype)
        num_value_bits_output = num_value_bits(dtype)

        if num_value_bits_input > num_value_bits_output:
            return data.bitwise_right_shift(
                num_value_bits_input - num_value_bits_output
            ).to(dtype)
        else:
            return data.to(dtype).bitwise_left_shift_(
                num_value_bits_output - num_value_bits_input
            )
