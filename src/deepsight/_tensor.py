##
##
##

from collections.abc import Sequence
from typing import Any, LiteralString, Self

import numpy.typing as npt
import torch

from deepsight.typing import DeviceLike, Number


class Tensor[S: LiteralString, DT](torch.Tensor):
    # ----------------------------------------------------------------------- #
    # Constructor and Factory Methods
    # ----------------------------------------------------------------------- #

    def __new__(cls, data: Any, device: DeviceLike | None = None) -> Self:
        tensor = torch.as_tensor(data, device=device)
        return tensor.as_subclass(cls)

    @classmethod
    def from_numpy(cls, ndarray: npt.NDArray[Any]) -> Self:
        """Creates a tensor from a numpy array.

        This method behaves as `torch.from_numpy`, thus the returned tensor
        will share memory with the input numpy array. Changes to the tensor will
        be reflected in the numpy array and vice versa.

        !!! note

            See [torch.from_numpy][] for more information.

        Args:
            ndarray: The numpy array to create the tensor from.

        Raises:
            ValueError: If the input numpy array is read-only.
        """
        if ndarray.flags.writeable is False:
            msg = "The input numpy array is read-only."
            raise ValueError(msg)

        tensor = torch.from_numpy(ndarray)
        return tensor.as_subclass(cls)

    @classmethod
    def from_data(
        cls,
        data: Any,
        dtype: torch.dtype | None = None,
        device: DeviceLike | None = None,
    ) -> Self:
        """Creates a tensor from the given data.

        This method behaves as `torch.as_tensor`, thus it will attempt to share
        memory and autograd history with the input data when possible.

        !!! note

            See [torch.as_tensor][] for more information.

        Args:
            data: The data to create the tensor from. This can be a scalar, a
                list, a tuple, a numpy array, a torch tensor, etc.
            dtype: The dtype of the tensor. If `None`, the dtype will be
                inferred from the data.
            device: The device to create the tensor on. If `None` and the data
                is a tensor, the tensor's device will be used. If `None` and the
                data is not a tensor, the current device will be used.
        """
        tensor = torch.as_tensor(data, dtype=dtype, device=device)
        return tensor.as_subclass(cls)

    @classmethod
    def empty(
        cls,
        shape: Sequence[int],
        *,
        dtype: torch.dtype | None = None,
        device: DeviceLike | None = None,
        pin_memory: bool = False,
        requires_grad: bool = False,
    ) -> Self:
        """Creates a tensor filled with uninitialized data.

        !!! note

            See [torch.empty][] for more information.

        Args:
            shape: The shape of the tensor.
            dtype: The desired dtype of the tensor. If `None`, the default dtype
                will be used.
            device: The desired device of the tensor. If `None`, the current
                device will be used.
            pin_memory: If `True`, the tensor will be allocated in pinned memory.
                This is only applicable for CPU tensors.
            requires_grad: If `True`, the autograd system will track operations
                on the tensor.
        """
        tensor = torch.empty(
            shape,
            dtype=dtype,
            device=device,
            pin_memory=pin_memory,
            requires_grad=requires_grad,
        )
        return tensor.as_subclass(cls)

    @classmethod
    def zeros(
        cls,
        shape: Sequence[int],
        *,
        dtype: torch.dtype | None = None,
        device: DeviceLike | None = None,
        pin_memory: bool = False,
        requires_grad: bool = False,
    ) -> Self:
        """Creates a tensor filled with zeros.

        !!! note

            See [torch.zeros][] for more information.

        Args:
            shape: The shape of the tensor.
            dtype: The desired dtype of the tensor. If `None`, the default dtype
                will be used.
            device: The desired device of the tensor. If `None`, the current
                device will be used.
            pin_memory: If `True`, the tensor will be allocated in pinned memory.
                This is only applicable for CPU tensors.
            requires_grad: If `True`, the autograd system will track operations
                on the tensor.
        """
        tensor = torch.zeros(
            shape,
            dtype=dtype,
            device=device,
            pin_memory=pin_memory,
            requires_grad=requires_grad,
        )
        return tensor.as_subclass(cls)

    @classmethod
    def ones(
        cls,
        shape: Sequence[int],
        *,
        dtype: torch.dtype | None = None,
        device: DeviceLike | None = None,
        pin_memory: bool = False,
        requires_grad: bool = False,
    ) -> Self:
        """Creates a tensor filled with ones.

        !!! note

            See [torch.ones][] for more information.

        Args:
            shape: The shape of the tensor.
            dtype: The desired dtype of the tensor. If `None`, the default dtype
                will be used.
            device: The desired device of the tensor. If `None`, the current
                device will be used.
            pin_memory: If `True`, the tensor will be allocated in pinned memory.
                This is only applicable for CPU tensors.
            requires_grad: If `True`, the autograd system will track operations
                on the tensor.
        """
        tensor = torch.ones(
            shape,
            dtype=dtype,
            device=device,
            pin_memory=pin_memory,
            requires_grad=requires_grad,
        )
        return tensor.as_subclass(cls)

    @classmethod
    def full(
        cls,
        shape: Sequence[int],
        fill_value: Number,
        *,
        dtype: torch.dtype | None = None,
        device: DeviceLike | None = None,
        pin_memory: bool = False,
        requires_grad: bool = False,
    ) -> Self:
        """Creates a tensor filled with a specified value.

        !!! note

            See [torch.full][] for more information.

        Args:
            shape: The shape of the tensor.
            fill_value: The value to fill the tensor with.
            dtype: The desired dtype of the tensor. If `None`, the default dtype
                will be used.
            device: The desired device of the tensor. If `None`, the current
                device will be used.
            pin_memory: If `True`, the tensor will be allocated in pinned memory.
                This is only applicable for CPU tensors.
            requires_grad: If `True`, the autograd system will track operations
                on the tensor.
        """
        tensor = torch.full(
            shape,
            fill_value,
            dtype=dtype,
            device=device,
            pin_memory=pin_memory,
            requires_grad=requires_grad,
        )
        return tensor.as_subclass(cls)

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def cast(
        self,
        dtype: torch.dtype,
        *,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> Self:
        """Casts the tensor to the specified dtype.

        Args:
            dtype: The dtype to cast the tensor to.
            non_blocking: If True, the operation will be non-blocking.
            copy: If `True`, a copy of the tensor will be created even if the
                tensor is already of the specified dtype. If `False`, the
                original tensor will be returned if it is already of the
                specified dtype.
        """
        return self.to(dtype, non_blocking=non_blocking, copy=copy)  # type: ignore

    def move(
        self,
        device: DeviceLike,
        *,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> Self:
        """Moves the tensor to the specified device.

        Args:
            device: The device to move the tensor to.
            non_blocking: If `True`, the operation will be non-blocking.
            copy: If `True`, a copy of the tensor will be created even if the
                tensor is already on the specified device. If `False`, the
                original tensor will be returned if it is already on the
                specified device.
        """
        return self.to(device, non_blocking=non_blocking, copy=copy)  # type: ignore
