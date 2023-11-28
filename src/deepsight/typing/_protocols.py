##
##
##

import typing
from typing import Any, Protocol

import torch
from typing_extensions import Self

from ._types import Configs, StateDict


@typing.runtime_checkable
class Moveable(Protocol):
    """An interface for objects that can be moved to a device."""

    @property
    def device(self) -> torch.device:
        """The device this object is on."""
        ...

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> Self:
        """Move this object to the given device.

        !!! note

            This interface does not specify whether the move should be in-place
            or not. This is left to the implementer.

        Args:
            device: The device to move to.
            non_blocking: Whether to perform the move asynchronously.

        Returns:
            The object on the given device.
        """
        ...


@typing.runtime_checkable
class Configurable(Protocol):
    """An interface for objects that can be configured."""

    def get_configs(self) -> Configs:
        """Get the configuration of this object."""
        ...


@typing.runtime_checkable
class Stateful(Protocol):
    """An interface for objects that have a state."""

    def state_dict(self) -> StateDict:
        """Get the state of this object."""
        ...

    def load_state_dict(self, state_dict: StateDict) -> Any:
        """Load the state of this object."""
        ...


@typing.runtime_checkable
class Detachable(Protocol):
    """An interface for objects that can be detached from the computation graph."""

    def detach(self) -> Self:
        """Detach this object from the computation graph."""
        ...
