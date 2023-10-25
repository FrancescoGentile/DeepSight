##
##
##

import typing
from typing import Any, Protocol

import torch
from typing_extensions import Self


@typing.runtime_checkable
class Moveable(Protocol):
    """An interface for objects that can be moved to a device."""

    @property
    def device(self) -> torch.device:
        """The device this object is on."""
        ...

    def move(self, device: torch.device, non_blocking: bool = False) -> Self:
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


JSONPrimitive = (
    bool
    | int
    | float
    | str
    | None
    | dict[str, "JSONPrimitive"]
    | list["JSONPrimitive"]
    | tuple["JSONPrimitive"]
)


@typing.runtime_checkable
class Configurable(Protocol):
    """An interface for objects that can be configured."""

    def get_config(self) -> JSONPrimitive:
        """Get the configuration of this object."""
        ...


@typing.runtime_checkable
class Stateful(Protocol):
    """An interface for objects that have a state."""

    def get_state(self) -> dict[str, Any]:
        """Get the state of this object."""
        ...

    def set_state(self, state: dict[str, Any]) -> None:
        """Set the state of this object."""
        ...
